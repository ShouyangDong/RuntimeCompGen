"""
Complete example: PyTorch nn.Module → Triton Megakernel.

Single-file, zero dependencies beyond torch + triton.
Maps `Diamond(x) = (linear_a(x) + linear_b(x)).relu()` onto a
single persistent Triton kernel with tile-level event-tensor sync.

Usage:
    python torch_to_megakernel.py
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ═══════════════════════════════════════════════════════════════════
# Step 1: PyTorch model
# ═══════════════════════════════════════════════════════════════════

class Diamond(nn.Module):
    """y = (linear_a(x) + linear_b(x)).relu()"""
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.linear_a = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_b = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (self.linear_a(x) + self.linear_b(x)).relu()

# ═══════════════════════════════════════════════════════════════════
# Step 2: Build tile-level Task DAG from model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OpSpec:
    name: str
    tiles_m: int
    tiles_n: int
    triton_body: str  # Triton source for ONE tile

@dataclass
class TaskDAG:
    ops: list[OpSpec]
    total_tiles: int
    sm_count: int
    constexpr_names: list[str]
    buffer_names: list[str]

def build_dag_from_diamond(model: Diamond, batch: int,
                           tile_m=32, tile_n=32, tile_k=32,
                           sm_count=2) -> TaskDAG:
    in_dim = model.linear_a.in_features
    out_dim = model.linear_a.out_features
    TM, TN, TK = tile_m, tile_n, tile_k
    tiles_m = (batch + TM - 1) // TM
    tiles_n = (out_dim + TN - 1) // TN
    num_tiles = tiles_m * tiles_n

    def _gemm_body(x_name: str, w_name: str, out_name: str) -> str:
        """Tiled GEMM body with specific pointer names."""
        return f"""\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
acc = tl.zeros((TM, TN), dtype=tl.float32)
for k_start in range(0, K_DIM, TK):
    k_offs = k_start + tl.arange(0, TK)
    a = tl.load({x_name}_ptr + r * K_DIM + k_offs[None, :])
    b = tl.load({w_name}_ptr + c * K_DIM + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store({out_name}_ptr + r * N_DIM + c, acc)
"""

    def _add_body(a_name: str, b_name: str, out_name: str) -> str:
        return f"""\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store({out_name}_ptr + idx, tl.load({a_name}_ptr + idx) + tl.load({b_name}_ptr + idx))
"""

    def _relu_body(in_name: str, out_name: str) -> str:
        return f"""\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store({out_name}_ptr + idx, tl.maximum(tl.load({in_name}_ptr + idx), 0.0))
"""

    return TaskDAG(
        ops=[
            OpSpec("mm_a",  tiles_m, tiles_n, _gemm_body("x", "w_a", "y_a")),
            OpSpec("mm_b",  tiles_m, tiles_n, _gemm_body("x", "w_b", "y_b")),
            OpSpec("add",   tiles_m, tiles_n, _add_body("y_a", "y_b", "y_add")),
            OpSpec("relu",  tiles_m, tiles_n, _relu_body("y_add", "y_out")),
        ],
        total_tiles=num_tiles, sm_count=sm_count,
        constexpr_names=["TM", "TN", "TK", "TILES_PER_ROW", "K_DIM", "N_DIM"],
        buffer_names=["x", "w_a", "w_b", "y_a", "y_b", "y_add", "y_out"],
    )

# ═══════════════════════════════════════════════════════════════════
# Step 3: Static scheduler
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScheduledTask:
    op_idx: int
    tile_id: int

def schedule(dag: TaskDAG) -> list[list[ScheduledTask]]:
    tasks = [ScheduledTask(oi, tid)
             for tid in range(dag.total_tiles)
             for oi in range(len(dag.ops))]
    queues: list[list[ScheduledTask]] = [[] for _ in range(dag.sm_count)]
    for i, t in enumerate(tasks):
        queues[i % dag.sm_count].append(t)
    return queues

# ═══════════════════════════════════════════════════════════════════
# Step 4: Triton emitter — uses QUEUE_PTR (runtime tensor) for task info
# ═══════════════════════════════════════════════════════════════════

def emit(dag: TaskDAG, sm_queues: list[list[ScheduledTask]]) -> str:
    num_ops = len(dag.ops)
    num_tiles = dag.total_tiles
    max_qlen = max(len(q) for q in sm_queues)

    L: list[str] = []
    A = L.append
    A("# Auto-generated persistent Triton megakernel")
    A(f"# Pattern: {' → '.join(o.name for o in dag.ops)}")
    A(f"# Tiles: {num_tiles}  SMs: {dag.sm_count}  Max qlen: {max_qlen}")
    A("import triton, triton.language as tl")
    A("@triton.jit")
    A("def megakernel(")
    for name in dag.buffer_names:
        A(f"    {name}_ptr,")
    A("    E_ptr,")
    A("    QUEUE_PTR,          # (SM_COUNT, MAX_QLEN, 2) int32")
    A("    QUEUE_LEN_PTR,      # (SM_COUNT,) int32")
    for name in dag.constexpr_names:
        A(f"    {name}: tl.constexpr,")
    A("):")
    A("    sm_id = tl.program_id(0)")
    A(f"    _N_OPS:    tl.constexpr = {num_ops}")
    A(f"    _N_TILES:  tl.constexpr = {num_tiles}")
    A(f"    _MAX_QLEN: tl.constexpr = {max_qlen}")
    A("")
    A("    qlen = tl.load(QUEUE_LEN_PTR + sm_id)")
    A("    task_idx = 0")
    A("    while task_idx < qlen:")
    A("        base = (sm_id * _MAX_QLEN + task_idx) * 2")
    A("        op      = tl.load(QUEUE_PTR + base + 0)")
    A("        tile_id = tl.load(QUEUE_PTR + base + 1)")
    A("")

    # wait
    for oi in range(num_ops):
        if oi == 0:
            A(f"        if op == {oi}: pass")
        else:
            A(f"        if op == {oi}:")
            A(f"            ev = {oi-1} * _N_TILES + tile_id")
            A("            while tl.atomic_or(E_ptr + ev, 0) > 0: pass")
    A("")

    # dispatch
    for oi, op_spec in enumerate(dag.ops):
        A(f"        if op == {oi}:  # {op_spec.name}")
        for line in op_spec.triton_body.strip().split("\n"):
            A(f"            {line}")
        A("")
    A("        if op < 0: pass")
    A("")

    # notify
    for oi in range(num_ops):
        A(f"        if op == {oi}:")
        A(f"            ev = {oi} * _N_TILES + tile_id")
        A("            tl.atomic_add(E_ptr + ev, -1)")
    A("")
    A("        task_idx += 1")
    return "\n".join(L)


def _build_queue_tensors(sm_queues: list[list[ScheduledTask]],
                         sm_count: int, device: torch.device):
    max_qlen = max(len(q) for q in sm_queues)
    q = torch.zeros((sm_count, max_qlen, 2), dtype=torch.int32, device=device)
    ln = torch.zeros((sm_count,), dtype=torch.int32, device=device)
    for sm, tasks in enumerate(sm_queues):
        for slot, t in enumerate(tasks):
            q[sm, slot, 0] = t.op_idx
            q[sm, slot, 1] = t.tile_id
        ln[sm] = len(tasks)
    return q, ln

# ═══════════════════════════════════════════════════════════════════
# Step 5: Compile
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Megakernel:
    source: str
    fn: Callable
    dag: TaskDAG
    sm_queues: list[list[ScheduledTask]]

def compile_dag(dag: TaskDAG, output_dir: str | None = None) -> Megakernel:
    sm_queues = schedule(dag)
    source = emit(dag, sm_queues)
    if output_dir:
        p = Path(output_dir) / "megakernel.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source)
        print(f"  Source → {p}")
    if not HAS_TRITON:
        return Megakernel(source, lambda *a, **kw: None, dag, sm_queues)
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("megakernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Megakernel(source, mod.megakernel, dag, sm_queues)

# ═══════════════════════════════════════════════════════════════════
# Step 6: Benchmark
# ═══════════════════════════════════════════════════════════════════

def benchmark(mk: Megakernel, model: Diamond, batch: int,
              warmup=10, iters=100):
    in_dim = model.linear_a.in_features
    out_dim = model.linear_a.out_features
    TM = TN = TK = 32
    tiles_per_row = (out_dim + TN - 1) // TN
    constexprs = {"TM": TM, "TN": TN, "TK": TK,
                  "TILES_PER_ROW": tiles_per_row,
                  "K_DIM": in_dim, "N_DIM": out_dim}

    model = model.cuda()
    device = model.linear_a.weight.device
    x = torch.randn(batch, in_dim, device=device)

    buffers = [
        x, model.linear_a.weight.data, model.linear_b.weight.data,
        torch.zeros(batch, out_dim, device=device),
        torch.zeros(batch, out_dim, device=device),
        torch.zeros(batch, out_dim, device=device),
        torch.zeros(batch, out_dim, device=device),
    ]
    queue, lens = _build_queue_tensors(mk.sm_queues, mk.dag.sm_count, device)
    n_events = len(mk.dag.ops) * mk.dag.total_tiles
    E = torch.full((n_events,), 1, dtype=torch.int32, device=device)

    # correctness
    with torch.no_grad():
        eager_out = model(x)
    mk.fn[(mk.dag.sm_count,)](*buffers, E, queue, lens, **constexprs)
    torch.cuda.synchronize()
    err = (buffers[-1] - eager_out).abs().max().item()
    print(f"  Correctness: max error = {err:.2e}  {'ok' if err < 1e-3 else 'FAIL'}")

    # timing
    def time_it(fn, wu, n, lbl):
        for _ in range(wu): fn()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            se.record(); fn(); ee.record()
            torch.cuda.synchronize()
            ts.append(se.elapsed_time(ee))
        ts.sort()
        return {"label": lbl, "med": ts[len(ts)//2], "min": ts[0], "max": ts[-1]}

    t_eager = time_it(lambda: model(x) if not torch.is_grad_enabled() else None,
                      warmup, iters, "Eager")
    with torch.no_grad():
        t_eager = time_it(lambda: model(x), warmup, iters, "Eager")

    mc = torch.compile(Diamond(in_dim, out_dim).eval().cuda(),
                       backend="inductor", mode="reduce-overhead")
    for _ in range(warmup):
        with torch.no_grad(): mc(x)
    t_comp = time_it(lambda: mc(x) if not torch.is_grad_enabled() else None,
                     0, iters, "torch.compile")
    with torch.no_grad():
        t_comp = time_it(lambda: mc(x), 0, iters, "torch.compile")

    def mk_fn():
        E.fill_(1)
        for b in buffers[3:]: b.zero_()
        mk.fn[(mk.dag.sm_count,)](*buffers, E, queue, lens, **constexprs)
    for _ in range(warmup): mk_fn()
    t_mk = time_it(mk_fn, 0, iters, "Megakernel")

    print(f"\n  {'Method':<16} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*40}")
    for t in [t_eager, t_comp, t_mk]:
        print(f"  {t['label']:<16} {t['med']:>7.4f}ms {t['min']:>7.4f}ms {t['max']:>7.4f}ms")
    print(f"\n  Speedup vs Eager:        {t_eager['med']/t_mk['med']:.2f}x")
    print(f"  Speedup vs torch.compile: {t_comp['med']/t_mk['med']:.2f}x")

# ═══════════════════════════════════════════════════════════════════

def main():
    B, IN, OUT = 1, 256, 128
    print("=" * 55)
    print(f"PyTorch → Triton Megakernel: Diamond  B={B} IN={IN} OUT={OUT}")
    print("=" * 55)

    model = Diamond(IN, OUT)
    dag = build_dag_from_diamond(model, B, sm_count=2)
    print(f"\n  Task DAG: {' → '.join(o.name for o in dag.ops)}")
    print(f"  Tiles: {dag.total_tiles}  SMs: {dag.sm_count}")

    mk = compile_dag(dag, output_dir="megakernel_output")
    print(f"  Kernel source: {len(mk.source)} chars")

    if HAS_TRITON and torch.cuda.is_available():
        benchmark(mk, model, B)
    else:
        print("\n  [skip] Triton or CUDA not available")
        print(f"  Generated source: megakernel_output/megakernel.py")

    print("\nDone.")

if __name__ == "__main__":
    main()
