"""
Standalone megakernel prototype — zero dependencies on CompGen IR or Inductor.

Takes a PyTorch nn.Module, generates a single persistent Triton kernel
that runs all tiled ops in one launch with event-tensor synchronization.

Usage:
    python standalone_megakernel.py           # runs diamond + FFN examples
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════
# 1. Model definitions
# ═══════════════════════════════════════════════════════════════════

class Diamond(nn.Module):
    """y = (linear_a(x) + linear_b(x)).relu()"""
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.linear_a = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_b = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return (self.linear_a(x) + self.linear_b(x)).relu()


class FFN(nn.Module):
    """y = linear_down(relu(linear_up(x)))"""
    def __init__(self, in_dim=256, hidden=512, out_dim=128):
        super().__init__()
        self.linear_up = nn.Linear(in_dim, hidden, bias=False)
        self.linear_down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        return self.linear_down(torch.relu(self.linear_up(x)))


# ═══════════════════════════════════════════════════════════════════
# 2. Task DAG — pure Python dataclasses, no external IR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OpSpec:
    """One operation: name, tile grid, and Triton body for ONE tile.

    The triton_body MUST NOT do event notify/wait — the emitter
    wraps each body with event synchronization automatically.

    Variables in scope inside the body:
      - tile_id         : which tile (0..num_tiles-1)
      - data pointers    : names from buffer_names (e.g. x_ptr, w_a_ptr)
      - constexprs       : names from constexpr_names (e.g. TM, TN, K_DIM)
    """
    name: str
    tiles_m: int
    tiles_n: int
    triton_body: str


@dataclass
class TaskDAG:
    """Tile-level DAG: ordered ops with event-tensor edges.

    op_i ──event[i]──► op_{i+1}
    event[i][tile_id]: op_i notifies, op_{i+1} waits.
    """
    ops: list[OpSpec]
    total_tiles: int
    sm_count: int
    tile_shape: tuple[int, int, int]
    constexpr_names: list[str]
    buffer_names: list[str]


# ═══════════════════════════════════════════════════════════════════
# 3. DAG builders
# ═══════════════════════════════════════════════════════════════════

def build_diamond_dag(batch, in_dim, out_dim, tile_m=32, tile_n=32, tile_k=32, sm_count=2):
    """Diamond: mm_a || mm_b → add → relu.  Buffers: [x, w_a, w_b, y_a, y_b, y_add, y_out]"""
    TM, TN, TK = tile_m, tile_n, tile_k
    tiles_m = (batch + TM - 1) // TM
    tiles_n = (out_dim + TN - 1) // TN
    num_tiles = tiles_m * tiles_n

    gemm_body = """\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
acc = tl.zeros((TM, TN), dtype=tl.float32)
for k_start in range(0, K_DIM, TK):
    k_offs = k_start + tl.arange(0, TK)
    a = tl.load(x_ptr + r * K_DIM + k_offs[None, :])
    b = tl.load(w_ptr + c * K_DIM + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store(out_ptr + r * N_DIM + c, acc)
"""

    add_body = """\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store(out_ptr + idx, tl.load(a_ptr + idx) + tl.load(b_ptr + idx))
"""

    relu_body = """\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store(out_ptr + idx, tl.maximum(tl.load(in_ptr + idx), 0.0))
"""

    return TaskDAG(
        ops=[
            OpSpec("mm_a", tiles_m, tiles_n, gemm_body),
            OpSpec("mm_b", tiles_m, tiles_n, gemm_body),
            OpSpec("add",  tiles_m, tiles_n, add_body),
            OpSpec("relu", tiles_m, tiles_n, relu_body),
        ],
        total_tiles=num_tiles, sm_count=sm_count, tile_shape=(TM, TN, TK),
        constexpr_names=["TM", "TN", "TK", "TILES_PER_ROW", "K_DIM", "N_DIM"],
        buffer_names=["x", "w_a", "w_b", "y_a", "y_b", "y_add", "y_out"],
    )


def build_ffn_dag(batch, in_dim, hidden, out_dim, tile_m=32, tile_n=32, tile_k=32, sm_count=2):
    """FFN: linear_up → relu → linear_down.  Buffers: [x, w_up, w_down, y_up, y_relu, y_out]"""
    TM, TN, TK = tile_m, tile_n, tile_k
    tiles_m = (batch + TM - 1) // TM
    tiles_n_up = (hidden + TN - 1) // TN
    tiles_n_down = (out_dim + TN - 1) // TN
    num_tiles_up = tiles_m * tiles_n_up
    num_tiles_down = tiles_m * tiles_n_down

    gemm_up_body = f"""\
row_tile = tile_id // {tiles_n_up}
col_tile = tile_id % {tiles_n_up}
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
acc = tl.zeros((TM, TN), dtype=tl.float32)
for k_start in range(0, {in_dim}, TK):
    k_offs = k_start + tl.arange(0, TK)
    a = tl.load(x_ptr + r * {in_dim} + k_offs[None, :])
    b = tl.load(w_up_ptr + c * {in_dim} + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store(y_up_ptr + r * {hidden} + c, acc)
"""

    relu_body = f"""\
row_tile = tile_id // {tiles_n_up}
col_tile = tile_id % {tiles_n_up}
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * {hidden} + c
tl.store(y_relu_ptr + idx, tl.maximum(tl.load(y_up_ptr + idx), 0.0))
"""

    gemm_down_body = f"""\
row_tile = tile_id // {tiles_n_down}
col_tile = tile_id % {tiles_n_down}
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
acc = tl.zeros((TM, TN), dtype=tl.float32)
for k_start in range(0, {hidden}, TK):
    k_offs = k_start + tl.arange(0, TK)
    a = tl.load(y_relu_ptr + r * {hidden} + k_offs[None, :])
    b = tl.load(w_down_ptr + c * {hidden} + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store(y_out_ptr + r * {out_dim} + c, acc)
"""

    return TaskDAG(
        ops=[
            OpSpec("linear_up",   tiles_m, tiles_n_up,   gemm_up_body),
            OpSpec("relu",        tiles_m, tiles_n_up,   relu_body),
            OpSpec("linear_down", tiles_m, tiles_n_down, gemm_down_body),
        ],
        total_tiles=num_tiles_up, sm_count=sm_count, tile_shape=(TM, TN, TK),
        constexpr_names=["TM", "TN", "TK"],
        buffer_names=["x", "w_up", "w_down", "y_up", "y_relu", "y_out"],
    )


# ═══════════════════════════════════════════════════════════════════
# 4. Static scheduler
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScheduledTask:
    op_idx: int
    tile_id: int


def schedule_tasks(dag: TaskDAG) -> list[list[ScheduledTask]]:
    """Toposort tiles, round-robin to SMs."""
    tasks = [
        ScheduledTask(op_idx, tile_id)
        for tile_id in range(dag.total_tiles)
        for op_idx in range(len(dag.ops))
    ]
    sm_queues: list[list[ScheduledTask]] = [[] for _ in range(dag.sm_count)]
    for i, t in enumerate(tasks):
        sm_queues[i % dag.sm_count].append(t)
    return sm_queues


# ═══════════════════════════════════════════════════════════════════
# 5. Triton emitter
# ═══════════════════════════════════════════════════════════════════

def emit_megakernel_source(dag: TaskDAG, sm_queues: list[list[ScheduledTask]]) -> str:
    num_ops = len(dag.ops)
    num_tiles = dag.total_tiles
    max_tasks = max(len(q) for q in sm_queues)
    sm_count = dag.sm_count

    flat: list[ScheduledTask] = []
    for q in sm_queues:
        flat.extend(q)
        flat.extend([ScheduledTask(-1, 0)] * (max_tasks - len(q)))

    L: list[str] = []
    A = L.append
    A("# Auto-generated megakernel")
    A(f"# {' → '.join(o.name for o in dag.ops)}  ({num_tiles} tiles, {sm_count} SMs)")
    A("import triton, triton.language as tl")
    A("@triton.jit")
    A("def megakernel(")

    for name in dag.buffer_names:
        A(f"    {name}_ptr,")
    A("    E_ptr,")
    for name in sorted(dag.constexpr_names):
        A(f"    {name}: tl.constexpr,")
    A("):")
    A("    sm_id = tl.program_id(0)")
    A(f"    _OP = [{','.join(str(t.op_idx) for t in flat)}]")
    A(f"    _TID = [{','.join(str(t.tile_id) for t in flat)}]")
    A(f"    _MAX: tl.constexpr = {max_tasks}")
    A(f"    _NT: tl.constexpr = {num_tiles}")
    A(f"    _NO: tl.constexpr = {num_ops}")
    A("    for slot in range(_MAX):")
    A("        tid = sm_id * _MAX + slot; op = _OP[tid]; tile_id = _TID[tid]")

    # wait
    for oi in range(num_ops):
        if oi == 0:
            A(f"        if op == {oi}: pass")
        else:
            A(f"        if op == {oi}:")
            A(f"            ev = {oi-1} * _NT + tile_id")
            A("            while tl.atomic_or(E_ptr + ev, 0) > 0: pass")

    # dispatch
    for oi, op in enumerate(dag.ops):
        A(f"        if op == {oi}:")
        for line in op.triton_body.strip().split("\n"):
            A(f"            {line}")

    A("        if op < 0: pass")

    # notify
    for oi in range(num_ops):
        A(f"        if op == {oi}:")
        A(f"            ev = {oi} * _NT + tile_id")
        A("            tl.atomic_add(E_ptr + ev, -1)")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════
# 6. Compile & run
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CompiledMegakernel:
    kernel_source: str
    kernel_fn: Callable
    dag: TaskDAG


def compile_megakernel(dag: TaskDAG, output_dir: str | None = None) -> CompiledMegakernel:
    sm_queues = schedule_tasks(dag)
    source = emit_megakernel_source(dag, sm_queues)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "megakernel.py").write_text(source)
        print(f"  Source → {out / 'megakernel.py'}")

    if not HAS_TRITON:
        return CompiledMegakernel(source, lambda *a, **kw: None, dag)

    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("megakernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return CompiledMegakernel(source, mod.megakernel, dag)


def _benchmark_runs(
    fn: Callable[[], Any],
    warmup: int = 10,
    iters: int = 100,
    label: str = "",
) -> dict[str, float]:
    """Run fn `warmup` + `iters` times, return timing stats (ms)."""
    # warmup
    for _ in range(warmup):
        fn()

    # measure using CUDA events
    times: list[float] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    times.sort()
    return {
        "label": label,
        "median_ms": times[len(times) // 2],
        "p10_ms": times[max(0, int(0.1 * len(times)) - 1)],
        "p90_ms": times[min(len(times) - 1, int(0.9 * len(times)))],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def run_and_check(
    compiled: CompiledMegakernel,
    buffers: list[torch.Tensor],
    eager_out: torch.Tensor,
    constexprs: dict[str, int],
):
    num_ops = len(compiled.dag.ops)
    n_events = num_ops * compiled.dag.total_tiles
    E = torch.full((n_events,), 1, dtype=torch.int32, device="cuda")

    compiled.kernel_fn[(compiled.dag.sm_count,)](*buffers, E, **constexprs)
    torch.cuda.synchronize()

    err = (buffers[-1] - eager_out).abs().max().item()
    return err


# ═══════════════════════════════════════════════════════════════════
# 7. Benchmark: megakernel vs eager vs torch.compile
# ═══════════════════════════════════════════════════════════════════

def benchmark_diamond(
    in_dim: int = 256,
    out_dim: int = 128,
    batch: int = 1,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    sm_count: int = 2,
    warmup: int = 10,
    iters: int = 100,
):
    """Compare megakernel vs eager vs torch.compile for Diamond."""
    TM, TN, TK = tile_m, tile_n, tile_k

    print(f"\n{'='*60}")
    print(f"Benchmark: Diamond  B={batch} IN={in_dim} OUT={out_dim}  TM={TM} TN={TN} TK={TK}")
    print(f"{'='*60}")

    model = Diamond(in_dim, out_dim).eval()
    x = torch.randn(batch, in_dim)

    # ── build & compile megakernel ──
    dag_d = build_diamond_dag(batch, in_dim, out_dim, TM, TN, TK, sm_count)
    compiled_d = compile_megakernel(dag_d)  # no output_dir for benchmark

    if not HAS_TRITON or not torch.cuda.is_available():
        print("  [skip] Triton or CUDA not available")
        return

    # Move to GPU
    model = model.cuda()
    x = x.cuda()

    tiles_per_row = (out_dim + TN - 1) // TN
    constexprs_d = {"TM": TM, "TN": TN, "TK": TK,
                    "TILES_PER_ROW": tiles_per_row,
                    "K_DIM": in_dim, "N_DIM": out_dim}
    n_events = len(dag_d.ops) * dag_d.total_tiles

    # ── correctness check first ──
    with torch.no_grad():
        eager_out = model(x)
    buffers = [
        x,
        model.linear_a.weight.data,
        model.linear_b.weight.data,
        torch.zeros(batch, out_dim, device="cuda"),
        torch.zeros(batch, out_dim, device="cuda"),
        torch.zeros(batch, out_dim, device="cuda"),
        torch.zeros(batch, out_dim, device="cuda"),
    ]
    err = run_and_check(compiled_d, [b.clone() for b in buffers], eager_out, constexprs_d)
    print(f"  correctness: max error = {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")

    # ── Eager benchmark ──
    def eager_fn():
        with torch.no_grad():
            return model(x)

    stats_eager = _benchmark_runs(eager_fn, warmup, iters, "Eager")

    # ── torch.compile benchmark ──
    model_compiled = torch.compile(Diamond(in_dim, out_dim).eval().cuda(),
                                   backend="inductor", mode="reduce-overhead")
    # warm torch.compile separately
    for _ in range(warmup):
        with torch.no_grad():
            model_compiled(x)

    def compile_fn():
        with torch.no_grad():
            return model_compiled(x)

    stats_compile = _benchmark_runs(compile_fn, 0, iters, "torch.compile")

    # ── Megakernel benchmark ──
    # Each iter: reset event tensor + buffers, then launch
    E = torch.full((n_events,), 1, dtype=torch.int32, device="cuda")

    def mk_fn():
        E.fill_(1)
        for b in buffers[3:]:
            b.zero_()
        compiled_d.kernel_fn[(sm_count,)](*buffers, E, **constexprs_d)

    # warmup
    for _ in range(warmup):
        mk_fn()
    stats_mk = _benchmark_runs(mk_fn, 0, iters, "Megakernel")

    # ── Print results ──
    print(f"\n  {'Method':<20} {'Median':>10} {'P10':>10} {'P90':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*60}")
    for s in [stats_eager, stats_compile, stats_mk]:
        print(f"  {s['label']:<20} {s['median_ms']:>8.4f}ms {s['p10_ms']:>8.4f}ms "
              f"{s['p90_ms']:>8.4f}ms {s['min_ms']:>8.4f}ms {s['max_ms']:>8.4f}ms")

    speedup_vs_eager = stats_eager["median_ms"] / stats_mk["median_ms"]
    speedup_vs_compile = stats_compile["median_ms"] / stats_mk["median_ms"]
    print(f"\n  Megakernel vs Eager:        {speedup_vs_eager:.2f}x")
    print(f"  Megakernel vs torch.compile: {speedup_vs_compile:.2f}x")

    return {"eager": stats_eager, "compile": stats_compile, "megakernel": stats_mk}


def benchmark_ffn(
    in_dim: int = 256,
    hidden: int = 512,
    out_dim: int = 128,
    batch: int = 1,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    sm_count: int = 2,
    warmup: int = 10,
    iters: int = 100,
):
    """Compare megakernel vs eager vs torch.compile for FFN."""
    TM, TN, TK = tile_m, tile_n, tile_k

    print(f"\n{'='*60}")
    print(f"Benchmark: FFN  B={batch} IN={in_dim} HIDDEN={hidden} OUT={out_dim}  TM={TM} TN={TN} TK={TK}")
    print(f"{'='*60}")

    model = FFN(in_dim, hidden, out_dim).eval()
    x = torch.randn(batch, in_dim)

    dag_f = build_ffn_dag(batch, in_dim, hidden, out_dim, TM, TN, TK, sm_count)
    compiled_f = compile_megakernel(dag_f)

    if not HAS_TRITON or not torch.cuda.is_available():
        print("  [skip] Triton or CUDA not available")
        return

    model = model.cuda()
    x = x.cuda()

    constexprs_f = {"TM": TM, "TN": TN, "TK": TK}
    n_events = len(dag_f.ops) * dag_f.total_tiles

    with torch.no_grad():
        eager_out = model(x)
    buffers = [
        x,
        model.linear_up.weight.data,
        model.linear_down.weight.data,
        torch.zeros(batch, hidden, device="cuda"),
        torch.zeros(batch, hidden, device="cuda"),
        torch.zeros(batch, out_dim, device="cuda"),
    ]
    err = run_and_check(compiled_f, [b.clone() for b in buffers], eager_out, constexprs_f)
    print(f"  correctness: max error = {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")

    # Eager
    def eager_fn():
        with torch.no_grad():
            return model(x)
    stats_eager = _benchmark_runs(eager_fn, warmup, iters, "Eager")

    # torch.compile
    model_compiled = torch.compile(FFN(in_dim, hidden, out_dim).eval().cuda(),
                                   backend="inductor", mode="reduce-overhead")
    for _ in range(warmup):
        with torch.no_grad():
            model_compiled(x)

    def compile_fn():
        with torch.no_grad():
            return model_compiled(x)
    stats_compile = _benchmark_runs(compile_fn, 0, iters, "torch.compile")

    # Megakernel
    E = torch.full((n_events,), 1, dtype=torch.int32, device="cuda")
    def mk_fn():
        E.fill_(1)
        for b in buffers[3:]:
            b.zero_()
        compiled_f.kernel_fn[(sm_count,)](*buffers, E, **constexprs_f)

    for _ in range(warmup):
        mk_fn()
    stats_mk = _benchmark_runs(mk_fn, 0, iters, "Megakernel")

    print(f"\n  {'Method':<20} {'Median':>10} {'P10':>10} {'P90':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*60}")
    for s in [stats_eager, stats_compile, stats_mk]:
        print(f"  {s['label']:<20} {s['median_ms']:>8.4f}ms {s['p10_ms']:>8.4f}ms "
              f"{s['p90_ms']:>8.4f}ms {s['min_ms']:>8.4f}ms {s['max_ms']:>8.4f}ms")

    speedup_vs_eager = stats_eager["median_ms"] / stats_mk["median_ms"]
    speedup_vs_compile = stats_compile["median_ms"] / stats_mk["median_ms"]
    print(f"\n  Megakernel vs Eager:        {speedup_vs_eager:.2f}x")
    print(f"  Megakernel vs torch.compile: {speedup_vs_compile:.2f}x")

    return {"eager": stats_eager, "compile": stats_compile, "megakernel": stats_mk}


# ═══════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--correctness-only", action="store_true",
                   help="Only check correctness (no benchmark)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--in-dim", type=int, default=256)
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    args = p.parse_args()

    if args.correctness_only:
        # Quick correctness check
        B, IN, OUT, HIDDEN = args.batch, args.in_dim, args.out_dim, args.hidden

        print("Correctness check: Diamond")
        dag_d = build_diamond_dag(B, IN, OUT, sm_count=2)
        compiled_d = compile_megakernel(dag_d)
        if HAS_TRITON and torch.cuda.is_available():
            model = Diamond(IN, OUT).eval().cuda()
            x = torch.randn(B, IN, device="cuda")
            buffers = [x, model.linear_a.weight.data, model.linear_b.weight.data,
                       torch.zeros(B, OUT, device="cuda"), torch.zeros(B, OUT, device="cuda"),
                       torch.zeros(B, OUT, device="cuda"), torch.zeros(B, OUT, device="cuda")]
            with torch.no_grad():
                eager = model(x)
            tiles_pr = (OUT + 32 - 1) // 32
            err = run_and_check(compiled_d, buffers, eager,
                               {"TM": 32, "TN": 32, "TK": 32, "TILES_PER_ROW": tiles_pr,
                                "K_DIM": IN, "N_DIM": OUT})
            print(f"  error: {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")

        print("Correctness check: FFN")
        dag_f = build_ffn_dag(B, IN, HIDDEN, OUT, sm_count=2)
        compiled_f = compile_megakernel(dag_f)
        if HAS_TRITON and torch.cuda.is_available():
            model = FFN(IN, HIDDEN, OUT).eval().cuda()
            x = torch.randn(B, IN, device="cuda")
            buffers = [x, model.linear_up.weight.data, model.linear_down.weight.data,
                       torch.zeros(B, HIDDEN, device="cuda"), torch.zeros(B, HIDDEN, device="cuda"),
                       torch.zeros(B, OUT, device="cuda")]
            with torch.no_grad():
                eager = model(x)
            err = run_and_check(compiled_f, buffers, eager, {"TM": 32, "TN": 32, "TK": 32})
            print(f"  error: {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")
    else:
        benchmark_diamond(
            in_dim=args.in_dim, out_dim=args.out_dim, batch=args.batch,
            warmup=args.warmup, iters=args.iters,
        )
        benchmark_ffn(
            in_dim=args.in_dim, hidden=args.hidden, out_dim=args.out_dim,
            batch=args.batch, warmup=args.warmup, iters=args.iters,
        )


if __name__ == "__main__":
    main()
