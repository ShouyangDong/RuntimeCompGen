"""
Triton Megakernel API — mirrors CompGen's ``compile_to_megakernel``.

Usage:
    from triton_megakernel_api import compile_to_triton_megakernel
    bundle = compile_to_triton_megakernel(model, sample_inputs)
    result = bundle.dispatch(x)
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


class UnsupportedModel(RuntimeError):
    """No pattern matched the model."""

@dataclass
class MatchedOp:
    name: str
    triton_body: str
    tiles: int
    reads: list[str]
    writes: list[str]

@dataclass
class MatchedDAG:
    ops: list[MatchedOp]
    buffer_names: list[str]
    constexpr_names: list[str]
    total_tiles: int
    sm_count: int
    pattern_name: str


def match_model(model: nn.Module, batch: int, sm_count=2,
                tile_m=32, tile_n=32, tile_k=32) -> MatchedDAG:
    TM, TN, TK = tile_m, tile_n, tile_k
    dag = _try_diamond(model, batch, TM, TN, TK, sm_count)
    if dag: return dag
    dag = _try_ffn(model, batch, TM, TN, TK, sm_count)
    if dag: return dag
    dag = _try_elementwise_chain(model, batch, TM, sm_count)
    if dag: return dag
    children = ", ".join(f"{n}={type(m).__name__}" for n, m in model.named_children())
    raise UnsupportedModel(f"{type(model).__name__} didn't match. Children: {children}")


def _try_diamond(model, batch, TM, TN, TK, sm_count) -> MatchedDAG | None:
    linears = [(n, m) for n, m in model.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != 2: return None
    (na, la), (nb, lb) = linears
    if la.in_features != lb.in_features or la.out_features != lb.out_features: return None
    if la.bias is not None or lb.bias is not None: return None

    in_dim, out_dim = la.in_features, la.out_features
    device = la.weight.device
    x = torch.randn(batch, in_dim, device=device)
    with torch.no_grad():
        model.eval()
        actual = model(x).cpu()
        expected = (la(x) + lb(x)).relu().cpu()
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4): return None

    tiles_m = (batch + TM - 1) // TM
    tiles_n = (out_dim + TN - 1) // TN
    num_tiles = tiles_m * tiles_n

    def _gemm(xn, wn, on):
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
    a = tl.load({xn}_ptr + r * K_DIM + k_offs[None, :])
    b = tl.load({wn}_ptr + c * K_DIM + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store({on}_ptr + r * N_DIM + c, acc)
"""

    def _add(an, bn, on):
        return f"""\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store({on}_ptr + idx, tl.load({an}_ptr + idx) + tl.load({bn}_ptr + idx))
"""

    def _relu(inn, on):
        return f"""\
row_tile = tile_id // TILES_PER_ROW
col_tile = tile_id % TILES_PER_ROW
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * N_DIM + c
tl.store({on}_ptr + idx, tl.maximum(tl.load({inn}_ptr + idx), 0.0))
"""

    return MatchedDAG(
        ops=[
            MatchedOp("mm_a", _gemm("x","w_a","y_a"), num_tiles, reads=["x","w_a"], writes=["y_a"]),
            MatchedOp("mm_b", _gemm("x","w_b","y_b"), num_tiles, reads=["x","w_b"], writes=["y_b"]),
            MatchedOp("add",  _add("y_a","y_b","y_add"), num_tiles, reads=["y_a","y_b"], writes=["y_add"]),
            MatchedOp("relu", _relu("y_add","y_out"), num_tiles, reads=["y_add"], writes=["y_out"]),
        ],
        buffer_names=["x","w_a","w_b","y_a","y_b","y_add","y_out"],
        constexpr_names=["TM","TN","TK","TILES_PER_ROW","K_DIM","N_DIM"],
        total_tiles=num_tiles, sm_count=sm_count, pattern_name="diamond",
    )


def _try_ffn(model, batch, TM, TN, TK, sm_count) -> MatchedDAG | None:
    linears = [(n, m) for n, m in model.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != 2: return None
    (n_up, lu), (n_down, ld) = linears
    if lu.bias is not None or ld.bias is not None: return None
    if lu.out_features != ld.in_features: return None

    in_dim, hidden, out_dim = lu.in_features, lu.out_features, ld.out_features
    device = lu.weight.device
    x = torch.randn(batch, in_dim, device=device)
    with torch.no_grad():
        model.eval()
        actual = model(x).cpu()
        expected = ld(torch.relu(lu(x))).cpu()
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4): return None

    tiles_m = (batch + TM - 1) // TM
    tiles_up_n = (hidden + TN - 1) // TN
    tiles_down_n = (out_dim + TN - 1) // TN
    num_tiles = tiles_m * tiles_up_n

    def _gemm(name, xb, wb, ob, kd, nd, tnr):
        return f"""\
row_tile = tile_id // {tnr}
col_tile = tile_id % {tnr}
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
acc = tl.zeros((TM, TN), dtype=tl.float32)
for k_start in range(0, {kd}, TK):
    k_offs = k_start + tl.arange(0, TK)
    a = tl.load({xb}_ptr + r * {kd} + k_offs[None, :])
    b = tl.load({wb}_ptr + c * {kd} + k_offs[None, :])
    acc += tl.dot(a, tl.trans(b))
tl.store({ob}_ptr + r * {nd} + c, acc)
"""

    relu_body = f"""\
row_tile = tile_id // {tiles_up_n}
col_tile = tile_id % {tiles_up_n}
row_start = row_tile * TM
col_start = col_tile * TN
r = row_start + tl.arange(0, TM)[:, None]
c = col_start + tl.arange(0, TN)[None, :]
idx = r * {hidden} + c
tl.store(y_relu_ptr + idx, tl.maximum(tl.load(y_up_ptr + idx), 0.0))
"""

    return MatchedDAG(
        ops=[
            MatchedOp("linear_up", _gemm("up","x","w_up","y_up",in_dim,hidden,tiles_up_n),
                      num_tiles, reads=["x","w_up"], writes=["y_up"]),
            MatchedOp("relu", relu_body, num_tiles, reads=["y_up"], writes=["y_relu"]),
            MatchedOp("linear_down", _gemm("down","y_relu","w_down","y_out",hidden,out_dim,tiles_down_n),
                      num_tiles, reads=["y_relu","w_down"], writes=["y_out"]),
        ],
        buffer_names=["x","w_up","w_down","y_up","y_relu","y_out"],
        constexpr_names=["TM","TN","TK"],
        total_tiles=num_tiles, sm_count=sm_count, pattern_name="ffn",
    )


def _try_elementwise_chain(model, batch, TM, sm_count) -> MatchedDAG | None:
    if any(isinstance(m, nn.Linear) for m in model.modules()): return None
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception:
        return None
    nodes = [n for n in gm.graph.nodes if n.op in ("call_function","call_module")]
    if len(nodes) < 2: return None
    tiles = (batch + TM - 1) // TM
    ops: list[MatchedOp] = []
    for i, node in enumerate(nodes):
        in_buf = "x" if i == 0 else f"buf{i}"
        out_buf = f"buf{i+1}" if i < len(nodes)-1 else "y_out"
        body = f"""\
row_start = tile_id * TM
r = row_start + tl.arange(0, TM)
mask = r < B
v = tl.load({in_buf}_ptr + r, mask=mask)
tl.store({out_buf}_ptr + r, v, mask=mask)"""
        ops.append(MatchedOp(node.name.replace(".","_"), body, tiles,
                             reads=[in_buf], writes=[out_buf]))
    return MatchedDAG(
        ops=ops,
        buffer_names=["x"] + [f"buf{i+1}" for i in range(len(nodes)-1)] + ["y_out"],
        constexpr_names=["B","TM"], total_tiles=tiles,
        sm_count=sm_count, pattern_name="elementwise_chain",
    )


# ═══════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _ScheduledTask:
    op_idx: int
    tile_id: int

def _schedule(dag: MatchedDAG) -> list[list[_ScheduledTask]]:
    tasks = [_ScheduledTask(oi, tid)
             for tid in range(dag.total_tiles)
             for oi in range(len(dag.ops))]
    queues: list[list[_ScheduledTask]] = [[] for _ in range(dag.sm_count)]
    for i, t in enumerate(tasks):
        queues[i % dag.sm_count].append(t)
    return queues

# ═══════════════════════════════════════════════════════════════════
# Triton emitter
# ═══════════════════════════════════════════════════════════════════

def _emit(dag: MatchedDAG, sm_queues: list[list[_ScheduledTask]]) -> str:
    num_ops = len(dag.ops)
    num_tiles = dag.total_tiles
    max_qlen = max(len(q) for q in sm_queues)
    L: list[str] = []
    A = L.append
    A("# Auto-generated persistent Triton megakernel")
    A(f"# Pattern: {dag.pattern_name}  {' → '.join(o.name for o in dag.ops)}")
    A("import triton, triton.language as tl")
    A("@triton.jit")
    A("def megakernel(")
    for name in dag.buffer_names:
        A(f"    {name}_ptr,")
    A("    E_ptr, QUEUE_PTR, QUEUE_LEN_PTR,")
    for name in dag.constexpr_names:
        A(f"    {name}: tl.constexpr,")
    A("):")
    A("    sm_id = tl.program_id(0)")
    A(f"    _N_OPS:    tl.constexpr = {num_ops}")
    A(f"    _N_TILES:  tl.constexpr = {num_tiles}")
    A(f"    _MAX_QLEN: tl.constexpr = {max_qlen}")
    A("    qlen = tl.load(QUEUE_LEN_PTR + sm_id)")
    A("    task_idx = 0")
    A("    while task_idx < qlen:")
    A("        base = (sm_id * _MAX_QLEN + task_idx) * 2")
    A("        op      = tl.load(QUEUE_PTR + base + 0)")
    A("        tile_id = tl.load(QUEUE_PTR + base + 1)")
    for oi in range(num_ops):
        if oi == 0:
            A(f"        if op == {oi}: pass")
        else:
            A(f"        if op == {oi}:")
            A(f"            ev = {oi-1} * _N_TILES + tile_id")
            A("            while tl.atomic_or(E_ptr + ev, 0) > 0: pass")
    A("")
    for oi, op in enumerate(dag.ops):
        A(f"        if op == {oi}:  # {op.name}")
        for line in op.triton_body.strip().split("\n"):
            A(f"            {line}")
        A("")
    A("        if op < 0: pass")
    A("")
    for oi in range(num_ops):
        A(f"        if op == {oi}:")
        A(f"            ev = {oi} * _N_TILES + tile_id")
        A("            tl.atomic_add(E_ptr + ev, -1)")
    A("        task_idx += 1")
    return "\n".join(L)


def _build_queue_tensors(sm_queues, sm_count, device):
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
# Public API
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TritonMegakernelBundle:
    kernel_source: str
    kernel_name: str
    dag: MatchedDAG
    sm_queues: list[list[_ScheduledTask]]
    output_dir: Path | None = None
    kernel_fn: Callable | None = None

    def dispatch(self, *args: torch.Tensor, **constexprs) -> torch.Tensor:
        if self.kernel_fn is None:
            raise RuntimeError("Kernel not compiled (Triton unavailable)")
        device = args[0].device
        x = args[0]
        queue, lens = _build_queue_tensors(self.sm_queues, self.dag.sm_count, device)
        n_events = len(self.dag.ops) * self.dag.total_tiles
        E = torch.full((n_events,), 1, dtype=torch.int32, device=device)

        buf_dict: dict[str, torch.Tensor] = {"x": x}
        for i, name in enumerate(self.dag.buffer_names[1:], start=1):
            if i < len(args):
                buf_dict[name] = args[i]
        for name in self.dag.buffer_names:
            if name not in buf_dict:
                buf_dict[name] = torch.zeros(x.shape, dtype=x.dtype, device=device)

        buffers = [buf_dict[n] for n in self.dag.buffer_names]
        self.kernel_fn[(self.dag.sm_count,)](*buffers, E, queue, lens, **constexprs)
        torch.cuda.synchronize()
        return buffers[-1]


def compile_to_triton_megakernel(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    sm_count: int = 2,
    tile_m: int = 32,
    tile_n: int = 32,
    tile_k: int = 32,
    output_dir: str | Path | None = None,
) -> TritonMegakernelBundle:
    x = sample_inputs[0]
    batch = x.shape[0]

    dag = match_model(model, batch, sm_count=sm_count,
                      tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    sm_queues = _schedule(dag)
    source = _emit(dag, sm_queues)

    if output_dir:
        p = Path(output_dir) / "megakernel.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source)

    kernel_fn: Callable | None = None
    if HAS_TRITON:
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(source)
        spec = importlib.util.spec_from_file_location("megakernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_fn = mod.megakernel

    return TritonMegakernelBundle(
        kernel_source=source, kernel_name="megakernel",
        dag=dag, sm_queues=sm_queues,
        output_dir=Path(output_dir) if output_dir else None,
        kernel_fn=kernel_fn,
    )
