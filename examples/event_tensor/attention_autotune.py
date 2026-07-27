"""Autotuned attention megakernel (locality-aware + autotune + search).

Companion to ``attention_megakernel.py``.  Uses locality-aware event ops
(``_event_notify`` / ``_event_wait``) and adds two autotuning modes:

    1. ``compile_attention_megakernel_autotune`` — ``@triton.autotune``
       sweeps ``num_warps`` × ``num_stages``.

    2. ``search_attention_megakernel`` — outer-loop search over
       ``q_tile_size`` + launch params, re-building IR per config.

Run as::

    python examples/event_tensor/attention_autotune.py
"""

from __future__ import annotations

import importlib.util
import linecache
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Auto-detect accelerator: MLU > CUDA
_HAS_MLU = hasattr(torch, "mlu") and torch.mlu.is_available()
_HAS_CUDA = torch.cuda.is_available()
_HAS_ACCEL = _HAS_MLU or _HAS_CUDA
_ACCEL_DEVICE = "mlu" if _HAS_MLU else "cuda"
_ACCEL_TAG = "MLU" if _HAS_MLU else "GPU"


def _accel_sync() -> None:
    if _HAS_MLU:
        torch.mlu.synchronize()
    elif _HAS_CUDA:
        torch.cuda.synchronize()


from xdsl.dialects.builtin import (
    ArrayAttr, IntegerAttr, IntegerType, ModuleOp, StringAttr, SymbolRefAttr,
)
from xdsl.ir import Block, Region

from compgen.ir.event.attrs import EventCoordAttr, EventTensorTypeAttr
from compgen.ir.event.ops import CallDeviceOp, EventTensorOp, GraphOp
from compgen.ir.payload.passes.megakernel_static_schedule import (
    StaticMegakernelSchedule,
)
from compgen.ir.tile.lower_megakernel import (
    DeviceFunctionSpec,
    MegakernelLoweringResult,
    MegakernelLoweringSpec,
    lower_megakernel,
)

# Reuse IR builder, Compiled* dataclass, and reference from the base file.
from examples.event_tensor.attention_megakernel import (
    CompiledAttentionMegakernel,
    _flatten_queue,
    build_attention_event_graph,
    reference_attention,
)


# ---------------------------------------------------------------------------
# MLU/CUDA-compatible launcher (base file is CUDA-only)
# ---------------------------------------------------------------------------


def run_attention_megakernel(
    compiled: CompiledAttentionMegakernel,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    if q.dtype != torch.float32:
        raise TypeError("float32 only")
    expected = (compiled.n_heads, compiled.seq_len, compiled.head_dim)
    for name, t in (("Q", q), ("K", k), ("V", v)):
        if tuple(t.shape) != expected:
            raise ValueError(f"{name} shape {tuple(t.shape)} != {expected}")
    device = q.device
    p = torch.zeros((compiled.n_heads, compiled.seq_len, compiled.seq_len),
                    dtype=torch.float32, device=device)
    o = torch.zeros((compiled.n_heads, compiled.seq_len, compiled.head_dim),
                    dtype=torch.float32, device=device)
    n_events = compiled.n_heads * compiled.q_tiles
    e = torch.full((n_events,), 1, dtype=torch.int32, device=device)
    queue, lens = _flatten_queue(compiled, device)
    inv_sqrt_d = 1.0 / (compiled.head_dim ** 0.5)
    compiled.kernel_callable[(compiled.sm_count,)](
        q, k, v, p, o, e, queue, lens,
        compiled.seq_len, compiled.head_dim, compiled.q_tile_size,
        compiled.q_tiles, inv_sqrt_d,
        compiled.sm_count, compiled.max_qlen,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    _accel_sync()
    return o


# ---------------------------------------------------------------------------
# Locality-aware bodies (use _event_notify / _event_wait from emitter)
# ---------------------------------------------------------------------------

_COMPUTE_SCORES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

q_rows   = q_tile * Q_TILE + tl.arange(0, Q_TILE)
key_cols = tl.arange(0, S)
d_cols   = tl.arange(0, D)

q_ptrs = Q_ptr + h * (S * D) + q_rows[:, None] * D + d_cols[None, :]
q      = tl.load(q_ptrs)
k_ptrs = K_ptr + h * (S * D) + key_cols[:, None] * D + d_cols[None, :]
k      = tl.load(k_ptrs)

scores = tl.dot(q, tl.trans(k)) * INV_SQRT_D
row_max  = tl.max(scores, axis=1)
scores   = scores - row_max[:, None]
exp_scs  = tl.exp(scores)
denom    = tl.sum(exp_scs, axis=1)
probs    = exp_scs / denom[:, None]

p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_cols[None, :]
tl.store(p_ptrs, probs)

sm_id = tl.program_id(0)
_event_notify(E_ptr, h * Q_TILES + q_tile, sm_id)
"""


_APPLY_VALUES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

sm_id = tl.program_id(0)
_event_wait(E_ptr, h * Q_TILES + q_tile, sm_id)

q_rows   = q_tile * Q_TILE + tl.arange(0, Q_TILE)
key_rows = tl.arange(0, S)
d_cols   = tl.arange(0, D)

p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_rows[None, :]
p      = tl.load(p_ptrs)
v_ptrs = V_ptr + h * (S * D) + key_rows[:, None] * D + d_cols[None, :]
v      = tl.load(v_ptrs)

out    = tl.dot(p, v)
o_ptrs = O_ptr + h * (S * D) + q_rows[:, None] * D + d_cols[None, :]
tl.store(o_ptrs, out)
"""


# ---------------------------------------------------------------------------
# Compile (with launch-param support)
# ---------------------------------------------------------------------------


def compile_attention_megakernel(
    n_heads: int = 4, seq_len: int = 64, head_dim: int = 32,
    q_tile_size: int = 16, num_warps: int = 4, num_stages: int = 3,
) -> CompiledAttentionMegakernel:
    if seq_len % q_tile_size != 0:
        raise ValueError(f"seq_len must be divisible by q_tile_size")
    q_tiles = seq_len // q_tile_size

    mod, graph = build_attention_event_graph(n_heads, q_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("Q_ptr", "K_ptr", "V_ptr", "P_ptr", "O_ptr"),
        constexpr_args=("S", "D", "Q_TILE", "Q_TILES", "INV_SQRT_D"),
        device_functions=(
            DeviceFunctionSpec(name="compute_scores", body_source=_COMPUTE_SCORES_BODY),
            DeviceFunctionSpec(name="apply_values", body_source=_APPLY_VALUES_BODY),
        ),
        num_warps=num_warps, num_stages=num_stages,
    )
    lowering = lower_megakernel(graph, spec=spec)
    if not lowering.kernel_source:
        raise RuntimeError(f"emitter rejected: {lowering.diagnostics}")

    fd, path = tempfile.mkstemp(prefix=f"{lowering.kernel_name}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(lowering.kernel_source)
    linecache.checkcache(path)
    m = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if m is None or m.loader is None:
        raise RuntimeError(f"importlib spec failed")
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    kc = getattr(mod, lowering.kernel_name)

    return CompiledAttentionMegakernel(
        kernel_name=lowering.kernel_name, kernel_source=lowering.kernel_source,
        kernel_callable=kc, lowering=lowering,
        n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
        q_tiles=q_tiles, q_tile_size=q_tile_size,
        sm_count=int(lowering.launch_config["grid"]),
        max_qlen=max((len(q) for q in lowering.task_queue.values()), default=1),
    )


def compile_attention_megakernel_autotune(
    n_heads: int = 4, seq_len: int = 64, head_dim: int = 32,
    q_tile_size: int = 16,
) -> CompiledAttentionMegakernel:
    if seq_len % q_tile_size != 0:
        raise ValueError(f"seq_len must be divisible by q_tile_size")
    q_tiles = seq_len // q_tile_size

    mod, graph = build_attention_event_graph(n_heads, q_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("Q_ptr", "K_ptr", "V_ptr", "P_ptr", "O_ptr"),
        constexpr_args=("S", "D", "Q_TILE", "Q_TILES", "INV_SQRT_D"),
        device_functions=(
            DeviceFunctionSpec(name="compute_scores", body_source=_COMPUTE_SCORES_BODY),
            DeviceFunctionSpec(name="apply_values", body_source=_APPLY_VALUES_BODY),
        ),
        tune_config={"num_warps": (1, 4), "num_stages": (1, 3, 5)},
    )
    lowering = lower_megakernel(graph, spec=spec)
    if not lowering.kernel_source:
        raise RuntimeError(f"emitter rejected: {lowering.diagnostics}")

    fd, path = tempfile.mkstemp(prefix=f"{lowering.kernel_name}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(lowering.kernel_source)
    linecache.checkcache(path)
    m = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if m is None or m.loader is None:
        raise RuntimeError(f"importlib spec failed")
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    kc = getattr(mod, lowering.kernel_name)

    return CompiledAttentionMegakernel(
        kernel_name=lowering.kernel_name, kernel_source=lowering.kernel_source,
        kernel_callable=kc, lowering=lowering,
        n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
        q_tiles=q_tiles, q_tile_size=q_tile_size,
        sm_count=int(lowering.launch_config["grid"]),
        max_qlen=max((len(q) for q in lowering.task_queue.values()), default=1),
    )


def run_attention_megakernel_autotune(
    compiled: CompiledAttentionMegakernel,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    if q.dtype != torch.float32:
        raise TypeError("float32 only")
    device = q.device
    p = torch.zeros((compiled.n_heads, compiled.seq_len, compiled.seq_len),
                    dtype=torch.float32, device=device)
    o = torch.zeros((compiled.n_heads, compiled.seq_len, compiled.head_dim),
                    dtype=torch.float32, device=device)
    n_events = compiled.n_heads * compiled.q_tiles
    e = torch.full((n_events,), 1, dtype=torch.int32, device=device)
    queue, lens = _flatten_queue(compiled, device)
    inv_sqrt_d = 1.0 / (compiled.head_dim ** 0.5)
    compiled.kernel_callable[(compiled.sm_count,)](
        q, k, v, p, o, e, queue, lens,
        compiled.seq_len, compiled.head_dim, compiled.q_tile_size,
        compiled.q_tiles, inv_sqrt_d,
        compiled.sm_count, compiled.max_qlen,
    )
    _accel_sync()
    return o


def search_attention_megakernel(
    n_heads: int = 4, seq_len: int = 64, head_dim: int = 32,
    *,
    q_tile_values: tuple[int, ...] = (8, 16, 32, 64),
    num_warps_values: tuple[int, ...] = (1, 4),
    num_stages_values: tuple[int, ...] = (1, 3, 5),
    verbose: bool = True,
) -> dict:
    from triton.testing import do_bench

    _accel_sync()
    if hasattr(torch, "mlu") and hasattr(torch.mlu, "empty_cache"):
        torch.mlu.empty_cache()

    q = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device=_ACCEL_DEVICE)
    k = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device=_ACCEL_DEVICE)
    v = torch.randn((n_heads, seq_len, head_dim), dtype=torch.float32, device=_ACCEL_DEVICE)

    best_time = float("inf")
    best_config: dict = {}
    total = sum(1 for qt in q_tile_values if seq_len % qt == 0) * len(num_warps_values) * len(num_stages_values)
    tried = 0

    if verbose:
        print(f"  Searching {total} configs over {_ACCEL_TAG}...")
        print(f"    q_tile_size ∈ {q_tile_values}")
        print(f"    num_warps × num_stages: {num_warps_values} × {num_stages_values}")
        print()

    for qt in q_tile_values:
        if seq_len % qt != 0:
            continue
        for nw in num_warps_values:
            for ns in num_stages_values:
                tried += 1
                label = f"[{tried}/{total}]"
                compiled = compile_attention_megakernel(
                    n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
                    q_tile_size=qt, num_warps=nw, num_stages=ns,
                )
                try:
                    _ = run_attention_megakernel(compiled, q, k, v)
                    _accel_sync()
                    t = do_bench(lambda: run_attention_megakernel(compiled, q, k, v))
                except RuntimeError as exc:
                    try: _accel_sync()
                    except RuntimeError: pass
                    if verbose:
                        print(f"    {label} q_tile={qt:>3} warps={nw} stages={ns}  → SKIP ({exc})")
                    continue

                if verbose:
                    print(f"    {label} q_tile={qt:>3} warps={nw} stages={ns}  → {t:.4f} ms{' *' if t < best_time else ''}")

                if t < best_time:
                    best_time = t
                    best_config = {"q_tile_size": qt, "num_warps": nw, "num_stages": ns,
                                   "time_ms": t, "compiled": compiled}

    if verbose:
        print()
        if best_config:
            print(f"  Best: q_tile={best_config['q_tile_size']}, "
                  f"num_warps={best_config['num_warps']}, num_stages={best_config['num_stages']} "
                  f"→ {best_config['time_ms']:.4f} ms")
    return best_config


# ---------------------------------------------------------------------------
# Standalone benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_ACCEL:
        raise SystemExit(f"Requires an accelerator ({_ACCEL_TAG} not available).")
    from triton.testing import do_bench

    H, S, D = 4, 64, 32
    Q_TILE = 16

    print("=" * 60)
    print(f"  Attention megakernel (H={H}, S={S}, D={D}) — locality-aware")
    print("=" * 60)
    print()

    compiled = compile_attention_megakernel(n_heads=H, seq_len=S, head_dim=D, q_tile_size=Q_TILE)
    print(f"  Emitted: {compiled.kernel_name} ({len(compiled.kernel_source):,} chars)")
    print(f"  Q_TILE={compiled.q_tile_size}, grid={compiled.sm_count}")
    print()

    q = torch.randn((H, S, D), dtype=torch.float32, device=_ACCEL_DEVICE)
    k = torch.randn((H, S, D), dtype=torch.float32, device=_ACCEL_DEVICE)
    v = torch.randn((H, S, D), dtype=torch.float32, device=_ACCEL_DEVICE)

    got = run_attention_megakernel(compiled, q, k, v)
    ref = reference_attention(q, k, v)
    err = (got - ref).abs().max().item()
    print(f"  Correctness: max |got - ref| = {err:.3e}  {'OK' if err < 1e-3 else 'FAIL'}")
    print()

    # Autotune
    print("-" * 60)
    print("  Autotune (num_warps × num_stages)")
    print("-" * 60)
    compiled_auto = compile_attention_megakernel_autotune(n_heads=H, seq_len=S, head_dim=D, q_tile_size=Q_TILE)
    _ = run_attention_megakernel_autotune(compiled_auto, q, k, v)
    _accel_sync()
    auto_ms = do_bench(lambda: run_attention_megakernel_autotune(compiled_auto, q, k, v))
    print(f"  autotuned:  {auto_ms:.3f} ms")
    print()

    # Search
    print("-" * 60)
    print("  Outer search (q_tile_size × num_warps × num_stages)")
    print("-" * 60)
    result = search_attention_megakernel(n_heads=H, seq_len=S, head_dim=D, verbose=True)
    search_ms = result.get("time_ms", float("nan"))
    print()

    # Baselines
    print("-" * 60)
    print("  Baselines")
    print("-" * 60)
    _ = run_attention_megakernel(compiled, q, k, v)
    _accel_sync()
    default_ms = do_bench(lambda: run_attention_megakernel(compiled, q, k, v))
    print(f"  megakernel (default):      {default_ms:.3f} ms")

    eager_fn = lambda q, k, v: F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False,
    ).squeeze(0)
    eager_ms = do_bench(lambda: eager_fn(q, k, v))
    print(f"  eager (F.sdpa):            {eager_ms:.3f} ms")

    comp_fn = torch.compile(eager_fn, dynamic=False)
    _ = comp_fn(q, k, v); _accel_sync()
    comp_ms = do_bench(lambda: comp_fn(q, k, v))
    print(f"  torch.compile:             {comp_ms:.3f} ms")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'':>30s} {'time (ms)':>10s}  {'vs eager':>10s}")
    print(f"  {'eager':>30s} {eager_ms:>10.3f}  {'1.00x':>10s}")
    print(f"  {'torch.compile':>30s} {comp_ms:>10.3f}  {comp_ms/eager_ms:>9.2f}x")
    print(f"  {'megakernel (default)':>30s} {default_ms:>10.3f}  {default_ms/eager_ms:>9.2f}x")
    print(f"  {'megakernel (autotuned)':>30s} {auto_ms:>10.3f}  {auto_ms/eager_ms:>9.2f}x")
    if result:
        print(f"  {'megakernel (search)':>30s} {search_ms:>10.3f}  {search_ms/eager_ms:>9.2f}x")
    if default_ms > 0 and result:
        print(f"\n  Autotune vs default:    {default_ms/auto_ms:.2f}x")
        print(f"  Search vs default:      {default_ms/search_ms:.2f}x")
