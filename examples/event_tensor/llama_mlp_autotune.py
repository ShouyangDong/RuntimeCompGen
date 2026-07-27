"""Autotuned Llama-style SwiGLU MLP megakernel.

Companion to ``llama_mlp_megakernel.py``.  Adds two autotuning modes:

    1. ``compile_mlp_megakernel_autotune`` — uses ``@triton.autotune``
       to sweep ``num_warps`` × ``num_stages`` (6 configs).

    2. ``search_mlp_megakernel`` — outer-loop search over tile sizes
       (BLOCK_M, BLOCK_I, BLOCK_N) + launch params.

Run as::

    python examples/event_tensor/llama_mlp_autotune.py
"""

from __future__ import annotations

import importlib.util
import linecache
import os
import tempfile

import torch

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


from compgen.ir.payload.passes.megakernel_static_schedule import (
    StaticMegakernelSchedule,
)
from compgen.ir.tile.lower_megakernel import (
    DeviceFunctionSpec,
    MegakernelLoweringSpec,
    lower_megakernel,
)
from examples.event_tensor.llama_mlp_megakernel import (
    CompiledMLPMegakernel,
    _flatten_queue,
    build_mlp_event_graph,
    reference_mlp,
)


# ---------------------------------------------------------------------------
# Locality-aware bodies + MLU/CUDA launcher
# ---------------------------------------------------------------------------

_GATE_PROJ_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, K)

x_ptrs = X_ptr + m_rows[:, None] * K + k_idx[None, :]
x      = tl.load(x_ptrs)
wg_ptrs = WG_ptr + i_cols[:, None] * K + k_idx[None, :]
wg      = tl.load(wg_ptrs)
proj    = tl.dot(x, tl.trans(wg))
gated   = proj * tl.sigmoid(proj)

g_ptrs  = G_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(g_ptrs, gated)

sm_id = tl.program_id(0)
_event_notify(EG_ptr, m_tile * I_TILES + i_tile, sm_id)
"""

_UP_PROJ_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, K)

x_ptrs = X_ptr + m_rows[:, None] * K + k_idx[None, :]
x      = tl.load(x_ptrs)
wu_ptrs = WU_ptr + i_cols[:, None] * K + k_idx[None, :]
wu      = tl.load(wu_ptrs)
proj    = tl.dot(x, tl.trans(wu))

u_ptrs  = U_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(u_ptrs, proj)

sm_id = tl.program_id(0)
_event_notify(EU_ptr, m_tile * I_TILES + i_tile, sm_id)
"""

_DOWN_PROJ_BODY = r"""
m_tile = task_id // N_TILES
n_tile = task_id %  N_TILES

sm_id = tl.program_id(0)
for it in tl.static_range(0, I_TILES):
    _event_wait(EG_ptr, m_tile * I_TILES + it, sm_id)
    _event_wait(EU_ptr, m_tile * I_TILES + it, sm_id)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
n_cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
i_idx  = tl.arange(0, I)

g_ptrs = G_ptr + m_rows[:, None] * I + i_idx[None, :]
u_ptrs = U_ptr + m_rows[:, None] * I + i_idx[None, :]
g      = tl.load(g_ptrs)
u      = tl.load(u_ptrs)
hid    = g * u

wd_ptrs = WD_ptr + n_cols[:, None] * I + i_idx[None, :]
wd      = tl.load(wd_ptrs)
y       = tl.dot(hid, tl.trans(wd))

y_ptrs  = Y_ptr + m_rows[:, None] * N + n_cols[None, :]
tl.store(y_ptrs, y)
"""


def run_mlp_megakernel(
    compiled: CompiledMLPMegakernel,
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> torch.Tensor:
    device = x.device
    g_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    u_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    y    = torch.zeros((compiled.M, compiled.N), dtype=torch.float32, device=device)

    m_tiles = compiled.M // compiled.BLOCK_M
    i_tiles = compiled.I // compiled.BLOCK_I
    n_tiles = compiled.N // compiled.BLOCK_N
    n_intermediate_events = m_tiles * i_tiles
    eg = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)
    eu = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    compiled.kernel_callable[(compiled.sm_count,)](
        x, w_gate, w_up, w_down, g_ws, u_ws, y,
        eg, eu, queue, lens,
        compiled.M, compiled.K, compiled.I, compiled.N,
        compiled.BLOCK_M, compiled.BLOCK_I, compiled.BLOCK_N,
        i_tiles, n_tiles,
        compiled.sm_count, compiled.max_qlen,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    _accel_sync()
    return y


# ---------------------------------------------------------------------------
# Autotuned compile (launch params only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compile (with launch-param support, locality-aware bodies)
# ---------------------------------------------------------------------------


def compile_mlp_megakernel(
    M: int = 32, K: int = 64, I: int = 128, N: int = 64,
    BLOCK_M: int = 16, BLOCK_I: int = 32, BLOCK_N: int = 16,
    num_warps: int = 4, num_stages: int = 3,
) -> CompiledMLPMegakernel:
    m_tiles = M // BLOCK_M
    i_tiles = I // BLOCK_I
    n_tiles = N // BLOCK_N

    mod, graph = build_mlp_event_graph(m_tiles, i_tiles, n_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("X_ptr", "WG_ptr", "WU_ptr", "WD_ptr", "G_ptr", "U_ptr", "Y_ptr"),
        constexpr_args=("M", "K", "I", "N", "BLOCK_M", "BLOCK_I", "BLOCK_N", "I_TILES", "N_TILES"),
        device_functions=(
            DeviceFunctionSpec(name="gate_proj_tile", body_source=_GATE_PROJ_BODY),
            DeviceFunctionSpec(name="up_proj_tile",   body_source=_UP_PROJ_BODY),
            DeviceFunctionSpec(name="down_proj_tile", body_source=_DOWN_PROJ_BODY),
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

    return CompiledMLPMegakernel(
        kernel_name=lowering.kernel_name, kernel_source=lowering.kernel_source,
        kernel_callable=kc, lowering=lowering,
        M=M, K=K, I=I, N=N,
        BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I, BLOCK_N=BLOCK_N,
        sm_count=int(lowering.launch_config["grid"]),
        max_qlen=max((len(q) for q in lowering.task_queue.values()), default=1),
    )


def compile_mlp_megakernel_autotune(
    M: int = 32, K: int = 64, I: int = 128, N: int = 64,
    BLOCK_M: int = 16, BLOCK_I: int = 32, BLOCK_N: int = 16,
) -> CompiledMLPMegakernel:
    m_tiles = M // BLOCK_M
    i_tiles = I // BLOCK_I
    n_tiles = N // BLOCK_N

    mod, graph = build_mlp_event_graph(m_tiles, i_tiles, n_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("X_ptr", "WG_ptr", "WU_ptr", "WD_ptr", "G_ptr", "U_ptr", "Y_ptr"),
        constexpr_args=("M", "K", "I", "N", "BLOCK_M", "BLOCK_I", "BLOCK_N", "I_TILES", "N_TILES"),
        device_functions=(
            DeviceFunctionSpec(name="gate_proj_tile", body_source=_GATE_PROJ_BODY),
            DeviceFunctionSpec(name="up_proj_tile",   body_source=_UP_PROJ_BODY),
            DeviceFunctionSpec(name="down_proj_tile", body_source=_DOWN_PROJ_BODY),
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
    module_spec = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"importlib spec failed for {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_callable = getattr(module, lowering.kernel_name)

    return CompiledMLPMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        M=M, K=K, I=I, N=N,
        BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I, BLOCK_N=BLOCK_N,
        sm_count=int(lowering.launch_config["grid"]),
        max_qlen=max((len(q) for q in lowering.task_queue.values()), default=1),
    )


def run_mlp_megakernel_autotune(
    compiled: CompiledMLPMegakernel,
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> torch.Tensor:
    """Same as run_mlp_megakernel but without num_warps/num_stages kwargs."""
    device = x.device
    g_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    u_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    y    = torch.zeros((compiled.M, compiled.N), dtype=torch.float32, device=device)

    m_tiles = compiled.M // compiled.BLOCK_M
    i_tiles = compiled.I // compiled.BLOCK_I
    n_tiles = compiled.N // compiled.BLOCK_N
    n_intermediate_events = m_tiles * i_tiles
    eg = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)
    eu = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    compiled.kernel_callable[(compiled.sm_count,)](
        x, w_gate, w_up, w_down, g_ws, u_ws, y,
        eg, eu,
        queue, lens,
        compiled.M, compiled.K, compiled.I, compiled.N,
        compiled.BLOCK_M, compiled.BLOCK_I, compiled.BLOCK_N,
        i_tiles, n_tiles,
        compiled.sm_count, compiled.max_qlen,
    )
    _accel_sync()
    return y


# ---------------------------------------------------------------------------
# Outer search over tile sizes + launch params
# ---------------------------------------------------------------------------


def search_mlp_megakernel(
    M: int = 32, K: int = 64, I: int = 128, N: int = 64,
    *,
    block_m_values: tuple[int, ...] = (8, 16, 32),
    block_i_values: tuple[int, ...] = (16, 32, 64, 128),
    block_n_values: tuple[int, ...] = (16, 32, 64),
    num_warps_values: tuple[int, ...] = (1, 4),
    num_stages_values: tuple[int, ...] = (1, 3, 5),
    verbose: bool = True,
) -> dict:
    """Outer search over BLOCK_M, BLOCK_I, BLOCK_N + launch params."""
    from triton.testing import do_bench

    _accel_sync()
    if hasattr(torch, "mlu") and hasattr(torch.mlu, "empty_cache"):
        torch.mlu.empty_cache()

    x = torch.randn((M, K), dtype=torch.float32, device=_ACCEL_DEVICE)
    w_gate = torch.randn((I, K), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05
    w_up   = torch.randn((I, K), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05
    w_down = torch.randn((N, I), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05

    best_time = float("inf")
    best_config: dict = {}
    total = 0
    tried = 0

    for bm in block_m_values:
        if M % bm != 0:
            continue
        for bi in block_i_values:
            if I % bi != 0:
                continue
            for bn in block_n_values:
                if N % bn != 0:
                    continue
                total += len(num_warps_values) * len(num_stages_values)

    if verbose:
        print(f"  Searching {total} configs over {_ACCEL_TAG}...")
        print(f"    BLOCK_M ∈ {block_m_values}, BLOCK_I ∈ {block_i_values}, BLOCK_N ∈ {block_n_values}")
        print(f"    num_warps × num_stages: {num_warps_values} × {num_stages_values}")
        print()

    for bm in block_m_values:
        if M % bm != 0:
            continue
        for bi in block_i_values:
            if I % bi != 0:
                continue
            for bn in block_n_values:
                if N % bn != 0:
                    continue
                for nw in num_warps_values:
                    for ns in num_stages_values:
                        tried += 1
                        label = f"[{tried}/{total}]"
                        compiled = compile_mlp_megakernel(
                            M=M, K=K, I=I, N=N,
                            BLOCK_M=bm, BLOCK_I=bi, BLOCK_N=bn,
                            num_warps=nw, num_stages=ns,
                        )
                        try:
                            _ = run_mlp_megakernel(compiled, x, w_gate, w_up, w_down)
                            _accel_sync()
                            t = do_bench(lambda: run_mlp_megakernel(compiled, x, w_gate, w_up, w_down))
                        except RuntimeError as exc:
                            try:
                                _accel_sync()
                            except RuntimeError:
                                pass
                            if verbose:
                                print(f"    {label} BLOCK_M={bm:>3} BLOCK_I={bi:>3} BLOCK_N={bn:>3}  "
                                      f"warps={nw} stages={ns}  → SKIP ({exc})")
                            continue

                        if verbose:
                            marker = " *" if t < best_time else ""
                            print(f"    {label} BLOCK_M={bm:>3} BLOCK_I={bi:>3} BLOCK_N={bn:>3}  "
                                  f"warps={nw} stages={ns}  → {t:.4f} ms{marker}")

                        if t < best_time:
                            best_time = t
                            best_config = {
                                "BLOCK_M": bm, "BLOCK_I": bi, "BLOCK_N": bn,
                                "num_warps": nw, "num_stages": ns,
                                "time_ms": t, "compiled": compiled,
                            }

    if verbose:
        print()
        if not best_config:
            raise RuntimeError("No config passed.")
        c = best_config
        print(f"  Best: BLOCK_M={c['BLOCK_M']}, BLOCK_I={c['BLOCK_I']}, BLOCK_N={c['BLOCK_N']}, "
              f"num_warps={c['num_warps']}, num_stages={c['num_stages']} → {c['time_ms']:.4f} ms")

    return best_config


# ---------------------------------------------------------------------------
# Standalone benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_ACCEL:
        raise SystemExit(f"This example requires an accelerator ({_ACCEL_TAG} not available).")
    from triton.testing import do_bench

    M, K, I, N = 32, 64, 128, 64
    BM, BI, BN = 16, 32, 16

    # ── Default ──
    print("=" * 60)
    print(f"  Llama MLP megakernel (M={M}, K={K}, I={I}, N={N})")
    print("=" * 60)

    compiled = compile_mlp_megakernel(M=M, K=K, I=I, N=N, BLOCK_M=BM, BLOCK_I=BI, BLOCK_N=BN)
    print(f"  Emitted: {compiled.kernel_name} ({len(compiled.kernel_source):,} chars)")
    print(f"  grid={compiled.sm_count}")
    print()

    x = torch.randn((M, K), dtype=torch.float32, device=_ACCEL_DEVICE)
    w_gate = torch.randn((I, K), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05
    w_up   = torch.randn((I, K), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05
    w_down = torch.randn((N, I), dtype=torch.float32, device=_ACCEL_DEVICE) * 0.05

    got = run_mlp_megakernel(compiled, x, w_gate, w_up, w_down)
    ref = reference_mlp(x, w_gate, w_up, w_down)
    err = (got - ref).abs().max().item()
    print(f"  Correctness: max |got - ref| = {err:.3e}  {'OK' if err < 5e-3 else 'FAIL'}")
    print()

    # ── Autotune ──
    print("-" * 60)
    print("  Autotune (num_warps × num_stages via @triton.autotune)")
    print("-" * 60)
    compiled_auto = compile_mlp_megakernel_autotune(M=M, K=K, I=I, N=N, BLOCK_M=BM, BLOCK_I=BI, BLOCK_N=BN)
    _ = run_mlp_megakernel_autotune(compiled_auto, x, w_gate, w_up, w_down)
    _accel_sync()
    auto_ms = do_bench(lambda: run_mlp_megakernel_autotune(compiled_auto, x, w_gate, w_up, w_down))
    print(f"  autotuned:  {auto_ms:.3f} ms")
    print()

    # ── Outer search ──
    print("-" * 60)
    print("  Outer search (BLOCK_M × BLOCK_I × BLOCK_N × launch params)")
    print("-" * 60)
    result = search_mlp_megakernel(M=M, K=K, I=I, N=N, verbose=True)
    search_ms = result.get("time_ms", float("nan"))
    print()

    # ── Baselines ──
    print("-" * 60)
    print("  Baselines")
    print("-" * 60)
    _ = run_mlp_megakernel(compiled, x, w_gate, w_up, w_down)
    _accel_sync()
    default_ms = do_bench(lambda: run_mlp_megakernel(compiled, x, w_gate, w_up, w_down))
    print(f"  megakernel (default):      {default_ms:.3f} ms")

    eager_ms = do_bench(lambda: reference_mlp(x, w_gate, w_up, w_down))
    print(f"  eager:                     {eager_ms:.3f} ms")

    compiled_fn = torch.compile(reference_mlp, dynamic=False)
    _ = compiled_fn(x, w_gate, w_up, w_down)
    _accel_sync()
    comp_ms = do_bench(lambda: compiled_fn(x, w_gate, w_up, w_down))
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
