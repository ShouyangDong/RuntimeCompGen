"""Autotuned row-sum megakernel (demonstrates MegakernelLoweringSpec.tune_config).

Companion to ``row_sum_megakernel.py``.  Whereas the base example uses
hard-coded ``BLOCK_M=32, BLOCK_K=32, num_warps=4, num_stages=2``, this
example lets Triton search for the best combination automatically via
:attr:`MegakernelLoweringSpec.tune_config`.

The autotune sweep covers two Triton launch-param axes
(tuned to MLU's valid range; CUDA supports wider):

    ============== ==============================
    Axis            Values
    ============== ==============================
    num_warps       1, 4
    num_stages      1, 3, 5
    ============== ==============================

Tile sizes (BLOCK_M, BLOCK_K) are NOT tuned because they determine the
IR task graph (number of tasks, event-tensor shape) which is fixed
at lowering time.  The emitter produces a ``@triton.autotune``-decorated
kernel with 3 × 4 = 12 configs.

Run as::

    python examples/event_tensor/row_sum_autotune.py
"""

from __future__ import annotations

import importlib.util
import linecache
import os
from dataclasses import dataclass

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
    MegakernelLoweringResult,
    MegakernelLoweringSpec,
    lower_megakernel,
)

# Reuse the bodies, IR builder, and CompiledMegakernel from the base
# row_sum example.  We must write our own launcher because the autotuned
# kernel does NOT accept explicit num_warps / num_stages kwargs (Triton's
# Autotuner selects and injects them).
from examples.event_tensor.row_sum_megakernel import (
    _FINAL_SUM_BODY,
    _PARTIAL_SUM_BODY,
    CompiledMegakernel,
    _flatten_queue,
    build_event_graph,
    reference,
)


# ---------------------------------------------------------------------------
# Autotuned launcher (no num_warps / num_stages)
# ---------------------------------------------------------------------------


def run_megakernel_autotune(
    compiled: CompiledMegakernel,
    a: torch.Tensor,
) -> torch.Tensor:
    """Launch the autotuned megakernel — same logic as ``run_megakernel``
    but WITHOUT ``num_warps`` and ``num_stages`` kwargs.  Triton's
    ``Autotuner`` selects the best config at call time and passes
    those params to the underlying ``@triton.jit`` function.
    """
    if a.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {a.dtype}")
    expected_rows = compiled.n_row_blocks * compiled.block_m
    expected_cols = compiled.j_chunks * compiled.block_k
    if a.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"input shape {tuple(a.shape)} != expected "
            f"({expected_rows}, {expected_cols})"
        )
    if not (a.is_cuda or (_HAS_MLU and a.is_mlu)):
        raise RuntimeError(f"megakernel requires an accelerator tensor ({_ACCEL_TAG})")

    device = a.device
    n_events = compiled.n_row_blocks * compiled.j_chunks

    b = torch.zeros(
        (compiled.j_chunks, compiled.n_row_blocks * compiled.block_m),
        dtype=torch.float32, device=device,
    )
    c = torch.zeros((expected_rows,), dtype=torch.float32, device=device)
    e = torch.full((n_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    compiled.kernel_callable[(compiled.sm_count,)](
        a, b, c,
        e,
        queue, lens,
        compiled.n_row_blocks, compiled.j_chunks,
        a.shape[1], compiled.block_m, compiled.block_k,
        compiled.sm_count, compiled.max_qlen,
    )
    _accel_sync()

    # NOTE: no event-counter drain check — @triton.autotune runs the
    # kernel dozens of times per config during benchmarking, so counters
    # will be deeply negative.  Correctness is verified via output vs
    # reference, not via counter state.

    return c


# ---------------------------------------------------------------------------


def compile_megakernel_autotune(
    n_row_blocks: int = 8,
    j_chunks: int = 4,
    block_m: int = 32,
    block_k: int = 32,
) -> CompiledMegakernel:
    """Same pipeline as ``compile_megakernel``, but with ``tune_config``.

    The emitted kernel is decorated with ``@triton.autotune`` so Triton
    benchmarks every config in the cartesian product on first launch and
    picks the fastest.

    .. note::

        ``block_m`` / ``block_k`` are NOT tuned — they determine the
        IR task graph (number of tasks, event-tensor shape) which is
        fixed at lowering time.  Only ``num_warps`` and ``num_stages``
        are swept.
    """
    mod, graph = build_event_graph(n_row_blocks, j_chunks)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("A_ptr", "B_ptr", "C_ptr"),
        constexpr_args=("N_ROW_BLOCKS", "J", "K", "BLOCK_M", "BLOCK_K"),
        device_functions=(
            DeviceFunctionSpec(name="partial_sum", body_source=_PARTIAL_SUM_BODY),
            DeviceFunctionSpec(name="final_sum", body_source=_FINAL_SUM_BODY),
        ),
        # ── the only difference from the base example ──
        # NOTE: BLOCK_M / BLOCK_K are NOT tuned — the IR task graph
        # (N_ROW_BLOCKS = M // BLOCK_M, event-tensor shape) depends on
        # them and is fixed at lowering time.
        tune_config={
            "num_warps":   (1, 4),
            "num_stages":  (1, 3, 5),
        },
    )
    lowering = lower_megakernel(graph, spec=spec)
    if not lowering.kernel_source:
        raise RuntimeError(f"emitter rejected the graph: {lowering.diagnostics}")

    path = os.path.join(os.path.dirname(__file__) or ".", f"{lowering.kernel_name}.py")
    with open(path, "w") as f:
        f.write(lowering.kernel_source)
    linecache.checkcache(path)
    module_spec = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_callable = getattr(module, lowering.kernel_name)

    sm_count = int(lowering.launch_config["grid"])
    max_qlen = max((len(q) for q in lowering.task_queue.values()), default=1)

    return CompiledMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_row_blocks=n_row_blocks,
        j_chunks=j_chunks,
        block_m=block_m,   # fixed (not in tune_config)
        block_k=block_k,   # fixed (not in tune_config)
        sm_count=sm_count,
        max_qlen=max_qlen,
        device_function_table=lowering.device_function_table,
    )


# ---------------------------------------------------------------------------
# Outer search: sweep BLOCK_M, BLOCK_K (re-build IR per tile size)
# ---------------------------------------------------------------------------


def search_row_sum_megakernel(
    M: int = 256,
    K: int = 128,
    *,
    block_m_values: tuple[int, ...] = (16, 32, 64, 128),
    block_k_values: tuple[int, ...] = (16, 32, 64, 128),
    num_warps_values: tuple[int, ...] = (1, 4),
    num_stages_values: tuple[int, ...] = (1, 3, 5),
    max_total_tasks: int = 64,
    verbose: bool = True,
) -> dict:
    """Outer-loop search over tile sizes + launch params.

    Unlike :func:`compile_megakernel_autotune` which only sweeps launch
    params via ``@triton.autotune`` (IR graph is fixed), this function
    re-builds the IR graph, event tensors, and task schedule for every
    (BLOCK_M, BLOCK_K) combination so tile sizes can be tuned.

    Args:
        max_total_tasks: Skip configs whose total task count
            (partial_sum + final_sum tasks = n_row_blocks * j_chunks + n_row_blocks)
            exceeds this threshold.  Too many tiny tasks → event-coordination
            overhead dominates → kernel timeout on some accelerators.

    Returns a dict with the best config and its timing.
    """
    from examples.event_tensor.row_sum_megakernel import (
        compile_megakernel,
        run_megakernel,
    )
    from triton.testing import do_bench

    _accel_sync()
    if hasattr(torch, "mlu") and hasattr(torch.mlu, "empty_cache"):
        torch.mlu.empty_cache()

    a = torch.randn((M, K), dtype=torch.float32, device=_ACCEL_DEVICE)

    best_time = float("inf")
    best_config: dict = {}
    total = 0
    tried = 0
    skipped = 0

    for block_m in block_m_values:
        if M % block_m != 0:
            continue
        n_row_blocks = M // block_m
        for block_k in block_k_values:
            if K % block_k != 0:
                continue
            j_chunks = K // block_k
            n_tasks = n_row_blocks * j_chunks + n_row_blocks
            if n_tasks > max_total_tasks:
                skipped += len(num_warps_values) * len(num_stages_values)
                continue
            total += len(num_warps_values) * len(num_stages_values)

    if verbose:
        print(f"  Searching {total} configs over {_ACCEL_TAG}...")
        if skipped:
            print(f"    ({skipped} skipped: > {max_total_tasks} total tasks)")
        print(f"    M={M}, K={K}")
        print(f"    BLOCK_M x BLOCK_K: {block_m_values} x {block_k_values}")
        print(f"    num_warps x num_stages: {num_warps_values} x {num_stages_values}")
        print()

    for block_m in block_m_values:
        if M % block_m != 0:
            continue
        n_row_blocks = M // block_m
        for block_k in block_k_values:
            if K % block_k != 0:
                continue
            j_chunks = K // block_k
            n_tasks = n_row_blocks * j_chunks + n_row_blocks
            if n_tasks > max_total_tasks:
                continue
            for nw in num_warps_values:
                for ns in num_stages_values:
                    tried += 1
                    label = f"[{tried}/{total}]"
                    compiled = compile_megakernel(
                        n_row_blocks=n_row_blocks,
                        j_chunks=j_chunks,
                        block_m=block_m,
                        block_k=block_k,
                        num_warps=nw,
                        num_stages=ns,
                    )
                    try:
                        _ = run_megakernel(compiled, a)
                        _accel_sync()
                        t = do_bench(lambda: run_megakernel(compiled, a))
                    except RuntimeError as exc:
                        # Flush any pending async MLU error so it doesn't
                        # leak to the next kernel launch.
                        try:
                            _accel_sync()
                        except RuntimeError:
                            pass
                        if verbose:
                            print(f"    {label} BLOCK_M={block_m:>4} BLOCK_K={block_k:>4}  "
                                  f"warps={nw} stages={ns}  → SKIP ({exc})")
                        continue

                    if verbose:
                        marker = " *" if t < best_time else ""
                        print(f"    {label} BLOCK_M={block_m:>4} BLOCK_K={block_k:>4}  "
                              f"warps={nw} stages={ns}  → {t:.4f} ms{marker}")

                    if t < best_time:
                        best_time = t
                        best_config = {
                            "BLOCK_M": block_m,
                            "BLOCK_K": block_k,
                            "num_warps": nw,
                            "num_stages": ns,
                            "time_ms": t,
                            "compiled": compiled,
                        }

    if verbose:
        print()
        if not best_config:
            raise RuntimeError(
                f"No config passed.  Try increasing max_total_tasks (currently {max_total_tasks}) "
                f"or using different block sizes."
            )
        print(f"  Best: BLOCK_M={best_config['BLOCK_M']}, BLOCK_K={best_config['BLOCK_K']}, "
              f"num_warps={best_config['num_warps']}, num_stages={best_config['num_stages']} "
              f"→ {best_config['time_ms']:.4f} ms")

    return best_config


# ---------------------------------------------------------------------------
# Standalone benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_ACCEL:
        raise SystemExit(f"This example requires an accelerator ({_ACCEL_TAG} not available).")
    from triton.testing import do_bench

    N_ROW_BLOCKS = 8
    J_CHUNKS = 4
    BLOCK_M = 32
    BLOCK_K = 32
    M = N_ROW_BLOCKS * BLOCK_M   # 256
    K = J_CHUNKS * BLOCK_K       # 128

    # ── Autotuned ──
    print("=" * 60)
    print("  Autotuned row-sum megakernel")
    print("=" * 60)
    print(f"  tune_config: num_warps={1,4}, num_stages={1,3,5}")
    print(f"  (2×3 = 6 configs; Triton benchmarks on first launch)")
    print()

    compiled_auto = compile_megakernel_autotune(
        n_row_blocks=N_ROW_BLOCKS, j_chunks=J_CHUNKS,
        block_m=BLOCK_M, block_k=BLOCK_K,
    )
    print(f"  Emitter produced: {compiled_auto.kernel_name}")
    print(f"  Source size:      {len(compiled_auto.kernel_source):,} chars")
    print(f"  Decorator:        @triton.autotune")
    print(f"  grid (SM_COUNT):  {compiled_auto.sm_count}")
    print()

    a = torch.randn((M, K), dtype=torch.float32, device=_ACCEL_DEVICE)

    # Correctness check
    got_auto = run_megakernel_autotune(compiled_auto, a)
    ref = reference(a)
    err = (got_auto - ref).abs().max().item()
    print(f"  Correctness: max |got - ref| = {err:.3e}  {'OK' if err < 1e-3 else 'FAIL'}")
    print()

    # First launch triggers autotuning (Triton benchmarks all 12 configs).
    print("  Running first launch (triggers autotune search)...")
    _ = run_megakernel_autotune(compiled_auto, a)
    _accel_sync()
    print("  Autotune complete.  Benchmarking best config:")
    print()

    auto_ms = do_bench(lambda: run_megakernel_autotune(compiled_auto, a))
    print(f"  autotuned megakernel:  {auto_ms:.3f} ms")
    print()

    # ── Outer search (sweep BLOCK_M, BLOCK_K + launch params) ──
    print("-" * 60)
    print("  Outer search (re-builds IR per tile config)")
    print("-" * 60)

    result = search_row_sum_megakernel(M=M, K=K, verbose=True)
    compiled_search = result["compiled"]
    search_ms = result["time_ms"]
    print()

    # ── Default (hardcoded) for comparison ──
    print("-" * 60)
    print("  Default (hardcoded) megakernel for comparison")
    print("-" * 60)

    from examples.event_tensor.row_sum_megakernel import compile_megakernel, run_megakernel

    # Reset device state: the search phase ran many kernel launches and may
    # have left the MLU driver in a fragile state.
    _accel_sync()
    if hasattr(torch, "mlu") and hasattr(torch.mlu, "empty_cache"):
        torch.mlu.empty_cache()
    a = torch.randn((M, K), dtype=torch.float32, device=_ACCEL_DEVICE)
    _accel_sync()

    compiled_default = compile_megakernel(
        n_row_blocks=N_ROW_BLOCKS, j_chunks=J_CHUNKS,
        block_m=BLOCK_M, block_k=BLOCK_K,
    )
    print(f"  Decorator: @triton.jit")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}")
    print(f"  num_warps=4, num_stages=3")
    print()

    try:
        _ = run_megakernel(compiled_default, a)
        _accel_sync()
        default_ms = do_bench(lambda: run_megakernel(compiled_default, a))
        print(f"  default megakernel:    {default_ms:.3f} ms")
    except RuntimeError as exc:
        print(f"  default megakernel:    FAILED ({exc})")
        default_ms = float("nan")
    print()

    # ── Baselines ──
    print("-" * 60)
    print("  Baselines")
    print("-" * 60)

    eager_ms = do_bench(lambda: a.sum(dim=-1))
    print(f"  eager (a.sum(-1)):     {eager_ms:.3f} ms")

    compiled_sum = torch.compile(lambda x: x.sum(dim=-1), dynamic=False)
    _ = compiled_sum(a)
    comp_ms = do_bench(lambda: compiled_sum(a))
    print(f"  torch.compile:         {comp_ms:.3f} ms")
    print()

    # ── Summary ──
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'':>30s} {'time (ms)':>10s}  {'vs eager':>10s}")
    print(f"  {'eager':>30s} {eager_ms:>10.3f}  {'1.00x':>10s}")
    print(f"  {'torch.compile':>30s} {comp_ms:>10.3f}  {comp_ms/eager_ms:>9.2f}x")
    print(f"  {'megakernel (default)':>30s} {default_ms:>10.3f}  {default_ms/eager_ms:>9.2f}x")
    print(f"  {'megakernel (autotuned)':>30s} {auto_ms:>10.3f}  {auto_ms/eager_ms:>9.2f}x")
    print(f"  {'megakernel (search)':>30s} {search_ms:>10.3f}  {search_ms/eager_ms:>9.2f}x")
    if default_ms > 0:
        speedup = default_ms / auto_ms
        print(f"\n  Autotune vs default speedup:    {speedup:.2f}x")
        speedup2 = default_ms / search_ms
        print(f"  Outer-search vs default speedup: {speedup2:.2f}x")
