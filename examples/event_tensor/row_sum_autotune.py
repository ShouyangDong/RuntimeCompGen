"""Autotuned row-sum megakernel (demonstrates MegakernelLoweringSpec.tune_config).

Companion to ``row_sum_megakernel.py``.  Whereas the base example uses
hard-coded ``BLOCK_M=32, BLOCK_K=32, num_warps=4, num_stages=2``, this
example lets Triton search for the best combination automatically via
:attr:`MegakernelLoweringSpec.tune_config`.

The autotune sweep covers two Triton launch-param axes:

    ============== ==============================
    Axis            Values
    ============== ==============================
    num_warps       2, 4, 8
    num_stages      1, 2, 3, 4
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
            "num_warps":   (2, 4, 8),
            "num_stages":  (1, 2, 3, 4),
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
    print(f"  tune_config: num_warps={2,4,8}, num_stages={1,2,3,4}")
    print(f"  (3×4 = 12 configs; Triton benchmarks on first launch)")
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

    # ── Default (hardcoded) for comparison ──
    print("-" * 60)
    print("  Default (hardcoded) megakernel for comparison")
    print("-" * 60)

    from examples.event_tensor.row_sum_megakernel import compile_megakernel, run_megakernel

    compiled_default = compile_megakernel(
        n_row_blocks=N_ROW_BLOCKS, j_chunks=J_CHUNKS,
        block_m=BLOCK_M, block_k=BLOCK_K,
    )
    print(f"  Decorator: @triton.jit")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_K={BLOCK_K}")
    print(f"  num_warps=4, num_stages=2")
    print()

    _ = run_megakernel(compiled_default, a)
    default_ms = do_bench(lambda: run_megakernel(compiled_default, a))
    print(f"  default megakernel:    {default_ms:.3f} ms")
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
    if default_ms > 0:
        speedup = default_ms / auto_ms
        print(f"\n  Autotune vs default speedup: {speedup:.2f}x")
