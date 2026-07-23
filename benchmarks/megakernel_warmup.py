"""Warmup-cost benchmark: CompGen AOT megakernel vs PyTorch JIT baselines.

Uses ``triton.testing.do_bench`` for accurate GPU kernel timing
(instead of wall-clock which includes Python overhead).  Wall-clock
is still used for the cold path where Triton compilation dominates.

Mirrors the structure of Table 1 of the Event Tensor Compiler paper
(``vLLM JIT 123 s`` / ``SGLang JIT 583 s`` / ``ETC AOT 35 s`` for
Qwen3-32B).

Run as::

    python -m benchmarks.megakernel_warmup
"""

from __future__ import annotations

import gc
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


def _accel_empty_cache() -> None:
    if _HAS_CUDA:
        torch.cuda.empty_cache()

from triton.testing import do_bench


def _wall_clock() -> float:
    _accel_sync()
    return time.perf_counter()


from examples.event_tensor.tinyllama_layer_megakernel import (
    DEFAULT_SEQ_LEN,
    compile_for_tinyllama,
    load_tinyllama_layer0,
    project_qkv,
    slice_weights_for_megakernel,
)
from examples.event_tensor.transformer_block_megakernel import (
    reference_block,
    run_transformer_block_megakernel,
)


@dataclass
class WarmupResult:
    label: str
    cold_seconds: float          # wall-clock: emit + compile + first launch
    warm_wall_seconds: float     # wall-clock: re-emit + launch (cache hit)
    kernel_ms: float             # do_bench: pure GPU kernel time (warm)
    compile_seconds: float       # wall-clock: just Triton compilation
    description: str


def _purge_triton_cache() -> None:
    cache = Path(os.path.expanduser("~/.triton/cache"))
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)


def measure_megakernel_aot(
    weights, sliced_cfg, x, *, label: str = "megakernel_aot",
) -> WarmupResult:
    _purge_triton_cache()
    gc.collect()
    _accel_empty_cache()

    cfg = sliced_cfg
    n_heads, hidden, intermediate = cfg["n_heads"], cfg["hidden_dim"], cfg["intermediate"]

    # --- Phase 1: autotune (includes emit + compile for each tile config) ---
    t0 = _wall_clock()
    compiled = compile_for_tinyllama(
        n_heads=n_heads, seq_len=DEFAULT_SEQ_LEN,
        head_dim=hidden // n_heads, intermediate_dim=intermediate,
    )
    t_compile = _wall_clock() - t0
    cold = t_compile

    # --- Warm: benchmark kernel only (autotune result is cached, instant) ---
    q, k, v = project_qkv(x, weights, sliced_cfg)
    _ = run_transformer_block_megakernel(
        compiled, q, k, v, x, weights.w_gate, weights.w_up, weights.w_down,
    )
    warm_wall = 0  # negligible after autotune

    def _mk_fn():
        run_transformer_block_megakernel(
            compiled, q, k, v, x, weights.w_gate, weights.w_up, weights.w_down,
        )
    kernel_ms = do_bench(_mk_fn)

    return WarmupResult(
        label=label, cold_seconds=cold, warm_wall_seconds=warm_wall,
        kernel_ms=kernel_ms, compile_seconds=t_compile,
        description=f"autotune+compile={t_compile:.1f}s, kernel={kernel_ms:.3f}ms",
    )


def measure_torch_compile_jit(
    weights, sliced_cfg, x, *, label: str = "torch.compile_jit",
) -> WarmupResult:
    def block(x_in, q, k, v, wg, wu, wd):
        return reference_block(q, k, v, x_in, wg, wu, wd)

    gc.collect()
    _accel_empty_cache()
    q, k, v = project_qkv(x, weights, sliced_cfg)

    try:
        torch._dynamo.reset()
    except Exception:
        pass

    compiled_block = torch.compile(block, dynamic=False)

    # Cold: first call (Dynamo trace + Inductor lower + Triton compile)
    t0 = _wall_clock()
    _ = compiled_block(x, q, k, v, weights.w_gate, weights.w_up, weights.w_down)
    cold = _wall_clock() - t0

    # Warm wall-clock
    t0 = _wall_clock()
    _ = compiled_block(x, q, k, v, weights.w_gate, weights.w_up, weights.w_down)
    warm_wall = _wall_clock() - t0

    # Pure GPU kernel time via do_bench
    def _jit_fn():
        compiled_block(x, q, k, v, weights.w_gate, weights.w_up, weights.w_down)
    kernel_ms = do_bench(_jit_fn)

    return WarmupResult(
        label=label, cold_seconds=cold, warm_wall_seconds=warm_wall,
        kernel_ms=kernel_ms, compile_seconds=cold - kernel_ms / 1000.0,
        description=f"compile={cold:.2f}s, kernel={kernel_ms:.3f}ms",
    )


def measure_eager_baseline(
    weights, sliced_cfg, x, *, label: str = "eager_pytorch",
) -> WarmupResult:
    """Measure eager PyTorch reference block (no compilation at all)."""
    q, k, v = project_qkv(x, weights, sliced_cfg)
    wg, wu, wd = weights.w_gate, weights.w_up, weights.w_down

    # Warmup once
    _ = reference_block(q, k, v, x, wg, wu, wd)
    _accel_sync()

    def _eager_fn():
        reference_block(q, k, v, x, wg, wu, wd)
    kernel_ms = do_bench(_eager_fn)

    return WarmupResult(
        label=label, cold_seconds=0, warm_wall_seconds=0,
        kernel_ms=kernel_ms, compile_seconds=0,
        description=f"eager PyTorch, kernel={kernel_ms:.3f}ms",
    )


def main() -> None:
    if not _HAS_ACCEL:
        raise SystemExit(f"This benchmark requires an accelerator ({_ACCEL_TAG} not available).")

    print(f"Loading TinyLlama-1.1B layer-0 weights ... (device={_ACCEL_DEVICE})")
    full = load_tinyllama_layer0(_ACCEL_DEVICE)
    sliced, sliced_cfg = slice_weights_for_megakernel(full)
    print(
        f"  workload: H={sliced_cfg['n_heads']}, "
        f"D={sliced_cfg['hidden_dim']}, I={sliced_cfg['intermediate']}, "
        f"S={DEFAULT_SEQ_LEN}"
    )

    torch.manual_seed(123)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device=_ACCEL_DEVICE,
        )
        * 0.1
    )

    print("\n[1/3] Measuring eager PyTorch baseline ...")
    eager = measure_eager_baseline(sliced, sliced_cfg, x)
    print(f"  kernel (do_bench):    {eager.kernel_ms:7.3f} ms")

    print("\n[2/3] Measuring CompGen megakernel AOT path ...")
    aot = measure_megakernel_aot(sliced, sliced_cfg, x)
    print(f"  cold (emit+compile):  {aot.cold_seconds:7.3f} s")
    print(f"  compile-only:         {aot.compile_seconds:7.3f} s")
    print(f"  warm (wall-clock):    {aot.warm_wall_seconds:7.3f} s")
    print(f"  kernel (do_bench):    {aot.kernel_ms:7.3f} ms")

    print("\n[3/3] Measuring torch.compile JIT baseline ...")
    jit = measure_torch_compile_jit(sliced, sliced_cfg, x)
    print(f"  cold (first call):    {jit.cold_seconds:7.3f} s")
    print(f"  warm (wall-clock):    {jit.warm_wall_seconds:7.3f} s")
    print(f"  kernel (do_bench):    {jit.kernel_ms:7.3f} ms")

    print("\n=== summary ===")
    print(f"{'path':28}  {'cold(s)':>8}  {'compile(s)':>10}  {'kernel(ms)':>10}")
    print(f"{eager.label:28}  {'—':>8}  {'—':>10}  {eager.kernel_ms:10.3f}")
    print(f"{aot.label:28}  {aot.cold_seconds:8.3f}  {aot.compile_seconds:10.3f}  {aot.kernel_ms:10.3f}")
    print(f"{jit.label:28}  {jit.cold_seconds:8.3f}  {jit.compile_seconds:10.3f}  {jit.kernel_ms:10.3f}")

    if eager.kernel_ms > 0:
        print(f"\nKernel time ratios (vs eager={eager.kernel_ms:.3f}ms):")
        print(f"  megakernel  / eager: {aot.kernel_ms / eager.kernel_ms:.2f}x")
        print(f"  torch.compile / eager: {jit.kernel_ms / eager.kernel_ms:.2f}x")
        print(f"  megakernel / torch.compile: {aot.kernel_ms / jit.kernel_ms:.2f}x" if jit.kernel_ms > 0 else "")

    if aot.cold_seconds > 0:
        speedup = jit.cold_seconds / aot.cold_seconds
        print(f"\nCold-start speedup (megakernel vs torch.compile): {speedup:.2f}x")
        if speedup >= 1.0:
            print("AOT wins on cold-start (less compile overhead).")
        else:
            print(
                "AOT loses on cold-start — Triton compilation dominates. "
                "The AOT advantage materializes at scale (paper: 35s vs 583s for Qwen3-32B)."
            )


if __name__ == "__main__":
    main()
