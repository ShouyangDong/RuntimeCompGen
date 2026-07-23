"""MLU CostModel — per-arch perf table for the roofline predictor.

MLU370 peak throughput numbers (conservative, from published specs):

- **BF16 Tensor Core (MFU)**: ~256 TFLOPS device-wide on MLU370-X4
  (96 cores). Per-core: ~2.67 TFLOPS.
- **FP32 SIMT**: ~32 TFLOPS device-wide. Per-core: ~0.33 TFLOPS.
- **FP16**: same as BF16 path (MFU handles both).

These numbers feed the universal ETC-vs-eager predictor
(``compgen.kernels.cost.predict_etc_dispatch``). In production,
these should be calibrated from a microbenchmark on first probe.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Per-core peak throughput (TFLOPS/s per compute core)
# ---------------------------------------------------------------------------

# FP32 SIMT peak per core. MLU370 cores run at ~1.0 GHz with
# 64 FP32 ops/cycle → ~64 GFLOPS/core. Conservative at 0.33 TFLOPS.
PEAK_FP32_TFLOPS_PER_CORE = 0.33

# BF16/FP16 MFU peak per core. MLU370's Matrix Function Unit
# delivers ~2.67 TFLOPS/core at BF16.
PEAK_BF16_TC_TFLOPS_PER_CORE = 2.67
PEAK_FP16_TC_TFLOPS_PER_CORE = 2.67

# FP8 peak (estimated — MLU590 roadmap; mlu370 doesn't have FP8).
PEAK_FP8_TC_TFLOPS_PER_CORE = 5.30  # placeholder

# ---------------------------------------------------------------------------
# Core counts per variant
# ---------------------------------------------------------------------------

CORE_COUNT_DEFAULT: dict[str, int] = {
    "mlu370": 48,       # MLU370-S4 / MLU370-M8
    "mlu370_x4": 96,    # MLU370-X4 (dual-die)
    "mlu590": 128,      # Next-gen placeholder
}

# ---------------------------------------------------------------------------
# Eager launch overhead (microseconds)
# ---------------------------------------------------------------------------

# CNRT eager kernel launch overhead. Comparable to CUDA's ~5-10 µs.
# Conservative at 10 µs.
EAGER_LAUNCH_OVERHEAD_US = 10.0


class MluCostModel:
    """Perf coefficients for MLU accelerators.

    Satisfies :class:`compgen.targets.gpu.contracts.GpuCostModel`.
    """

    def peak_tflops_per_sm(self, *, dtype: str, tensor_core: bool) -> float:
        """Per-core peak throughput at the given dtype.

        Args:
            dtype: One of ``"fp32"``, ``"fp16"``, ``"bf16"``, ``"fp8"``.
            tensor_core: If True, returns MFU throughput; otherwise
                SIMT throughput.
        """
        if tensor_core:
            if dtype == "fp32":
                return PEAK_FP32_TFLOPS_PER_CORE * 0.5  # FP32 on MFU is slower
            elif dtype in ("fp16", "float16"):
                return PEAK_FP16_TC_TFLOPS_PER_CORE
            elif dtype in ("bf16", "bfloat16"):
                return PEAK_BF16_TC_TFLOPS_PER_CORE
            elif dtype in ("fp8", "float8"):
                return PEAK_FP8_TC_TFLOPS_PER_CORE
            return PEAK_BF16_TC_TFLOPS_PER_CORE
        else:
            return PEAK_FP32_TFLOPS_PER_CORE

    def sm_count(self) -> int:
        """Number of compute cores on the current device.

        Defaults to mlu370 (48 cores). In production this should
        be probed via ``cnrtDeviceGetAttribute``.
        """
        return CORE_COUNT_DEFAULT.get("mlu370", 48)

    def scheduling_overhead_us(self) -> float:
        """Per-task megakernel scheduling overhead. MLU doesn't
        have cooperative-launch (no cluster-sync), so the overhead
        is mainly the per-task dispatch cost in CNRT. Conservative
        at 1.0 µs (same as NVIDIA's default)."""
        return 1.0

    def eager_launch_overhead_us(self) -> float:
        """CNRT eager-kernel launch cost. Conservative at 10 µs,
        matching NVIDIA's cuBLAS launch overhead as a baseline."""
        return EAGER_LAUNCH_OVERHEAD_US
