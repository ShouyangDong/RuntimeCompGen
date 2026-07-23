"""MLU target acceptance tests.

Follows the same pattern as ``test_cpu_x86_stub.py`` — the gold
standard for target-adapter acceptance tests.

Since MLU hardware + CNRT SDK may not be available in CI, tests
are structured in two tiers:

1. **Always-run** (no MLU hardware needed):
   - Protocol satisfaction (isinstance checks).
   - Body emitter produces well-formed BangC source.
   - Registration via ``compgen.targets`` import.
   - Audit metadata pinned.

2. **MLU-required** (skipped without CNRT + MLU device):
   - E2E JIT compile + dispatch round-trip.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_cnrt() -> bool:
    """Cheap probe — is CNRT reachable? (May still have no device.)"""
    from compgen.targets.gpu.cambricon.mlu.probe import _resolve_cnrt_lib_path

    return _resolve_cnrt_lib_path() is not None


@pytest.fixture(autouse=True)
def _ensure_registered():
    import compgen.targets as targets_mod

    targets_mod._register_in_tree()
    yield


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


class TestProbe:
    def test_probe_satisfies_gpu_protocol(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe
        from compgen.targets.gpu.contracts import GpuProbe

        p = MluProbe()
        assert isinstance(p, GpuProbe)

    def test_probe_basic_contract(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

        p = MluProbe()
        # All calls return without raising.
        assert isinstance(p.is_available(), bool)
        assert isinstance(p.device_arch(), str)
        assert p.device_arch().startswith("mlu")
        assert isinstance(p.supports_clusters(), bool)
        assert isinstance(p.supports_tensor_cores(), bool)
        assert isinstance(p.library_paths(), dict)
        assert isinstance(p.vendor_extras(), dict)

    def test_no_clusters(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

        p = MluProbe()
        assert p.supports_clusters() is False

    def test_has_tensor_cores(self) -> None:
        """MLU's MFU is analogous to tensor cores."""
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

        p = MluProbe()
        assert p.supports_tensor_cores() is True

    def test_vendor_extras_carries_metadata(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

        extras = MluProbe().vendor_extras()
        assert extras["triton_compatible"] is True
        assert extras["kernel_language"] == "BangC"
        assert extras["jit_compiler"] == "CNCC"


# ---------------------------------------------------------------------------
# Body emitter — source quality checks
# ---------------------------------------------------------------------------


class TestBodyEmitter:
    def test_body_emitter_satisfies_gpu_protocol(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter
        from compgen.targets.gpu.contracts import GpuBodyEmitter

        assert isinstance(MluBodyEmitter(), GpuBodyEmitter)

    def test_preferred_tile_shape(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

        emitter = MluBodyEmitter()
        # BF16 uses MFU-friendly tile
        assert emitter.preferred_tile_shape(op="gemm", dtype="bf16") == (64, 64, 16)
        # FP32 uses smaller tile for register pressure
        assert emitter.preferred_tile_shape(op="gemm", dtype="fp32") == (32, 32, 32)

    def test_gemm_emits_bangc_source(self) -> None:
        """GEMM body contains BangC-specific constructs."""
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

        result = MluBodyEmitter().gemm(
            b_dim=32,
            k_dim=32,
            n_dim=32,
            n_tiles_per_row=1,
            x_buf=0,
            w_buf=1,
            out_buf=2,
            precision="bf16_fp32",
            tile_m=64,
            tile_n=64,
            tile_k=16,
        )
        body = result.body

        # BangC-specific constructs
        assert "__nram__" in body, "Should use NRAM for on-chip memory"
        assert "__memcpy" in body, "Should use BangC memcpy intrinsics"
        assert "__bang_gemm" in body, "Should use MFU gemm intrinsic"
        assert "taskId" in body, "Should use BangC task model"
        assert "buffers[0]" in body
        assert "buffers[1]" in body
        assert "buffers[2]" in body

        # Metadata
        assert result.name == "mlu_gemm"
        assert "#include <bang.h>" in result.included_headers

    def test_relu_emits_bangc_source(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

        result = MluBodyEmitter().elementwise(
            op="relu",
            total_elems=128,
            in_bufs=(0,),
            out_buf=1,
            tile_m=32,
            tile_n=32,
        )
        body = result.body

        assert "taskId" in body
        assert "taskDim" in body
        assert "val > 0.0f" in body
        assert result.name == "mlu_elementwise_relu"

    def test_gelu_emits_bangc_source(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

        result = MluBodyEmitter().elementwise(
            op="gelu",
            total_elems=128,
            in_bufs=(0,),
            out_buf=1,
            tile_m=32,
            tile_n=32,
        )
        body = result.body
        assert "__bang_tanh" in body, "GELU should use BangC tanh intrinsic"

    def test_add_emits_bangc_source(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

        result = MluBodyEmitter().elementwise(
            op="add",
            total_elems=128,
            in_bufs=(0, 1),
            out_buf=2,
            tile_m=32,
            tile_n=32,
        )
        body = result.body
        assert "in0[i] + in1[i]" in body


# ---------------------------------------------------------------------------
# Runtime — structural checks (no MLU hardware needed)
# ---------------------------------------------------------------------------


class TestRuntime:
    def test_runtime_satisfies_gpu_protocol(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.runtime import MluRuntime
        from compgen.targets.gpu.contracts import GpuRuntime

        assert isinstance(MluRuntime(), GpuRuntime)

    def test_synchronize_noop_without_init(self) -> None:
        """Synchronize is safe to call without prior initialization."""
        from compgen.targets.gpu.cambricon.mlu.runtime import MluRuntime

        rt = MluRuntime()
        # Should not raise even without MLU hardware
        rt.synchronize()


# ---------------------------------------------------------------------------
# Cost model — structural checks
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_cost_model_satisfies_gpu_protocol(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.cost import MluCostModel
        from compgen.targets.gpu.contracts import GpuCostModel

        assert isinstance(MluCostModel(), GpuCostModel)

    def test_peak_tflops_per_sm(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.cost import MluCostModel

        cm = MluCostModel()
        # BF16 tensor core
        bf16_tc = cm.peak_tflops_per_sm(dtype="bf16", tensor_core=True)
        assert bf16_tc > 1.0, "BF16 MFU should deliver >1 TFLOPS/core"

        # FP32 SIMT
        fp32_simt = cm.peak_tflops_per_sm(dtype="fp32", tensor_core=False)
        assert fp32_simt > 0.1, "FP32 SIMT should be reasonable"
        assert fp32_simt < bf16_tc, "BF16 MFU should exceed FP32 SIMT"

    def test_sm_count(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.cost import MluCostModel

        cm = MluCostModel()
        assert cm.sm_count() >= 1

    def test_overheads(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.cost import MluCostModel

        cm = MluCostModel()
        assert cm.scheduling_overhead_us() >= 0.0
        assert cm.eager_launch_overhead_us() > 0.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_mlu_vendor_registered(self) -> None:
        from compgen.targets.registry import registry

        pkg = registry().get("gpu.cambricon")
        assert pkg is not None
        assert pkg.target_class == "gpu"
        assert pkg.vendor == "cambricon"

    def test_mlu_arch_registered(self) -> None:
        from compgen.targets.registry import registry

        pkg = registry().get("gpu.cambricon.mlu")
        assert pkg is not None
        assert pkg.target_id == "gpu.cambricon.mlu"
        # All four adapters wired
        assert pkg.probe is not None
        assert pkg.body_emitter is not None
        assert pkg.runtime is not None
        assert pkg.cost_model is not None

    def test_registry_tree_includes_cambricon(self) -> None:
        from compgen.targets.registry import registry

        tree = registry().tree()
        assert "cambricon" in tree.get("gpu", {}), (
            f"cambricon should appear under gpu in registry tree, got: "
            f"{list(tree.get('gpu', {}).keys())}"
        )

    def test_audit_metadata_pinned(self) -> None:
        """Audit metadata visible via describe() — pin so surface
        doesn't drift."""
        from compgen.targets.registry import registry

        pkg = registry().get("gpu.cambricon.mlu")
        assert pkg is not None
        m = pkg.metadata
        assert m["supports_clusters"] is False
        assert m["supports_tensor_cores"] is True
        assert m["default_tile_shape"] == [64, 64, 16]
        assert m["preferred_precision"] == "bf16_fp32"
        assert "core_count" in m

    def test_describe_returns_dict(self) -> None:
        from compgen.targets.registry import registry

        pkg = registry().get("gpu.cambricon.mlu")
        assert pkg is not None
        d = pkg.to_dict()
        assert d["target_id"] == "gpu.cambricon.mlu"
        assert "adapters" in d


# ---------------------------------------------------------------------------
# E2E — requires MLU hardware + CNRT SDK
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_cnrt(),
    reason="CNRT not reachable — skip MLU JIT tests on CPU-only hosts",
)
class TestEndToEndJIT:
    """End-to-end JIT compilation + dispatch on real MLU hardware."""

    def test_cnrt_probe_reports_available(self) -> None:
        from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

        p = MluProbe()
        assert p.is_available(), (
            "CNRT is installed but probe says MLU is unavailable — "
            "check device count"
        )
