"""Tests for the persistent-Triton megakernel emitter."""

from __future__ import annotations

import pytest
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
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region


def _build_gemm_rs(sm_count: int = 4) -> tuple[ModuleOp, GraphOp]:
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([4]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("partial_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "out_edges": ArrayAttr([EventCoordAttr("E", [str(i)], 1) for i in range(4)]),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "in_edges": ArrayAttr([EventCoordAttr("E", [str(i)], 1) for i in range(4)]),
            },
        ),
    )
    graph = GraphOp(sym_name="mm_rs", policy="static", sm_count=sm_count, body=Region([block]))
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_lowering_requires_static_schedule_annotation() -> None:
    _, graph = _build_gemm_rs()
    with pytest.raises(ValueError, match="static_schedule"):
        lower_megakernel(graph)


def test_lowering_returns_diagnostic_when_schedule_was_rejected() -> None:
    _, graph = _build_gemm_rs()
    graph.attributes["compgen.static_schedule"] = StringAttr('{"status": "rejected", "errors": ["bogus"]}')
    result = lower_megakernel(graph)
    assert result.kernel_source == ""
    assert any("rejected" in d for d in result.diagnostics)


# ---------------------------------------------------------------------------
# End-to-end lowering surface
# ---------------------------------------------------------------------------


def test_lowering_produces_named_persistent_kernel() -> None:
    mod, graph = _build_gemm_rs(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert isinstance(result, MegakernelLoweringResult)
    assert result.kernel_name == "megakernel_mm_rs"
    src = result.kernel_source
    assert "@triton.jit" in src
    assert "def megakernel_mm_rs(" in src
    assert "tl.program_id(0)" in src
    assert "while task_idx < qlen" in src


def test_lowering_grid_matches_sm_count() -> None:
    mod, graph = _build_gemm_rs(sm_count=8)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert result.launch_config["grid"] == 8


def test_lowering_emits_event_pointer_arg_per_event() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert "E_ptr" in result.kernel_source
    assert result.event_layout[0]["name"] == "E"
    assert result.event_layout[0]["size"] == 4


def test_lowering_emits_atomic_notify_and_wait_helpers() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    assert "_event_notify" in src
    assert "tl.atomic_add" in src
    assert "_event_wait" in src
    assert "tl.atomic_or" in src


def test_lowering_emits_per_device_function_stubs() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    assert "_run_partial_sum" in src
    assert "_run_final_sum" in src


def test_lowering_task_queue_partitions_all_tasks_across_sms() -> None:
    mod, graph = _build_gemm_rs(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    flat = [tid for q in result.task_queue.values() for tid, _ in q]
    assert len(flat) == 8
    assert len(set(flat)) == 8  # no duplicates
    # Both functions appear.
    assert any(tid.startswith("partial_sum:") for tid in flat)
    assert any(tid.startswith("final_sum:") for tid in flat)


def test_lowering_kernel_source_is_syntactically_valid_python() -> None:
    """Compile the emitted source as Python (Triton decorators no-op when
    triton isn't actually loading the kernel) -- catches indentation /
    syntax slips early."""
    import ast

    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    ast.parse(src)


# ---------------------------------------------------------------------------
# Autotuning (tune_config)
# ---------------------------------------------------------------------------


def test_tune_config_empty_emits_plain_jit() -> None:
    """When tune_config is empty (default), emit @triton.jit."""
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    spec = MegakernelLoweringSpec(tune_config={})
    src = lower_megakernel(graph, spec=spec).kernel_source
    assert "@triton.jit" in src
    assert "@triton.autotune" not in src


def test_tune_config_non_empty_emits_autotune() -> None:
    """When tune_config has entries, emit @triton.autotune wrapping @triton.jit."""
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    spec = MegakernelLoweringSpec(
        tune_config={"num_warps": (2, 4), "num_stages": (1, 2)},
    )
    src = lower_megakernel(graph, spec=spec).kernel_source
    # Both decorators present, autotune wraps jit
    autotune_pos = src.index("@triton.autotune")
    jit_pos = src.index("@triton.jit", autotune_pos)
    assert autotune_pos < jit_pos, "@triton.autotune must wrap @triton.jit"
    # Device functions (_run_*, _event_*) still use @triton.jit
    assert "@triton.jit" in src[:autotune_pos]  # device funcs before megakernel


def test_tune_config_launch_params_only() -> None:
    """Only num_warps/num_stages axes → key falls back to SM_COUNT."""
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    spec = MegakernelLoweringSpec(
        tune_config={"num_warps": (2, 4), "num_stages": (1, 2, 3)},
    )
    src = lower_megakernel(graph, spec=spec).kernel_source
    assert "@triton.autotune" in src
    # key is required by Triton; SM_COUNT is the fallback
    assert "key=['SM_COUNT']" in src


def test_tune_config_constexpr_axes_add_key() -> None:
    """Constexpr axes in tune_config → emit key=[...]."""
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    spec = MegakernelLoweringSpec(
        constexpr_args=("BLOCK_M", "BLOCK_K"),
        tune_config={"BLOCK_M": (16, 32), "BLOCK_K": (32, 64)},
    )
    src = lower_megakernel(graph, spec=spec).kernel_source
    assert "@triton.autotune" in src
    assert "key=['BLOCK_M', 'BLOCK_K']" in src


def test_tune_config_cartesian_product_in_configs() -> None:
    """Configs list should contain the cartesian product of all axis values."""
    spec = MegakernelLoweringSpec(
        constexpr_args=("BLOCK_M",),
        tune_config={"BLOCK_M": (16, 32), "num_warps": (2, 4)},
    )
    from compgen.ir.tile.lower_megakernel import _build_autotune_configs

    src = _build_autotune_configs(spec)
    # 2 × 2 = 4 configs
    assert src.count("triton.Config(") == 4
    # Default num_warps fallback for unmatched keys
    assert "num_warps=2" in src
    assert "num_warps=4" in src


def test_tune_config_missing_axes_use_defaults() -> None:
    """num_stages not in tune_config → uses spec.num_stages default in every config."""
    spec = MegakernelLoweringSpec(
        num_stages=3,
        tune_config={"num_warps": (2, 4, 8)},
    )
    from compgen.ir.tile.lower_megakernel import _build_autotune_configs

    src = _build_autotune_configs(spec)
    # All 3 configs should have num_stages=3 (the default)
    assert src.count("num_stages=3") == 3


def test_tune_config_source_is_valid_python() -> None:
    """The full emitted source with @triton.autotune must parse as valid Python."""
    import ast

    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    spec = MegakernelLoweringSpec(
        constexpr_args=("BLOCK_M", "BLOCK_K"),
        tune_config={"BLOCK_M": (16, 32), "BLOCK_K": (32, 64), "num_warps": (2, 4)},
    )
    src = lower_megakernel(graph, spec=spec).kernel_source
    ast.parse(src)
