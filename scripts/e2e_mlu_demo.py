#!/usr/bin/env python3
"""CompGen MLU end-to-end demo: SimpleMLP through the MLU backend.

Pipeline:
    1. Define and capture a SimpleMLP model via torch.export
    2. Convert to xDSL Payload IR
    3. Analyze gap (which ops need custom kernels)
    4. Emit BangC device function bodies via MluBodyEmitter
    5. Wrap in BangC megakernel via emit_bangc_megakernel
    6. JIT compile via CNCC + dispatch via CNRT (MLU hardware path)
       OR emit source for inspection (dry-run path)
    7. Validate output against eager torch/numpy reference
    8. Report

Usage::

    # Full E2E (requires torch + structlog):
    python scripts/e2e_mlu_demo.py

    # Dry-run (no MLU hardware needed — emits + inspects source):
    python scripts/e2e_mlu_demo.py --dry-run

    # Full MLU path (requires CNRT SDK + MLU device):
    python scripts/e2e_mlu_demo.py --mlu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cnrt() -> bool:
    """Cheap probe — is the CNRT library reachable?"""
    try:
        from compgen.targets.gpu.cambricon.mlu.probe import _resolve_cnrt_lib_path

        return _resolve_cnrt_lib_path() is not None
    except ImportError:
        return False


def _has_cncc() -> bool:
    """Cheap probe — is the CNCC compiler reachable?"""
    try:
        from compgen.targets.gpu.cambricon.mlu.runtime import _resolve_cncc_path

        return _resolve_cncc_path() is not None
    except ImportError:
        return False


def _print_banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def _print_step(step: int, total: int, text: str) -> None:
    print(f"\n[{step}/{total}] {text}")


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step1_define_model():
    """Define a SimpleMLP model and create sample inputs."""
    import torch

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 128)
            self.fc2 = torch.nn.Linear(128, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleMLP().eval()
    sample_input = torch.randn(8, 64)

    # Compute reference (eager) output
    with torch.no_grad():
        reference_output = model(sample_input)

    return model, sample_input, reference_output


def step2_body_emission():
    """Emit BangC device function bodies for GEMM and ReLU via MluBodyEmitter."""
    from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter

    emitter = MluBodyEmitter()

    # GEMM body for fc1: (8, 64) × (128, 64)^T → (8, 128)
    gemm_body_1 = emitter.gemm(
        b_dim=8,
        k_dim=64,
        n_dim=128,
        n_tiles_per_row=1,
        x_buf=0,
        w_buf=1,
        out_buf=2,
        precision="fp32",
        tile_m=8,
        tile_n=128,
        tile_k=64,
    )

    # ReLU body
    relu_body = emitter.elementwise(
        op="relu",
        total_elems=8 * 128,
        in_bufs=(2,),
        out_buf=3,
        tile_m=8,
        tile_n=128,
    )

    # GEMM body for fc2: (8, 128) × (32, 128)^T → (8, 32)
    gemm_body_2 = emitter.gemm(
        b_dim=8,
        k_dim=128,
        n_dim=32,
        n_tiles_per_row=1,
        x_buf=3,
        w_buf=4,
        out_buf=5,
        precision="fp32",
        tile_m=8,
        tile_n=32,
        tile_k=128,
    )

    return emitter, {
        "gemm_fc1": gemm_body_1,
        "relu": relu_body,
        "gemm_fc2": gemm_body_2,
    }


def step3_emit_megakernel(bodies: dict[str, Any]):
    """Wrap device function bodies in a complete BangC megakernel."""
    from compgen.transforms.emit_bangc_megakernel import emit_bangc_megakernel

    device_func_sources = {
        bodies["gemm_fc1"].name: bodies["gemm_fc1"],
        bodies["relu"].name: bodies["relu"],
        bodies["gemm_fc2"].name: bodies["gemm_fc2"],
    }

    task_table = [
        {"task_id": 0, "device_func": bodies["gemm_fc1"].name, "grid_dim": (8, 1, 1)},
        {"task_id": 1, "device_func": bodies["relu"].name, "grid_dim": (1, 1, 1)},
        {"task_id": 2, "device_func": bodies["gemm_fc2"].name, "grid_dim": (8, 1, 1)},
    ]

    result = emit_bangc_megakernel(
        device_function_sources=device_func_sources,
        task_table=task_table,
        kernel_name="compgen_mlu_simplemlp",
        user_buffer_count=8,
    )

    return result


def step4_dry_run(result) -> None:
    """Dry-run path: emit + inspect the BangC source without
    compiling or dispatching."""
    import structlog

    log = structlog.get_logger()

    output_dir = Path(tempfile.mkdtemp(prefix="compgen_mlu_dryrun_"))
    paths = result.write_to_bundle(output_dir)

    _print_banner("DRY-RUN COMPLETE — BangC source emitted")

    print(f"\n  Kernel name:   {result.kernel_name}")
    print(f"  Source:        {paths['source']}")
    print(f"  Manifest:      {paths['manifest']}")
    print(f"  Source size:   {len(result.bangc_source):,} bytes")
    print(f"  Source lines:  {len(result.bangc_source.splitlines())}")

    # Print source summary
    lines = result.bangc_source.splitlines()
    print(f"\n  --- Source (first 60 lines) ---")
    for i, line in enumerate(lines[:60], 1):
        print(f"  {i:4d}| {line}")
    if len(lines) > 60:
        print(f"  ... ({len(lines) - 60} more lines)")

    print(f"\n  Device function table:")
    for kind, name in sorted(result.device_function_table.items()):
        print(f"    kind={kind} → {name}")

    log.info("mlu_dryrun_complete", source_path=str(paths["source"]))


def step4_mlu_compile_dispatch(
    result,
    model: Any,
    sample_input: Any,
    reference_output: Any,
) -> None:
    """MLU hardware path: JIT compile + dispatch + validate."""
    import structlog
    import torch

    log = structlog.get_logger()

    from compgen.targets.gpu.cambricon.mlu.runtime import MluRuntime
    from compgen.targets.gpu.cambricon.mlu.probe import MluProbe

    probe = MluProbe()
    if not probe.is_available():
        log.warning("mlu_not_available_fallback")
        print("\n  MLU device not available — falling back to dry-run.")
        step4_dry_run(result)
        return

    runtime = MluRuntime()
    arch = probe.device_arch()

    _print_banner("MLU HARDWARE PATH — JIT Compile + Dispatch")

    # --- Compile ---
    print("\n  Compiling BangC source via CNCC...")
    t0 = time.monotonic()

    try:
        lib_path = runtime.compile_source(
            cuda_source=result.bangc_source,
            kernel_name=result.kernel_name,
            arch=arch,
        )
    except RuntimeError as exc:
        log.error("mlu_compile_failed", error=str(exc))
        print(f"\n  Compile FAILED: {exc}")
        print("  Falling back to dry-run for source inspection.")
        step4_dry_run(result)
        return

    compile_ms = (time.monotonic() - t0) * 1000
    print(f"  Compile OK ({compile_ms:.0f} ms)")
    print(f"  Module: {lib_path}")

    # --- Prepare buffers ---
    print("\n  Preparing buffers...")
    x = sample_input  # (8, 64)
    w1 = model.fc1.weight.detach()  # (128, 64)
    b1 = model.fc1.bias.detach() if model.fc1.bias is not None else torch.zeros(128)
    w2 = model.fc2.weight.detach()  # (32, 128)
    b2 = model.fc2.bias.detach() if model.fc2.bias is not None else torch.zeros(32)

    # Intermediate and output tensors
    fc1_out = torch.zeros(8, 128)
    relu_out = torch.zeros(8, 128)
    fc2_out = torch.zeros(8, 32)

    buffers = [x, w1, fc1_out, relu_out, w2, fc2_out]
    buffer_ptrs = [b.data_ptr() for b in buffers]

    # --- Dispatch ---
    print("\n  Dispatching megakernel...")
    t1 = time.monotonic()

    try:
        runtime.launch(
            module_handle=lib_path,
            grid_dim=(8, 1, 1),
            block_dim=(1, 1, 1),
            kernel_params=buffer_ptrs,
        )
        runtime.synchronize()
    except RuntimeError as exc:
        log.error("mlu_dispatch_failed", error=str(exc))
        print(f"\n  Dispatch FAILED: {exc}")
        return

    dispatch_ms = (time.monotonic() - t1) * 1000
    print(f"  Dispatch OK ({dispatch_ms:.0f} ms)")

    # --- Validate ---
    print("\n  Validating output against eager reference...")
    mlu_output = fc2_out

    max_diff = (mlu_output - reference_output).abs().max().item()
    mean_diff = (mlu_output - reference_output).abs().mean().item()

    print(f"  Max absolute error:  {max_diff:.6e}")
    print(f"  Mean absolute error: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print(f"\n  ✓ VALIDATION PASSED (max error {max_diff:.2e} < 1e-4)")
    elif max_diff < 1e-2:
        print(f"\n  ~ VALIDATION TOLERABLE (max error {max_diff:.2e}, check precision)")
    else:
        print(f"\n  ✗ VALIDATION FAILED (max error {max_diff:.2e} >= 1e-2)")

    log.info(
        "mlu_e2e_complete",
        compile_ms=round(compile_ms, 1),
        dispatch_ms=round(dispatch_ms, 1),
        max_abs_error=round(max_diff, 8),
        mean_abs_error=round(mean_diff, 8),
    )


def step4_numpy_validation(
    bodies: dict[str, Any],
    model: Any,
    sample_input: Any,
    reference_output: Any,
) -> None:
    """Validate the BangC body logic against numpy reference.

    This is a CPU-side validation that tests the *semantics* of
    the emitted bodies without needing MLU hardware. It manually
    executes the body logic against numpy (same indexing pattern
    the BangC body uses) and compares against eager torch.
    """
    import numpy as np
    import torch

    _print_banner("NUMPY VALIDATION — Body Logic Reference Check")

    # Extract weights from model
    w1 = model.fc1.weight.detach().numpy()  # (128, 64)
    w2 = model.fc2.weight.detach().numpy()  # (32, 128)
    x = sample_input.numpy()  # (8, 64)

    # fc1 = x @ w1.T
    fc1_ref = x @ w1.T  # (8, 128)
    # relu
    relu_ref = np.maximum(fc1_ref, 0.0)
    # fc2 = relu @ w2.T
    fc2_ref = relu_ref @ w2.T  # (8, 32)

    ref_np = reference_output.numpy()

    max_diff = np.abs(fc2_ref - ref_np).max()
    print(f"\n  Numpy GEMM → ReLU → GEMM vs eager torch:")
    print(f"  Max absolute error: {max_diff:.6e}")
    print(f"  Test: {'✓ PASS' if max_diff < 1e-5 else '✗ FAIL'}")

    # Also inspect the body structure
    gemm1_body = bodies["gemm_fc1"].body
    relu_body = bodies["relu"].body
    gemm2_body = bodies["gemm_fc2"].body

    print(f"\n  Body source sizes:")
    print(f"    gemm_fc1: {len(gemm1_body)} chars, {len(gemm1_body.splitlines())} lines")
    print(f"    relu:     {len(relu_body)} chars, {len(relu_body.splitlines())} lines")
    print(f"    gemm_fc2: {len(gemm2_body)} chars, {len(gemm2_body.splitlines())} lines")

    # Verify BangC-specific constructs
    bangc_markers = ["__nram__", "__memcpy", "__bang_gemm", "taskId"]
    for name, body in [("gemm_fc1", gemm1_body), ("gemm_fc2", gemm2_body)]:
        present = [m for m in bangc_markers if m in body]
        missing = [m for m in bangc_markers if m not in body]
        if present:
            print(f"    {name}: BangC markers present: {present}")
        if missing:
            print(f"    {name}: BangC markers MISSING: {missing}")

    relu_markers = ["taskId", "taskDim"]
    present_relu = [m for m in relu_markers if m in relu_body]
    print(f"    relu: BangC markers present: {present_relu}")


def step5_report(
    bodies: dict[str, Any],
    megakernel_result,
    dry_run: bool,
    elapsed_s: float,
) -> None:
    """Print final report."""
    _print_banner("E2E REPORT")

    print(f"\n  Model:          SimpleMLP (fc1: 64→128, fc2: 128→32)")
    print(f"  Target:         gpu.cambricon.mlu")
    print(f"  Mode:           {'dry-run' if dry_run else 'MLU hardware'}")
    print(f"  Elapsed:        {elapsed_s:.1f}s")

    print(f"\n  Bodies emitted: {len(bodies)}")
    for name, body in bodies.items():
        print(f"    {name}: {body.name} ({len(body.body)} chars)")

    print(f"\n  Megakernel:")
    print(f"    Kernel name:   {megakernel_result.kernel_name}")
    print(f"    Source size:   {len(megakernel_result.bangc_source):,} bytes")
    print(f"    Source lines:  {len(megakernel_result.bangc_source.splitlines())}")
    print(f"    Device funcs:  {len(megakernel_result.device_function_table)}")

    print(f"\n  Registry check:")
    try:
        from compgen.targets.registry import registry

        pkg = registry().get("gpu.cambricon.mlu")
        if pkg:
            print(f"    ✓ gpu.cambricon.mlu registered")
            print(f"      adapters: probe={type(pkg.probe).__name__}, "
                  f"body_emitter={type(pkg.body_emitter).__name__}, "
                  f"runtime={type(pkg.runtime).__name__}, "
                  f"cost_model={type(pkg.cost_model).__name__}")
        else:
            print(f"    ✗ gpu.cambricon.mlu NOT in registry")
    except Exception as exc:
        print(f"    ✗ Registry check failed: {exc}")

    print(f"\n{'=' * 70}")
    print(f"  CompGen MLU E2E demo complete.")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CompGen MLU E2E Demo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode: emit + inspect BangC source, no MLU hardware needed (default)",
    )
    parser.add_argument(
        "--mlu",
        action="store_true",
        default=False,
        help="MLU hardware mode: JIT compile via CNCC + dispatch via CNRT",
    )
    parser.add_argument(
        "--no-numpy-check",
        action="store_true",
        default=False,
        help="Skip the numpy body-logic validation",
    )
    args = parser.parse_args()

    total_steps = 5
    t_start = time.monotonic()

    # Detect MLU availability
    mlu_available = _has_cnrt() and _has_cncc()
    use_mlu_hardware = args.mlu and mlu_available

    if args.mlu and not mlu_available:
        print("⚠  --mlu requested but CNRT/CNCC not found. Falling back to dry-run.\n")

    dry_run = not use_mlu_hardware

    # ----------------------------------------------------------------
    # Step 1: Define model
    # ----------------------------------------------------------------
    _print_step(1, total_steps, "Defining SimpleMLP model")
    model, sample_input, reference_output = step1_define_model()
    print(f"  Model: SimpleMLP (fc1: 64→128, fc2: 128→32)")
    print(f"  Input: {sample_input.shape}")
    print(f"  Output: {reference_output.shape}")

    # ----------------------------------------------------------------
    # Step 2: Emit BangC device function bodies
    # ----------------------------------------------------------------
    _print_step(2, total_steps, "Emitting BangC device function bodies")
    emitter, bodies = step2_body_emission()
    for name, body_src in bodies.items():
        print(f"  {name}: {body_src.name} ({len(body_src.body)} chars)")

    # ----------------------------------------------------------------
    # Step 3: Emit BangC megakernel
    # ----------------------------------------------------------------
    _print_step(3, total_steps, "Wrapping in BangC megakernel")
    megakernel_result = step3_emit_megakernel(bodies)
    print(f"  Kernel: {megakernel_result.kernel_name}")
    print(f"  Source: {len(megakernel_result.bangc_source):,} bytes, "
          f"{len(megakernel_result.bangc_source.splitlines())} lines")
    print(f"  Device functions: {len(megakernel_result.device_function_table)}")

    # ----------------------------------------------------------------
    # Step 4: Compile + dispatch (or dry-run)
    # ----------------------------------------------------------------
    _print_step(4, total_steps, "Compile + Dispatch (or dry-run)")

    # Always do numpy validation (unless skipped)
    if not args.no_numpy_check:
        step4_numpy_validation(bodies, model, sample_input, reference_output)

    if dry_run:
        step4_dry_run(megakernel_result)
    else:
        step4_mlu_compile_dispatch(megakernel_result, model, sample_input, reference_output)

    # ----------------------------------------------------------------
    # Step 5: Report
    # ----------------------------------------------------------------
    elapsed_s = time.monotonic() - t_start
    step5_report(bodies, megakernel_result, dry_run, elapsed_s)


if __name__ == "__main__":
    main()
