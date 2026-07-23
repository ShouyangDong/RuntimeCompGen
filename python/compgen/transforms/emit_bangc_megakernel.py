"""BangC megakernel emitter — MLU analogue of ``emit_cuda_megakernel.py``.

Takes a set of ``__mlu_func__`` device function bodies (emitted by
:class:`MluBodyEmitter`) and wraps them in a self-contained BangC
kernel source that can be JIT-compiled via CNCC and dispatched via
CNRT.

The emitted BangC kernel follows Cambricon's programming model:

- ``__mlu_entry__`` — the top-level kernel (analogous to CUDA ``__global__``).
- ``taskId`` — per-task index within the core (like ``threadIdx``).
- ``taskDim`` — total tasks per core (like ``blockDim``).
- ``coreId`` — which MLU core this task is running on (like ``blockIdx``).

For the initial MLU backend, the megakernel is a simple serial dispatcher:
each task runs its assigned device function in order. The full
event-tensor scheduling layer (peer-to-peer synchronization, SM queues,
cooperative launch) is a future step.

The output is a complete BangC source string + a simple manifest,
analogous to :class:`CudaMegakernelEmitResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BangcMegakernelEmitResult:
    """Result of BangC megakernel emission.

    Attributes:
        kernel_name: Symbol of the emitted ``__mlu_entry__`` function.
        bangc_source: Complete BangC source string. Pass to CNCC
            for JIT compilation.
        manifest: Structured launch / buffer metadata consumed by
            the runtime launcher.
        device_function_table: ``kind → device_func name`` mapping.
    """

    kernel_name: str
    bangc_source: str
    manifest: dict[str, Any]
    device_function_table: dict[int, str] = field(default_factory=dict)

    def write_to_bundle(self, bundle_dir: Path | str) -> dict[str, Path]:
        """Stage source + manifest into a bundle directory."""
        out = Path(bundle_dir)
        out.mkdir(parents=True, exist_ok=True)
        src_path = out / "kernel.mlu"
        man_path = out / "manifest.json"
        src_path.write_text(self.bangc_source)
        man_path.write_text(_json_dumps(self.manifest))
        return {"source": src_path, "manifest": man_path}


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def emit_bangc_megakernel(
    *,
    device_function_sources: dict[str, DeviceFunctionSource],
    task_table: list[dict[str, Any]],
    kernel_name: str = "compgen_mlu_megakernel",
    user_buffer_count: int = 4,
    extra_includes: tuple[str, ...] = (),
) -> BangcMegakernelEmitResult:
    """Emit a complete BangC megakernel from device function bodies.

    Args:
        device_function_sources: Map of ``device_func`` name →
            :class:`DeviceFunctionSource`. Each body must be valid
            BangC (``__mlu_func__``, NRAM intrinsics, etc.).
        task_table: Ordered list of tasks. Each task is a dict with:
            - ``task_id`` (int)
            - ``device_func`` (str) — must be in device_function_sources
            - ``grid_dim`` (tuple[int,int,int]) — tasks per core
        kernel_name: Symbol name for the ``__mlu_entry__`` function.
        user_buffer_count: Number of ``void *`` buffer pointers the
            kernel receives.
        extra_includes: Additional ``#include`` lines.

    Returns:
        :class:`BangcMegakernelEmitResult`.
    """

    # Stable kind → name mapping
    distinct_names = sorted(set(t["device_func"] for t in task_table))
    missing = [n for n in distinct_names if n not in device_function_sources]
    if missing:
        raise MegakernelEmitError(
            f"Task table references device_func(s) without a body: "
            + ", ".join(sorted(missing))
        )
    name_to_kind = {n: i for i, n in enumerate(distinct_names)}
    kind_to_name = {i: n for n, i in name_to_kind.items()}

    # --- Build source ---

    # Gather all needed includes
    all_includes: set[str] = set()
    # BangC standard headers
    all_includes.add("#include <bang.h>")
    for extra in extra_includes:
        all_includes.add(extra)
    for src in device_function_sources.values():
        for h in src.included_headers:
            all_includes.add(h)

    header_block = "\n".join(sorted(all_includes))

    # Device function bodies
    device_func_blocks: list[str] = []
    for src in device_function_sources.values():
        sig = src.signature if src.signature else "int task_id, int sm_id, void **buffers"
        device_func_blocks.append(
            f"__mlu_func__ void {src.name}({sig}) {{\n{src.body}\n}}"
        )

    device_funcs_source = "\n\n".join(device_func_blocks)

    # Task descriptor table (constant array)
    task_entries: list[str] = []
    for t in task_table:
        kind = name_to_kind[t["device_func"]]
        gx, gy, gz = t.get("grid_dim", (1, 1, 1))
        task_entries.append(f"    {{{kind}, {gx}, {gy}, {gz}}}")

    # Dispatch switch
    dispatch_cases: list[str] = []
    for i, name in kind_to_name.items():
        dispatch_cases.append(f"    case {i}: {name}(task_id, sm_id, buffers); break;")

    total_tasks = len(task_table)

    source = f"""\
// ============================================================================
// CompGen BangC Megakernel — auto-generated
// Target: Cambricon MLU
// Kernel: {kernel_name}
// Tasks: {total_tasks}
// Device functions: {len(distinct_names)}
// ============================================================================

{header_block}

// ---------------------------------------------------------------------------
// Device function bodies
// ---------------------------------------------------------------------------

{device_funcs_source}

// ---------------------------------------------------------------------------
// Task descriptor table (constant)
// ---------------------------------------------------------------------------

typedef struct {{
    int kind;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
}} TaskDescriptor;

__mlu_constant__ TaskDescriptor task_table[{total_tasks}] = {{
{chr(10).join(task_entries)}
}};

// ---------------------------------------------------------------------------
// Megakernel entry point
// ---------------------------------------------------------------------------

__mlu_entry__ void {kernel_name}(
    void **buffers,
    int num_tasks
) {{
    int task_id = taskId;
    int sm_id = coreId;

    if (task_id >= num_tasks) return;

    TaskDescriptor desc = task_table[task_id];

    switch (desc.kind) {{
{chr(10).join(dispatch_cases)}
        default: break;
    }}
}}
"""

    return BangcMegakernelEmitResult(
        kernel_name=kernel_name,
        bangc_source=source,
        manifest={
            "kernel_name": kernel_name,
            "total_tasks": total_tasks,
            "device_function_count": len(distinct_names),
            "task_table": task_table,
            "kind_to_name": {str(k): v for k, v in kind_to_name.items()},
            "user_buffer_count": user_buffer_count,
        },
        device_function_table=kind_to_name,
    )


# ---------------------------------------------------------------------------
# Helpers — simple non-scheduled megakernel for testing
# ---------------------------------------------------------------------------


def emit_simple_gemm_kernel(
    *,
    body: DeviceFunctionSource,
    m: int = 64,
    n: int = 64,
    k: int = 64,
) -> BangcMegakernelEmitResult:
    """Emit a minimal single-GEMM BangC kernel — useful for testing
    the JIT compile → dispatch round-trip without the full megakernel
    infrastructure.

    The emitted kernel calls ``body.name`` once per task, with each
    task handling one output row.
    """

    # Build a task table where each row is one task
    task_table = [
        {
            "task_id": i,
            "device_func": body.name,
            "grid_dim": (m, 1, 1),
        }
        for i in range(m)
    ]

    return emit_bangc_megakernel(
        device_function_sources={body.name: body},
        task_table=task_table,
        kernel_name="mlu_simple_gemm",
        user_buffer_count=3,
    )


def emit_simple_elementwise_kernel(
    *,
    body: DeviceFunctionSource,
    total_elems: int,
    tasks: int = 1,
) -> BangcMegakernelEmitResult:
    """Emit a minimal single-elementwise BangC kernel."""

    task_table = [
        {
            "task_id": 0,
            "device_func": body.name,
            "grid_dim": (tasks, 1, 1),
        }
    ]

    return emit_bangc_megakernel(
        device_function_sources={body.name: body},
        task_table=task_table,
        kernel_name="mlu_simple_elementwise",
        user_buffer_count=4,
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MegakernelEmitError(RuntimeError):
    """Raised when a BangC megakernel cannot be emitted."""
