"""Cambricon MLU vendor entry — GPU-class target.

Cambricon's MLU (Machine Learning Unit) accelerators are
GPU-class devices with:

- A PyTorch-compatible frontend (``torch.mlu`` replacing
  ``torch.cuda`` — same tensor API, different device string).
- Triton-language compatibility via Cambricon's Triton port.
- A CUDA-C-like kernel language called **BangC** for device-side
  programming, compiled via ``cncc`` (Cambricon Neuware Compiler).
- Runtime API via **CNRT** (Cambricon Neuware Runtime), analogous
  to the CUDA Runtime API.

Vendor-common code lives here; arch-specific specializations
(mlu370, mlu590, ...) land in sub-packages.

Per the unified target hierarchy: see
``docs/architecture/target-hierarchy.md``.
"""

from __future__ import annotations

from compgen.targets.registry import register_target


def _register_vendor_common() -> None:
    """Register the vendor-common ``gpu.cambricon`` entry.

    Arch-leaf packages (``mlu``, etc.) override this with
    concrete adapters. The vendor-common entry provides
    fallback metadata for any unregistered arch under this vendor.
    """
    register_target(
        target_class="gpu",
        vendor="cambricon",
        arch="",  # vendor-common
        rationale=(
            "Cambricon MLU vendor-common entry. Holds CNRT driver "
            "wrapper + BangC compiler primitives shared across all "
            "MLU arches (mlu370, mlu590, ...). Arch-specific "
            "specializations land in ``gpu.cambricon.mlu``, etc. "
            "Per the unified target hierarchy: see "
            "docs/architecture/target-hierarchy.md."
        ),
        registration_path="in_tree",
        metadata={
            "vendor_url": "https://www.cambricon.com/",
            "jit_toolchain": "CNCC (BangC)",
            "runtime_library": "CNRT",
            "memory_model": "unified-device-memory",
            "triton_compatible": True,
        },
    )


_register_vendor_common()
