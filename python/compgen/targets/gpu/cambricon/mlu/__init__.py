"""Cambricon MLU arch-leaf (mlu370 / mlu590 series).

MLU is Cambricon's general-purpose ML accelerator. Key properties:

- **Torch layer**: ``torch.mlu`` mirrors ``torch.cuda`` — same API,
  different device (``"mlu"`` instead of ``"cuda"``). Transparent
  for CompGen's model capture path.
- **Triton layer**: Cambricon ships a Triton port that maps Triton
  IR to BangC. CompGen's Triton-based paths are mostly compatible.
- **Kernel layer**: BangC is a CUDA-C-like language. The body
  emitter emits BangC source instead of CUDA C; CNCC JIT-compiles
  it; CNRT dispatches it. The semantic mapping:
  - ``__device__`` → BangC ``__mlu_func__``
  - ``__global__`` → BangC ``__mlu_entry__``
  - ``threadIdx.x`` → ``taskId`` (BangC's unified task model)
  - ``blockIdx.x`` → BangC's cluster-level indexing
  - ``__shared__`` → ``__nram__`` (NRAM on-chip memory)
  - ``atomicAdd`` → ``__bang_atomic_add``

Supported MLU series:
- **mlu370** (S370 / MLU370-X4 / MLU370-M8): 48-96 compute cores,
  BF16 native, 256 TFLOPS BF16 peak.
- **mlu590**: Next-gen series (placeholder).

Architecture references:
- Cambricon Neuware Programming Guide
- BangC Language Specification
- CNRT API Reference
"""

from __future__ import annotations

from compgen.targets.gpu.cambricon.mlu.body_emitter import MluBodyEmitter
from compgen.targets.gpu.cambricon.mlu.cost import MluCostModel
from compgen.targets.gpu.cambricon.mlu.probe import MluProbe
from compgen.targets.gpu.cambricon.mlu.runtime import MluRuntime
from compgen.targets.registry import register_target


def _register_mlu() -> None:
    register_target(
        target_class="gpu",
        vendor="cambricon",
        arch="mlu",
        probe=MluProbe(),
        body_emitter=MluBodyEmitter(),
        runtime=MluRuntime(),
        cost_model=MluCostModel(),
        rationale=(
            "Cambricon MLU (mlu370 / mlu590 series). BangC kernel "
            "backend via CNCC JIT + CNRT dispatch. Torch-level "
            "drop-in for CUDA (torch.mlu). Triton-compatible "
            "via Cambricon's Triton port. "
            "Per the unified target hierarchy: see "
            "docs/architecture/target-hierarchy.md."
        ),
        registration_path="in_tree",
        metadata={
            "compute_capability_major": 3,
            "compute_capability_minor": 7,
            "supports_clusters": False,
            "supports_tensor_cores": True,
            "supports_cncc_jit": True,
            "default_tile_shape": [64, 64, 16],
            "preferred_precision": "bf16_fp32",
            "core_count": {"mlu370": 48, "mlu370_x4": 96, "mlu590": 128},
            "nram_size_kb": 512,
            "wram_size_kb": 1024,
        },
    )


_register_mlu()
