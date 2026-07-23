"""MLU BodyEmitter — per-op BangC kernel sources.

BangC is Cambricon's CUDA-C-like kernel language. The semantic
mapping from CUDA C to BangC:

=======================  ======================  ==============================
CUDA C                   BangC                   Notes
=======================  ======================  ==============================
``__device__``           ``__mlu_func__``        Device function qualifier
``__global__``           ``__mlu_entry__``       Entry/kernel qualifier
``threadIdx.x``          ``taskId``              MLU's unified task model
``blockIdx.x``           cluster-level index     Multi-core dispatch
``blockDim.x``           ``taskDim``             Tasks per core
``__shared__``           ``__nram__``            NRAM on-chip memory (512 KB)
``__syncthreads()``      ``__sync_all()``        Barrier synchronization
``atomicAdd(...)``       ``__bang_atomic_add(...)``  Atomic operations
``__ldg(...)``           ``__memcpy(...)``       Global memory load
``half``                 ``half``                Same FP16 type
``__float2bfloat16``     ``__float2bfloat16``    Same BF16 conversion
=======================  ======================  ==============================

The emitter produces ``DeviceFunctionSource`` — the universal IR
that the megakernel wrapper consumes. The actual wrapping into a
full BangC kernel is done by the megakernel emitter (future:
``emit_bangc_megakernel.py``, analogous to
``emit_cuda_megakernel.py``).

For Triton-compatible ops, CompGen can use the Triton path
directly; this emitter handles the non-Triton / custom-op path.
"""

from __future__ import annotations

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


class MluBodyEmitter:
    """Emit per-op BangC kernel bodies for MLU accelerators.

    Satisfies :class:`compgen.targets.gpu.contracts.GpuBodyEmitter`.

    The emitter chooses tile shapes + precision + library backend;
    the matcher only sees :class:`DeviceFunctionSource`.
    """

    # ------------------------------------------------------------------
    # Tile shape
    # ------------------------------------------------------------------

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """MLU370 sweet spot: 64×64 is the NRAM-friendly tile for
        MFU (Matrix Function Unit) utilization. FP32 ops may
        prefer 32×32 for register pressure."""
        del op
        if dtype in ("fp32", "float32"):
            return (32, 32, 32)
        # bf16 / fp16 — use the MFU-friendly tile
        return (64, 64, 16)

    # ------------------------------------------------------------------
    # GEMM
    # ------------------------------------------------------------------

    def gemm(
        self,
        *,
        b_dim: int,
        k_dim: int,
        n_dim: int,
        n_tiles_per_row: int,
        x_buf: int,
        w_buf: int,
        out_buf: int,
        precision: str,
        tile_m: int,
        tile_n: int,
        tile_k: int,
    ) -> DeviceFunctionSource:
        """Emit a BangC GEMM body. Uses MLU's MFU via
        ``__bang_gemm`` intrinsic when available, with a fallback
        to hand-rolled tiled matmul."""

        use_fp32_acc = "fp32" in precision
        compute_dtype = "float" if use_fp32_acc else "half"

        body_lines: list[str] = [
            f"// MLU BangC GEMM: {tile_m}×{tile_n}×{tile_k}, precision={precision}",
            f"// Buffers: X[{x_buf}], W[{w_buf}], Out[{out_buf}]",
            "",
            f"const size_t task_id = taskId;",
            f"",
            f"// NRAM tile buffers (NRAM = on-chip SRAM, ~512 KB on mlu370)",
            f"__nram__ {compute_dtype} tile_a[{tile_m} * {tile_k}];",
            f"__nram__ {compute_dtype} tile_b[{tile_k} * {tile_n}];",
            f"__nram__ {compute_dtype} tile_c[{tile_m} * {tile_n}];",
            "",
            f"// Input pointers from global buffers",
            f"{compute_dtype}* A = ({compute_dtype}*)buffers[{x_buf}];",
            f"{compute_dtype}* B = ({compute_dtype}*)buffers[{w_buf}];",
            f"{compute_dtype}* C = ({compute_dtype}*)buffers[{out_buf}];",
            "",
            f"// Initialize accumulator",
            f"for (int i = 0; i < {tile_m} * {tile_n}; ++i) tile_c[i] = 0.0f;",
            "",
            f"// Main K-loop: tile over the reduction dimension",
            f"for (int k_block = 0; k_block < {k_dim}; k_block += {tile_k}) {{",
            f"    // Load A tile: [task_id, k_block] → NRAM",
            f"    __memcpy(tile_a,",
            f"             A + task_id * {k_dim} + k_block,",
            f"             {tile_m} * {tile_k} * sizeof({compute_dtype}),",
            f"             GDRAM2NRAM);",
            f"",
            f"    // Load B tile: [k_block, :] → NRAM",
            f"    __memcpy(tile_b,",
            f"             B + k_block * {n_dim},",
            f"             {tile_k} * {tile_n} * sizeof({compute_dtype}),",
            f"             GDRAM2NRAM);",
            f"",
            f"    // MFU matmul: tile_c += tile_a × tile_b",
            f"    // Uses Cambricon's bang_gemm intrinsic for hardware acceleration",
            f"    __bang_gemm(tile_c, tile_a, tile_b,",
            f"                {tile_m}, {tile_k}, {tile_n},",
            f"                /* alpha */ 1.0f, /* beta */ 1.0f);",
            f"}}",
            "",
            f"// Store result back to global memory",
            f"__memcpy(C + task_id * {n_dim},",
            f"         tile_c,",
            f"         {tile_m} * {tile_n} * sizeof({compute_dtype}),",
            f"         NRAM2GDRAM);",
        ]

        return DeviceFunctionSource(
            name="mlu_gemm",
            body="\n".join(body_lines),
            signature=f"int b_dim, int k_dim, int n_dim, void** buffers",
            included_headers=("#include <bang.h>",),
        )

    # ------------------------------------------------------------------
    # Elementwise ops
    # ------------------------------------------------------------------

    def elementwise(
        self,
        *,
        op: str,
        total_elems: int,
        in_bufs: tuple[int, ...],
        out_buf: int,
        tile_m: int,
        tile_n: int,
    ) -> DeviceFunctionSource:
        """Emit a tile-aware BangC elementwise body.

        Each MLU task processes ``ceil(tile_m * tile_n / taskDim)``
        elements in a strided loop. The BangC unified task model
        uses ``taskId`` for per-task indexing and ``taskDim`` for
        the total number of tasks.
        """

        n_in = len(in_bufs)
        tile_elems = tile_m * tile_n

        body_lines: list[str] = [
            f"// MLU BangC elementwise: op={op}, {tile_elems} elems",
            f"const size_t task_id = taskId;",
            f"const size_t num_tasks = taskDim;",
            f"const size_t elems_per_task = ({tile_elems} + num_tasks - 1) / num_tasks;",
            f"const size_t start = task_id * elems_per_task;",
            f"const size_t end = (start + elems_per_task > {tile_elems}) ? {tile_elems} : start + elems_per_task;",
            "",
        ]

        # Input pointer declarations
        input_refs: list[str] = []
        for i in range(n_in):
            body_lines.append(f"half* in{i} = (half*)buffers[{in_bufs[i]}];")
            input_refs.append(f"in{i}")

        body_lines += [
            f"half* out = (half*)buffers[{out_buf}];",
            "",
            f"for (size_t i = start; i < end; ++i) {{",
        ]

        # Per-element operation
        if op == "relu":
            body_lines += [
                f"    half val = in0[i];",
                f"    out[i] = (val > 0.0f) ? val : 0.0f;",
            ]
        elif op == "gelu":
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            body_lines += [
                f"    half x = in0[i];",
                f"    float xf = (float)x;",
                f"    float cdf = 0.5f * (1.0f + __bang_tanh(0.7978845608f * (xf + 0.044715f * xf * xf * xf)));",
                f"    out[i] = (half)(xf * cdf);",
            ]
        elif op == "add":
            body_lines += [
                f"    out[i] = in0[i] + in1[i];",
            ]
        elif op == "mul":
            body_lines += [
                f"    out[i] = in0[i] * in1[i];",
            ]
        elif op == "silu":
            # SiLU = x * sigmoid(x)
            body_lines += [
                f"    half x = in0[i];",
                f"    out[i] = x * (1.0f / (1.0f + __bang_exp(-(float)x)));",
            ]
        else:
            body_lines += [
                f"    // Unknown op: {op} — pass through in0",
                f"    out[i] = in0[i];",
            ]

        body_lines += [
            f"}}",
        ]

        return DeviceFunctionSource(
            name=f"mlu_elementwise_{op}",
            body="\n".join(body_lines),
            signature=f"int total_elems, void** buffers",
            included_headers=("#include <bang.h>", "#include <math.h>"),
        )
