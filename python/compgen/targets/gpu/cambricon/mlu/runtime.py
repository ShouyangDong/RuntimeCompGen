"""MLU Runtime — CNCC JIT compile + CNRT dispatch.

The runtime provides JIT compilation of BangC source via CNCC
(Cambricon Neuware Compiler) and dispatch via CNRT (Cambricon
Neuware Runtime). This is the MLU equivalent of NVIDIA's NVRTC +
CUDA driver dispatch.

CNCC JIT flow::

    BangC source → cncc --bangc ... → .o / .so → dlopen →
    cnrtInvokeKernel dispatch

Key CNRT primitives:
- ``cnrtInit()`` — initialize the runtime (like ``cuInit``).
- ``cnrtCreateQueue()`` — create a compute queue.
- ``cnrtInvokeKernel()`` — launch a kernel (like ``cuLaunchKernel``).
- ``cnrtSyncQueue()`` — synchronize (like ``cuStreamSynchronize``).

The runtime satisfies :class:`compgen.targets.gpu.contracts.GpuRuntime`.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from compgen.targets.gpu.cambricon.mlu.probe import _load_cnrt, _resolve_cnrt_lib_path


# ---------------------------------------------------------------------------
# CNRT ctypes bindings (extended — kernel launch subset)
# ---------------------------------------------------------------------------

_CNRT_EXT_LIB: Any | None = None


def _load_cnrt_ext() -> Any | None:
    """Load CNRT with the full kernel-launch API bound."""
    global _CNRT_EXT_LIB
    if _CNRT_EXT_LIB is not None:
        return _CNRT_EXT_LIB

    lib_path = _resolve_cnrt_lib_path()
    if lib_path is None:
        return None

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return None

    # --- cnrtInit ---
    lib.cnrtInit.restype = ctypes.c_int
    lib.cnrtInit.argtypes = [ctypes.c_int]

    # --- cnrtCreateQueue ---
    lib.cnrtCreateQueue.restype = ctypes.c_int
    lib.cnrtCreateQueue.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    # --- cnrtDestroyQueue ---
    lib.cnrtDestroyQueue.restype = ctypes.c_int
    lib.cnrtDestroyQueue.argtypes = [ctypes.c_void_p]

    # --- cnrtSyncQueue ---
    lib.cnrtSyncQueue.restype = ctypes.c_int
    lib.cnrtSyncQueue.argtypes = [ctypes.c_void_p]

    # --- cnrtInvokeKernel ---
    # Note: actual signature varies by Neuware version; this
    # is the simplified call shape.
    lib.cnrtInvokeKernel.restype = ctypes.c_int
    lib.cnrtInvokeKernel.argtypes = [
        ctypes.c_void_p,  # kernel handle
        ctypes.c_uint,    # dimX
        ctypes.c_uint,    # dimY
        ctypes.c_uint,    # dimZ
        ctypes.c_void_p,  # queue
        ctypes.c_void_p,  # params
        ctypes.c_void_p,  # extra
    ]

    # --- cnrtLoadLibrary ---
    lib.cnrtLoadLibrary.restype = ctypes.c_int
    lib.cnrtLoadLibrary.argtypes = [ctypes.c_char_p]

    # --- cnrtGetSymbolAddress ---
    lib.cnrtGetSymbolAddress.restype = ctypes.c_int
    lib.cnrtGetSymbolAddress.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]

    _CNRT_EXT_LIB = lib
    return lib


def _resolve_cncc_path() -> str | None:
    """Locate the CNCC (BangC compiler) binary."""
    import shutil

    env = os.environ.get("CNCC_PATH")
    if env and Path(env).is_file():
        return env

    neuware_home = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
    cand = Path(neuware_home) / "bin" / "cncc"
    if cand.is_file():
        return str(cand)

    return shutil.which("cncc")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class MluRuntime:
    """JIT compile BangC source, load the module, dispatch on MLU.

    Satisfies :class:`compgen.targets.gpu.contracts.GpuRuntime`.
    """

    def __init__(self) -> None:
        self._queue: Any = None
        self._initialized: bool = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        lib = _load_cnrt_ext()
        if lib is None:
            raise RuntimeError("CNRT not available — cannot dispatch on MLU")
        ret = lib.cnrtInit(0)
        if ret != 0:
            raise RuntimeError(f"cnrtInit failed with code {ret}")

        queue = ctypes.c_void_p()
        ret = lib.cnrtCreateQueue(ctypes.byref(queue))
        if ret != 0:
            raise RuntimeError(f"cnrtCreateQueue failed with code {ret}")
        self._queue = queue
        self._initialized = True

    def compile_source(
        self,
        *,
        cuda_source: str,
        kernel_name: str = "",
        arch: str = "",
        extra_options: tuple[str, ...] = (),
        extra_include_paths: tuple[str, ...] = (),
    ) -> Any:
        """JIT compile BangC source via CNCC.

        Writes ``cuda_source`` to a temp file, invokes CNCC to
        produce a shared library, and returns a handle to the
        loaded module.

        CNCC invocation::

            cncc --bangc -O2 -fPIC -shared <source>.mlu -o <output>.so

        Returns a ctypes handle to the dlopen'd library.
        """
        cncc_path = _resolve_cncc_path()
        if cncc_path is None:
            raise RuntimeError(
                "CNCC compiler not found. Set $CNCC_PATH or $NEUWARE_HOME."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "kernel.mlu"
            so_path = Path(tmpdir) / "kernel.so"

            src_path.write_text(cuda_source)

            cmd = [
                cncc_path,
                "--bangc",
                "-O2",
                "-fPIC",
                "-shared",
                str(src_path),
                "-o",
                str(so_path),
            ]
            for opt in extra_options:
                cmd.append(opt)
            for inc in extra_include_paths:
                cmd.extend(["-I", inc])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"CNCC compile failed (exit {result.returncode}):\n"
                    f"STDERR:\n{result.stderr}\n"
                    f"STDOUT:\n{result.stdout}"
                )

            # Load the compiled .so via CNRT
            lib = _load_cnrt_ext()
            if lib is None:
                raise RuntimeError("CNRT not available")

            so_bytes = str(so_path).encode("utf-8")
            ret = lib.cnrtLoadLibrary(so_bytes)
            if ret != 0:
                raise RuntimeError(f"cnrtLoadLibrary failed with code {ret}")

            # Return the .so path — caller uses it with get_symbol
            return str(so_path)

    def launch(
        self,
        *,
        module_handle: Any,
        grid_dim: tuple[int, int, int] = (1, 1, 1),
        block_dim: tuple[int, int, int] = (1, 1, 1),
        cluster_dim: tuple[int, int, int] | None = None,
        shared_mem_bytes: int = 0,
        kernel_params: Any = None,
        cooperative: bool = False,
    ) -> None:
        """Launch a compiled BangC kernel on MLU.

        MLU's launch model differs from CUDA's:
        - ``grid_dim`` maps to the number of tasks across cores.
        - ``block_dim`` maps to tasks per core.
        - ``cluster_dim`` is ignored (MLU370 doesn't support cluster launch).

        The kernel function is resolved via ``cnrtGetSymbolAddress``
        from the loaded module.
        """
        self._ensure_init()
        lib = _load_cnrt_ext()
        if lib is None:
            raise RuntimeError("CNRT not available")

        # Resolve the kernel symbol
        kernel_ptr = ctypes.c_void_p()
        so_bytes = module_handle.encode("utf-8") if isinstance(module_handle, str) else module_handle
        ret = lib.cnrtGetSymbolAddress(ctypes.byref(kernel_ptr), so_bytes)
        if ret != 0:
            raise RuntimeError(f"cnrtGetSymbolAddress failed with code {ret}")

        # MLU uses a 3D grid for multi-core dispatch
        dim_x = ctypes.c_uint(grid_dim[0])
        dim_y = ctypes.c_uint(grid_dim[1])
        dim_z = ctypes.c_uint(grid_dim[2])

        ret = lib.cnrtInvokeKernel(
            kernel_ptr,
            dim_x, dim_y, dim_z,
            self._queue,
            kernel_params,
            None,
        )
        if ret != 0:
            raise RuntimeError(f"cnrtInvokeKernel failed with code {ret}")

    def dispatch(
        self,
        *,
        library_handle: Any,
        kernel_params: Any,
    ) -> None:
        """CPU-style dispatch — delegates to launch for MLU."""
        self.launch(
            module_handle=library_handle,
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
            kernel_params=kernel_params,
        )

    def synchronize(self) -> None:
        """Block until all queued MLU work completes."""
        if not self._initialized or self._queue is None:
            return

        lib = _load_cnrt_ext()
        if lib is None:
            return

        lib.cnrtSyncQueue(self._queue)
