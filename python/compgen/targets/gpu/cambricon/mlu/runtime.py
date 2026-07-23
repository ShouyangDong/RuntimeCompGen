"""MLU Runtime — CNCC JIT compile + CNRT dispatch.

The runtime provides JIT compilation of BangC source via CNCC
(Cambricon Neuware Compiler) and dispatch via CNRT (Cambricon
Neuware Runtime).

CNRT function names vary across Neuware versions (``cnInit`` vs
``cnrtInit``, ``cnCreateQueue`` vs ``cnrtCreateQueue``, etc.).
All bindings are resolved lazily via the shared helper in
:mod:`compgen.targets.gpu.cambricon.mlu.probe`.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from compgen.targets.gpu.cambricon.mlu.probe import (
    _bind_cnrt_func,
    _load_cnrt,
    _resolve_cnrt_lib_path,
)


# ---------------------------------------------------------------------------
# CNCC discovery
# ---------------------------------------------------------------------------


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
    """JIT compile BangC source, load module, dispatch on MLU.

    Satisfies :class:`compgen.targets.gpu.contracts.GpuRuntime`.

    Real CNRT flow::

        cnrtSetDevice(0) → cnrtQueueCreate → CNCC compile .so
        → ctypes.CDLL(.so) → call kernel function directly.

    There is NO ``cnrtInvokeKernel`` / ``cnrtLoadLibrary`` in CNRT.
    BangC ``<<<dim, func_type, queue>>>`` is lowered by CNCC into
    host-side code in the output .so.
    """

    def __init__(self) -> None:
        self._queue: Any = None
        self._cnrt_lib: Any = None
        self._initialized: bool = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        lib = _load_cnrt()
        if lib is None:
            raise RuntimeError("CNRT not available — cannot dispatch on MLU")

        # cnrtSetDevice(0)
        if not _bind_cnrt_func(lib, "cnrtSetDevice", ctypes.c_int, [ctypes.c_int]):
            raise RuntimeError("cnrtSetDevice not found in libcnrt.so")
        ret = lib.cnrtSetDevice(0)
        if ret != 0:
            raise RuntimeError(f"cnrtSetDevice(0) failed with code {ret}")

        # cnrtQueueCreate
        if not _bind_cnrt_func(lib, "cnrtQueueCreate", ctypes.c_int,
                               [ctypes.POINTER(ctypes.c_void_p)]):
            raise RuntimeError("cnrtQueueCreate not found in libcnrt.so")
        q = ctypes.c_void_p()
        ret = lib.cnrtQueueCreate(ctypes.byref(q))
        if ret != 0:
            raise RuntimeError(f"cnrtQueueCreate failed with code {ret}")

        self._queue = q
        self._cnrt_lib = lib
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
        """JIT compile BangC source via CNCC → ctypes.CDLL."""
        cncc_path = _resolve_cncc_path()
        if cncc_path is None:
            raise RuntimeError("CNCC compiler not found. Set $CNCC_PATH or $NEUWARE_HOME.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "kernel.mlu"
            so_path = Path(tmpdir) / "kernel.so"
            src_path.write_text(cuda_source)

            cmd = [cncc_path, "--bangc", "-O2", "-fPIC", "-shared",
                   str(src_path), "-o", str(so_path)]
            for opt in extra_options:
                cmd.append(opt)
            for inc in extra_include_paths:
                cmd.extend(["-I", inc])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"CNCC compile failed (exit {result.returncode}):\n"
                    f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
                )

            try:
                lib = ctypes.CDLL(str(so_path))
            except OSError as exc:
                raise RuntimeError(f"Failed to load compiled .so: {exc}") from exc

            return lib

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
        """Call the kernel function directly from the loaded .so.

        ``kernel_params`` is (kernel_name, *c_args).
        """
        self._ensure_init()
        del grid_dim, block_dim, cluster_dim, shared_mem_bytes, cooperative

        if not isinstance(kernel_params, tuple) or len(kernel_params) < 1:
            raise ValueError("kernel_params must be (kernel_name, *args)")

        kernel_name, *args = kernel_params
        kernel_func = getattr(module_handle, kernel_name, None)
        if kernel_func is None:
            raise RuntimeError(f"Kernel '{kernel_name}' not found in compiled module")
        kernel_func(*args)

    def dispatch(
        self,
        *,
        library_handle: Any,
        kernel_params: Any,
    ) -> None:
        """CPU-style dispatch — delegates to launch."""
        self.launch(module_handle=library_handle, kernel_params=kernel_params)

    def synchronize(self) -> None:
        """Block via ``cnrtQueueSync``."""
        if not self._initialized:
            return
        lib = self._cnrt_lib
        if lib is not None and _bind_cnrt_func(lib, "cnrtQueueSync", ctypes.c_int,
                                                [ctypes.c_void_p]):
            lib.cnrtQueueSync(self._queue)
