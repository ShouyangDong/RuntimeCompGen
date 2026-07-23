"""MLU Probe — hardware detection via CNRT (Cambricon Neuware Runtime).

Detects MLU device availability, arch, and capabilities at
compile time. The autotune layer calls into this once per process.

CNRT is the equivalent of CUDA's driver API:
- ``cnrtInit()`` / ``cnrtGetDeviceCount()`` → like ``cuInit`` / ``cuDeviceGetCount``
- ``cnrtGetDeviceInfo()`` → like ``cuDeviceGetAttribute``
- ``cnrtGetLibVersion()`` → runtime version

When CNRT is not installed (CPU-only hosts), ``is_available()``
returns False without raising — same contract as NVIDIA's probe.
"""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CNRT ctypes bindings (minimal subset for probing)
# ---------------------------------------------------------------------------

_CNRT_LIB: Any | None = None


def _resolve_cnrt_lib_path() -> str | None:
    """Locate ``libcnrt.so`` (Linux) or the CNRT dynamic library.

    Search order:
    1. ``$CNRT_LIB_PATH`` env var (absolute path to libcnrt.so).
    2. ``$NEUWARE_HOME/lib64/libcnrt.so`` — standard Neuware install.
    3. System paths: ``/usr/local/neuware/lib64/``, ``/usr/lib/``.
    4. ``shutil.which("cnrt")``-adjacent paths.
    """
    env = os.environ.get("CNRT_LIB_PATH")
    if env and Path(env).is_file():
        return env

    neuware_home = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
    candidates = [
        Path(neuware_home) / "lib64" / "libcnrt.so",
        Path(neuware_home) / "lib" / "libcnrt.so",
        Path("/usr/local/neuware/lib64/libcnrt.so"),
        Path("/usr/lib/libcnrt.so"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)

    which_cnrt = shutil.which("cnrt")
    if which_cnrt:
        lib_dir = Path(which_cnrt).parent.parent / "lib64" / "libcnrt.so"
        if lib_dir.is_file():
            return str(lib_dir)

    return None


def _load_cnrt() -> Any | None:
    """Load + cache the CNRT ctypes wrapper. Returns None if CNRT
    isn't reachable."""
    global _CNRT_LIB
    if _CNRT_LIB is not None:
        return _CNRT_LIB

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

    # --- cnrtGetDeviceCount ---
    lib.cnrtGetDeviceCount.restype = ctypes.c_int
    lib.cnrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint)]

    # --- cnrtGetDeviceInfo (minimal) ---
    # We mostly need device count + lib version for probing.
    lib.cnrtGetLibVersion.restype = ctypes.c_int
    lib.cnrtGetLibVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

    _CNRT_LIB = lib
    return lib


def _cnrt_probe_success() -> bool:
    """Initialize CNRT and check for >=1 devices."""
    lib = _load_cnrt()
    if lib is None:
        return False
    try:
        ret = lib.cnrtInit(0)
        if ret != 0:
            return False
        count = ctypes.c_uint(0)
        ret = lib.cnrtGetDeviceCount(ctypes.byref(count))
        return ret == 0 and count.value > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class MluProbe:
    """Compile-time hardware probe for Cambricon MLU devices.

    Satisfies :class:`compgen.targets.gpu.contracts.GpuProbe`.
    """

    def is_available(self) -> bool:
        """Cheap probe — is CNRT reachable + >=1 MLU device?"""
        return _cnrt_probe_success()

    def device_arch(self) -> str:
        """Return the MLU arch tag. Currently returns ``"mlu370"``
        as the default; future probes may detect mlu590 etc."""
        # In production this would query cnrtDeviceGetAttribute for
        # the actual arch. For now we return the primary supported
        # series.
        if not self.is_available():
            return "mlu370"
        return "mlu370"

    def supports_clusters(self) -> bool:
        """MLU370 doesn't expose a CUDA-style cluster-launch
        primitive. Future MLU series may."""
        return False

    def supports_tensor_cores(self) -> bool:
        """MLU has native matrix-multiply units (MFU — Matrix
        Function Unit), analogous to NVIDIA tensor cores."""
        return True

    def library_paths(self) -> dict[str, str | None]:
        """CNCC include + library paths the JIT compiler needs.

        Keys: ``cncc_bin``, ``cnrt_include``, ``cnrt_lib``,
        ``bangc_include`` (BangC builtins).
        """
        neuware_home = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
        base = Path(neuware_home)
        return {
            "cncc_bin": str(base / "bin" / "cncc") if (base / "bin" / "cncc").is_file() else shutil.which("cncc"),
            "cnrt_include": str(base / "include") if (base / "include").is_dir() else None,
            "cnrt_lib": str(base / "lib64") if (base / "lib64").is_dir() else None,
            "bangc_include": str(base / "bangc" / "include") if (base / "bangc" / "include").is_dir() else None,
        }

    def vendor_extras(self) -> dict[str, Any]:
        """Surfaces MLU-specific metadata for audit queries."""
        return {
            "neuware_home": os.environ.get("NEUWARE_HOME", "/usr/local/neuware"),
            "cnrt_lib": _resolve_cnrt_lib_path(),
            "triton_compatible": True,
            "kernel_language": "BangC",
            "jit_compiler": "CNCC",
        }
