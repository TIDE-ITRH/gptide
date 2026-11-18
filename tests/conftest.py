"""
Test configuration

The GPU tests run in one of two modes:

- A real GPU is found (e.g. the HPC): numba.cuda and CuPy run on the device.
- No GPU: the numba CUDA *simulator* is switched on (kernels run in plain Python on
  the CPU, slowly) and only the 'numpy' linear algebra backend is used.

Set GPTIDE_CUDASIM=1 to force the simulator, or =0 to force a real GPU.
"""
import os

import pytest


def _have_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


HAVE_GPU = _have_gpu()

_force = os.environ.get("GPTIDE_CUDASIM")
USE_SIM = (not HAVE_GPU) if _force is None else (_force == "1")
if USE_SIM:
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    if request.param == "cupy" and (USE_SIM or not HAVE_GPU):
        pytest.skip("needs a real GPU and CuPy")
    return request.param
