# GPtideCUDA development notes (TEMPORARY)

Working notes for the `cuda_cupy` branch. **Delete this file (and fold anything useful into
the docs) before merging.** Things marked `<...>` are placeholders I could not check from
the development machine (no GPU, no HPC access) - fill them in.

## What this is

`gptide.GPtideCUDA` (`src/gptide/gpcuda.py`): GP regression on an NVIDIA GPU.

- Covariance matrices are built by a **numba.cuda** kernel (one thread per matrix element).
- The Cholesky factorisation and triangular solves are done by **CuPy** (cuSOLVER/cuBLAS).
- Same API as `GPtideScipy` (`__call__`, `log_marg_likelihood`, `prior`, `conditional`,
  `update_xm`, `update_params`), so it works with `mle` / `mcmc`.
- Importing `gptide` does not import numba.cuda or CuPy; they are only needed when a
  `GPtideCUDA` is built.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Target hardware | NVIDIA A100 | what we have; numba.cuda is NVIDIA only (AMD GPUs would need PyTorch/JAX instead) |
| Libraries | numba.cuda for the kernel fill, CuPy for dense linear algebra | reuses the gpvecchia/browseGP numba kernels; Cholesky is O(N^3) and dominates, so cuSOLVER is needed, numba has none |
| Size | N ~ 15k points x 2 outputs, matrix dimension < ~30k | one 30k x 30k float64 matrix is 7.2 GB (plus a transient copy in the Cholesky), fits on an A100 |
| Precision | float64 everywhere | GP matrices are ill conditioned |
| Workloads | MLE, MCMC, in-sample prediction | large gridded predictions stay on gpvecchia (Vecchia) |
| Kernel form | **pair form** `f(x1, x2, params) -> float` | maps to one GPU thread per element; same as gpvecchia |
| Scope so far | scalar (P=1) kernels only | multi-output (ICM, Helmholtz) is a later phase |
| MCMC method | unchanged | redesign deferred; only the likelihood now reuses one GP object via `update_params` |

### Pair form vs base form kernels (to be written up for users)

- **Base form** (`gptide.cov`, used by `GPtideScipy`, `GPtideToeplitz`, `GPtideDask`):
  `f(x, xpr, params)`, `x` is [N, D], `xpr` is [D, M], returns the [N, M] matrix using
  numpy broadcasting.
- **Pair form** (`gptide.cov_pair`, used by `GPtideCUDA`): `f(x1, x2, params)`, `x1` and `x2`
  are single points (1D arrays of length D), returns a float. Rules: plain Python function
  (no `@njit`), `math` functions only, self contained (no calls to other Python functions),
  `params` is a tuple of floats.
- `gptide.cov_pair.as_base_cov(pair_fn)` wraps a pair kernel for the CPU classes
  (needs numba), so one kernel definition serves both.

### Other changes made on this branch

- `GPtide.update_params(cov_params, sd, mean_params)` added to the base class (all backends).
  `mle` and the serial `mcmc.mcmc` (emcee) path build one GP object and then call it.
  The parallel (`Pool`) emcee path and `mh` still rebuild the object each call.
- `GPtide.update_xm` now also refreshes the mean at the new output points.
- Fixed: `GPtideScipy.prior(noise=...)` reused the same random draw for the noise and for
  the correlated part (variance was K + 2*noise*L + noise^2 I). It is now independent.
  `GPtideCUDA.prior` matches.
- Fixed: `GPtideToeplitz` built `Kmd` with the wrong shape (predictions were wrong or
  failed). It is now the full [M, N] matrix.
- `tqdm` added to the dependencies (`mcmc.py` imports it); `[numba]` and `[cuda]` extras added.

## Creating the environment

Pinned versions are in `requirements-cuda.txt` (Python 3.12, numba 0.68, numba-cuda 0.30,
cupy-cuda12x 14.2, numpy 2.5, scipy 1.18). **Do not copy a venv between machines, recreate it.**

### Local machine (WSL, no GPU: runs the numba CUDA simulator)

```bash
/usr/bin/python3.12 -m venv ~/venvs/gptide-cuda      # keep it on the Linux filesystem, not /mnt/c
source ~/venvs/gptide-cuda/bin/activate
pip install --upgrade pip
pip install -r requirements-cuda.txt
pip install -e . --no-deps                           # run from the gptide repo root
```

`nvidia-cuda-runtime-cu12` is in the requirements because numba-cuda loads `libcudart` even
for the simulator; without a CUDA install the import fails.

### HPC (A100)

```bash
module load cuda/12.6.3          # or cuda/12.4.1. NOT the default (cuda/13.4.2): cupy-cuda12x needs CUDA 12
module load <python 3.10+ module>   # or a conda env; check `module avail python`
python -m venv $HOME/venvs/gptide-cuda      # <or a project/scratch path>
source $HOME/venvs/gptide-cuda/bin/activate
pip install --upgrade pip
pip install -r requirements-cuda.txt
pip install -e . --no-deps
```

Notes:

- `module avail cuda` on the cluster listed: 11.2.2, 12.4.1, 12.6.3, 12.9.2, 13.4.2 (default).
- numba-cuda compiles kernels with NVVM/libdevice from the CUDA toolkit, which the `cuda`
  module provides. If it cannot find it, check `CUDA_HOME` is set by the module.
- If `nvidia-cuda-runtime-cu12` clashes with the module's CUDA runtime, drop that line from
  the install (`grep -v nvidia-cuda-runtime requirements-cuda.txt`); the module supplies it.
- Do the install on a login node or an interactive job, whichever the cluster allows
  internet access from `<check>`.
- Remember to `module load cuda/12.x` in every job script, before activating the venv.

Check that the GPU is visible (inside a GPU job, not on the login node):

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.runtime.getDeviceProperties(0)['name'])"
python -c "from numba import cuda; cuda.detect()"
```

## Running the tests

```bash
source ~/venvs/gptide-cuda/bin/activate
cd <gptide repo root>
pytest -q tests
```

`tests/conftest.py` picks the mode automatically:

- **No GPU** (local machine): the numba CUDA **simulator** is switched on
  (`NUMBA_ENABLE_CUDASIM=1`), kernels run in plain Python on the CPU, only the `numpy`
  linear algebra backend is used and the `cupy` cases are skipped. Takes ~2 minutes.
- **GPU found** (HPC): real numba.cuda + CuPy, and the `cupy` cases run as well.
- Override with `GPTIDE_CUDASIM=1` (force simulator) or `GPTIDE_CUDASIM=0` (force a real GPU).

**The tests that matter on the A100** are `tests/test_gpcuda.py` with the `[cupy-...]` ids
(nothing skipped). Run it verbosely and check that no cupy test is `SKIPPED`:

```bash
pytest -v tests/test_gpcuda.py
```

Template SLURM script (partition, account and limits are placeholders):

```bash
#!/bin/bash
#SBATCH --job-name=gptide-cuda-test
#SBATCH --partition=<gpu partition>
#SBATCH --account=<account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load cuda/12.6.3
source $HOME/venvs/gptide-cuda/bin/activate
cd <path to gptide>
nvidia-smi
pytest -v tests 2>&1 | tee test_gpu.log
```

### Timing a likelihood evaluation (to decide if analytic gradients are worth it)

```python
import time, numpy as np
from gptide import GPtideCUDA, cov_pair

for N in (10000, 20000, 30000):
    x = np.random.uniform(0, 100, (N, 2)); y = np.random.randn(N, 1)
    gp = GPtideCUDA(x, x, 0.1, cov_pair.matern32, (1.0, 10.0))   # first call includes JIT compile
    t = time.time()
    for _ in range(3):
        gp.update_params((1.0, 10.0))
        gp.log_marg_likelihood(y)
    print(N, (time.time() - t) / 3, 's per likelihood')
```

## Status and TODO

Done and tested on the simulator (not yet on a real GPU):

- pair kernels (matern12/32/52, gaussian, cosine) and the `as_base_cov` adaptor
- `GPtideCUDA`: factorisation, likelihood, mean, posterior variance/covariance, prior and
  conditional samples, mean functions, `update_params`, `update_xm`, non-PD error
- `update_params` in the base class; Toeplitz and prior-noise fixes

Not yet done:

1. Run `tests/test_gpcuda.py` on the A100 (**first thing to do**) and send/inspect failures.
2. Time a likelihood at N = 10k/20k/30k (snippet above).
3. Analytic gradients for MLE (fused kernel for tr(K^-1 dK)), if the timing says finite
   differences are too slow.
4. Multi-output kernels (ICM from gpvecchia, Helmholtz from browseGP); kernels take
   augmented points with the output index in the last column.
5. MCMC redesign (batched/gradient based), `mh` and `Pool` paths still rebuild the GP.
6. In-place Cholesky to avoid the transient second N x N buffer, only if memory is a problem.
7. User-facing docs for pair vs base form kernels and the `[cuda]` extra; delete this note.
