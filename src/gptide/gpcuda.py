"""
GPU Gaussian Process regression (numba.cuda + CuPy)

The covariance matrix is assembled on the GPU by a numba.cuda kernel (one thread per
matrix element) that evaluates a *pair form* covariance function
(see :mod:`gptide.cov_pair`). The Cholesky factorisation and triangular solves are
done by cuSOLVER/cuBLAS through CuPy. The two libraries share the same device memory,
so there are no copies between them.

Importing this module does not need numba.cuda or CuPy, they are only imported
when a `GPtideCUDA` object is built.
"""

import numpy as np
from scipy import linalg as la

from .gp import GPtide

####
# Kernel launch

_BLOCK = (32, 8)  # threads (along columns, along rows). Columns are contiguous in memory.

_kernel_cache = {}


def _launch_config(nrows, ncols):
    bx, by = _BLOCK
    return ((ncols + bx - 1) // bx, (nrows + by - 1) // by), _BLOCK


def _get_kernels(pair_fn):
    """
    Compile (once per pair function) the CUDA kernels that evaluate `pair_fn`

    Returns
    -------
    fill: out[i,j] = pair_fn(X1[i], X2[j], p), plus `diag_add` where i == j
    diag: out[i] = pair_fn(X[i], X[i], p)
    """
    if pair_fn in _kernel_cache:
        return _kernel_cache[pair_fn]

    # `cuda` must be a module level global (not a local) so the numba CUDA simulator
    # can substitute its own `cuda` when it runs the kernels
    global cuda
    from numba import cuda

    dev = cuda.jit(device=True)(pair_fn)

    @cuda.jit
    def fill(out, X1, X2, p, diag_add):
        j, i = cuda.grid(2)  # x (fastest) runs along the columns -> coalesced writes
        if i < out.shape[0] and j < out.shape[1]:
            v = dev(X1[i], X2[j], p)
            if i == j:
                v += diag_add
            out[i, j] = v

    @cuda.jit
    def diag(out, X, p):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = dev(X[i], X[i], p)

    _kernel_cache[pair_fn] = (fill, diag)
    return fill, diag


####
# Linear algebra backends

class _NumpyBackend:
    """
    Host (scipy) linear algebra. Same code path as CuPy, used for testing the class
    on machines without a GPU (with NUMBA_ENABLE_CUDASIM=1) and as a reference.
    """
    name = 'numpy'
    xp = np

    def asarray(self, a):
        return np.ascontiguousarray(a, dtype=np.float64)

    def asnumpy(self, a):
        return np.asarray(a)

    def cholesky(self, K):
        return la.cholesky(K, lower=True)  # raises numpy.linalg.LinAlgError

    def solve_tri(self, L, b, trans=False):
        return la.solve_triangular(L, b, lower=True, trans='T' if trans else 'N')


class _CupyBackend:
    """cuSOLVER / cuBLAS through CuPy."""
    name = 'cupy'

    def __init__(self):
        import cupy as cp
        import cupyx
        from cupyx.scipy import linalg as cla
        self.xp = cp
        self._cupyx = cupyx
        self._cla = cla

    def asarray(self, a):
        return self.xp.ascontiguousarray(self.xp.asarray(a, dtype=self.xp.float64))

    def asnumpy(self, a):
        return self.xp.asnumpy(a)

    def cholesky(self, K):
        # CuPy ignores a failed factorisation unless asked to raise
        with self._cupyx.errstate(linalg='raise'):
            return self.xp.linalg.cholesky(K)

    def solve_tri(self, L, b, trans=False):
        return self._cla.solve_triangular(L, b, trans='T' if trans else 'N', lower=True)


def _get_backend(backend):
    if isinstance(backend, str):
        if backend == 'cupy':
            return _CupyBackend()
        elif backend == 'numpy':
            return _NumpyBackend()
        raise ValueError("backend must be 'cupy' or 'numpy'")
    return backend


####
# The GP class

class GPtideCUDA(GPtide):
    """
    Gaussian Process regression class

    Uses numba.cuda to build the covariance matrices on the GPU and CuPy
    (cuSOLVER/cuBLAS) to factorise and solve. Everything is done in float64.

    Note that `cov_func` is a *pair form* kernel, not the base form kernel of the other
    gptide classes, see :mod:`gptide.cov_pair`.

    Parameters
    ----------
    xd: numpy.ndarray [N, D]
        Input data locations
    xm: numpy.ndarray [M, D]
        Output/target point locations
    sd: float
        Data noise parameter
    cov_func: callable function
        Pair form kernel ``cov_func(x1, x2, params) -> float``, e.g. `gptide.cov_pair.matern32`
    cov_params: tuple
        Parameters passed to `cov_func`

    Other Parameters
    ----------------
    mean_func: callable function
        Returns the mean function (evaluated on the host with numpy)
    mean_params: tuple
        parameters passed to the mean function
    mean_kwargs: dict
        kwargs passed to the mean function
    backend: str ['cupy']
        'cupy' for the GPU linear algebra, or 'numpy' to do it on the host with scipy.
        (Kernels still need numba.cuda, or the simulator `NUMBA_ENABLE_CUDASIM=1`.)
    chunk_size: int [4096]
        Number of output points handled at once when predicting (limits memory use)

    Notes
    -----
    Only scalar (P=1) outputs are supported so far.

    Memory: a dense [N, N] float64 matrix is held on the device (plus a temporary copy
    while it is factorised), i.e. ~7 GB each for N=30000.
    """
    backend = 'cupy'
    chunk_size = 4096

    _be = None
    _xd_dev = None
    _xm_dev = None

    def __init__(self, xd, xm, sd, cov_func, cov_params, **kwargs):
        GPtide.__init__(self, xd, xm, sd, cov_func, cov_params, **kwargs)

    # ------------------------------------------------------------------
    # Building the factorisation

    def _setup(self):
        """Create the backend and upload the locations (first call only)"""
        if self._be is None:
            assert self.P == 1, 'Only univariate outputs (P=1) are supported by GPtideCUDA so far.'
            assert not self.cov_kwargs, 'cov_kwargs are not supported by GPtideCUDA, put them in cov_params.'
            self._be = _get_backend(self.backend)
            self._fill, self._diag = _get_kernels(self.cov_func)
        if self._xd_dev is None:
            self._xd_dev = self._be.asarray(self.xd)
        if self._xm_dev is None:
            self._xm_dev = self._be.asarray(self.xm)

    def _p(self):
        from .cov_pair import as_params_tuple
        return as_params_tuple(self.cov_params)

    def _cov(self, X1, X2, diag_add=0.):
        """[len(X1), len(X2)] covariance matrix evaluated on the device"""
        # numba launches on the default stream, as does CuPy, so the fill is ordered before
        # the factorisation that uses it without an explicit synchronise
        out = self._be.xp.empty((X1.shape[0], X2.shape[0]), dtype=np.float64)
        grid, block = _launch_config(*out.shape)
        self._fill[grid, block](out, X1, X2, self._p(), float(diag_add))
        return out

    def _build(self):
        """Fill Kdd, factorise it and evaluate the mean function"""
        self._setup()
        be, xp = self._be, self._be.xp

        K = self._cov(self._xd_dev, self._xd_dev, self.sd**2 + 1e-7)
        self.L = be.cholesky(K)
        del K
        self.w_md = None
        self.Kdd = None  # not stored, K is rebuilt when needed
        self.Kmd = None  # computed in chunks when predicting

        d = xp.diagonal(self.L)
        self._logdet = 2*float(xp.sum(xp.log(d)))
        if not np.isfinite(self._logdet):
            raise np.linalg.LinAlgError('Covariance matrix is not positive definite')

        self._update_mean()

    def update_xm(self, xm):
        """Update the output locations"""
        self.M = xm.shape[0]*self.P
        self.xm = xm
        self._xm_dev = None
        self._setup()
        self._update_mean()

    # ------------------------------------------------------------------
    # Solves

    def _to_dev(self, yd):
        r = np.asarray(yd - self.mu_d, dtype=np.float64)
        return self._be.asarray(r.reshape(self.N, -1))

    def _alpha(self, r):
        """K^-1 r, with K = Kdd + noise, given the factorisation"""
        v = self._be.solve_tri(self.L, r)
        return self._be.solve_tri(self.L, v, trans=True)

    def _chunks(self, n):
        for a in range(0, n, self.chunk_size):
            yield a, min(a + self.chunk_size, n)

    # ------------------------------------------------------------------
    # Public methods

    def __call__(self, yd):
        """
        Predict the GP posterior mean given data

        Parameters
        ----------

        yd: numpy.ndarray [N,1]
            Observed data

        Returns
        --------

        numpy.ndarray [M,1]
            Prediction

        """
        assert yd.shape[0] == self.N, ' first dimension in input data must equal '

        be = self._be
        alpha = self._alpha(self._to_dev(yd))

        out = np.empty((self.M, alpha.shape[1]))
        for a, b in self._chunks(self.M):
            Kmd = self._cov(self._xm_dev[a:b], self._xd_dev)
            out[a:b] = be.asnumpy(Kmd @ alpha)

        return self.mu_m + out

    def log_marg_likelihood(self, yd):
        """Compute the log of the marginal likelihood"""
        v = self._be.solve_tri(self.L, self._to_dev(yd))
        qdist = float(self._be.xp.sum(v[:, 0]**2))  # r^T K^-1 r = |L^-1 r|^2

        fac = self.N * np.log(2*np.pi)

        return -0.5*(self._logdet + qdist + fac)

    def prior(self, samples=1, noise=0.):
        """
        Sample from the prior distribution

        Parameters
        ----------

        samples: int, optional (default=1)
            number of samples
        noise: float, optional (default=0)
            standard deviation of the independent noise added to the samples

        Returns
        -------

        prior_sample: numpy.ndarray [N,samples]
            array of output samples

        """
        z = self._be.asarray(np.random.normal(size=(self.N, samples)))
        y = self._be.asnumpy(self.L @ z)

        return self.mu_d + y + noise*np.random.normal(size=(self.N, samples))

    def conditional(self, yd, samples=1):
        """
        Sample from the conidtional distribution

        Parameters
        ----------

        yd: numpy.ndarray [N,1]
            Observed data
        samples: int, optional (default=1)
            number of samples

        Returns
        -------

        conditional_sample: numpy.ndarray [M,samples]
            output array

        """
        be, xp = self._be, self._be.xp

        ymu = self.__call__(yd)

        S = self._calc_err(diag=False)
        S[xp.diag_indices(self.M)] += 1e-7 + self.sd**2
        Ls = be.cholesky(S)
        del S

        z = be.asarray(np.random.normal(size=(self.M, samples)))

        return ymu + be.asnumpy(Ls @ z)

    def _calc_err(self, diag=True):
        """
        Compute the covariance of the conditional distribution (on the device)

        Used by .conditional. The diagonal is returned as a numpy array, the full matrix
        stays on the device. The diagonal is computed in chunks of output points
        and never needs an [M, M] matrix.
        """
        be, xp = self._be, self._be.xp

        if diag:
            out = xp.empty(self.M)
            for a, b in self._chunks(self.M):
                Xm = self._xm_dev[a:b]
                kmm = xp.empty(b - a)
                self._diag[(b - a + 127)//128, 128](kmm, Xm, self._p())
                V = be.solve_tri(self.L, self._cov(self._xd_dev, Xm))
                out[a:b] = kmm - xp.sum(V*V, axis=0)
            return be.asnumpy(out)

        Kmm = self._cov(self._xm_dev, self._xm_dev)
        V = be.solve_tri(self.L, self._cov(self._xd_dev, self._xm_dev))
        return Kmm - V.T @ V

    # Not used: the covariance matrices are built on the device by _build
    def _calc_cov(self, cov_func, cov_params):
        raise NotImplementedError

    def _calc_weights(self, Kdd, sd, Kmd):
        raise NotImplementedError
