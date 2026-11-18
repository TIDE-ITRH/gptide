"""
Pairwise ("pair form") covariance kernels

Every kernel here has the signature ``kernel(x1, x2, params)``: it evaluates the
covariance between *one pair* of D-dimensional points (``x1`` and ``x2`` are 1D
arrays of length D, ``params`` is a tuple of floats) and returns a scalar.

This is different from the "base form" used in :mod:`gptide.cov`, which is
``kernel(x, xpr, params)`` with ``x`` of shape [N, D] and ``xpr`` of shape [D, M]
and which returns the whole [N, M] matrix using numpy broadcasting.

The pair form is what the GPU solver (:class:`gptide.gpcuda.GPtideCUDA`) needs: one
GPU thread evaluates one matrix element. It is also the form used in gpvecchia.
The same function can be used on the CPU via :func:`as_base_cov`, which wraps it
in a (numba compiled) loop so it can be passed as `cov_func` to the base-form
classes (:class:`gptide.GPtideScipy` etc.).

Rules for writing a pair kernel (so it compiles for both the CPU and the GPU):

- Write it as a plain Python function, i.e. no `@njit` decorator.
- It must be self contained: use `math` functions only and do not call other
  Python functions.
- ``params`` is passed as a tuple of floats (arrays are converted by the callers).
"""

import math

import numpy as np

####
# Pair form kernels

def matern12(x1, x2, params):
    """Matern 1/2 (exponential) covariance. params = (eta, l)"""
    eta, l = params
    d2 = 0.0
    for i in range(x1.shape[0]):
        d2 += (x1[i] - x2[i])**2
    r = math.sqrt(d2)
    return eta**2 * math.exp(-r/l)


def matern32(x1, x2, params):
    """Matern 3/2 covariance. params = (eta, l)"""
    eta, l = params
    d2 = 0.0
    for i in range(x1.shape[0]):
        d2 += (x1[i] - x2[i])**2
    r = math.sqrt(d2)
    fac = math.sqrt(3.0) * r / l
    return eta**2 * (1 + fac) * math.exp(-fac)


def matern52(x1, x2, params):
    """Matern 5/2 covariance. params = (eta, l)"""
    eta, l = params
    d2 = 0.0
    for i in range(x1.shape[0]):
        d2 += (x1[i] - x2[i])**2
    r = math.sqrt(d2)
    fac = math.sqrt(5.0) * r / l
    return eta**2 * (1 + fac + (5*r**2)/(3*l**2)) * math.exp(-fac)


def gaussian(x1, x2, params):
    """Gaussian/squared exponential/RBF covariance. params = (eta, l)"""
    eta, l = params
    d2 = 0.0
    for i in range(x1.shape[0]):
        d2 += (x1[i] - x2[i])**2
    return eta**2 * math.exp(-0.5 * d2 / l**2)


def cosine(x1, x2, params):
    """Cosine covariance. params = (eta, l), l is the period"""
    eta, l = params
    d2 = 0.0
    for i in range(x1.shape[0]):
        d2 += (x1[i] - x2[i])**2
    r = math.sqrt(d2)
    return eta**2 * math.cos(2*math.pi*r/l)


####
# Helpers

def as_params_tuple(params):
    """Convert (array-like) parameters to the tuple of floats the pair kernels expect"""
    return tuple(float(v) for v in np.ravel(params))


_adaptor_cache = {}

def as_base_cov(pair_fn, parallel=True):
    """
    Adapt a pair form kernel to the base form used by the CPU GP classes

    Parameters
    ----------
    pair_fn: callable
        Pair form kernel, ``pair_fn(x1, x2, params) -> float``
        (see the module docstring for the rules)
    parallel: bool, optional (default=True)
        Evaluate the rows in parallel (numba `prange`)

    Returns
    -------
    cov_func: callable
        Base form kernel, ``cov_func(x, xpr, params)`` with ``x`` [N, D] and
        ``xpr`` [D, M] (or [D] for a single point), returning an [N, M] matrix.
        This is the signature expected by :class:`gptide.GPtideScipy`, `mle`, etc.

    Notes
    -----
    Needs numba. The kernel is compiled on the first call.
    """
    key = (pair_fn, parallel)
    if key in _adaptor_cache:
        return _adaptor_cache[key]

    from numba import njit, prange

    pair = njit(pair_fn)

    @njit(parallel=parallel)
    def _fill(X1, X2, p):
        N = X1.shape[0]
        M = X2.shape[0]
        out = np.empty((N, M))
        for i in prange(N):
            for j in range(M):
                out[i, j] = pair(X1[i], X2[j], p)
        return out

    def cov_func(x, xpr, params, **kwargs):
        X1 = np.ascontiguousarray(x, dtype=np.float64)
        xpr = np.asarray(xpr, dtype=np.float64)
        if xpr.ndim == 1:
            X2 = np.ascontiguousarray(xpr[None, :])
        else:
            X2 = np.ascontiguousarray(xpr.T)
        return _fill(X1, X2, as_params_tuple(params))

    cov_func.__name__ = getattr(pair_fn, '__name__', 'pair_cov')
    cov_func.pair_fn = pair_fn

    _adaptor_cache[key] = cov_func
    return cov_func
