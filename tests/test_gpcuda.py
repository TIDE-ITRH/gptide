"""
Tests for the pair form kernels and GPtideCUDA

Without a GPU these run on the numba CUDA simulator (see conftest.py). On a machine
with a GPU they run on the device, including the CuPy backend.
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

pytest.importorskip("numba")

from gptide import GPtideScipy, GPtideCUDA, mle
from gptide import cov, cov_pair
from gptide.cov_pair import as_base_cov

PAIR_KERNELS = [cov_pair.matern12, cov_pair.matern32, cov_pair.matern52,
                cov_pair.gaussian, cov_pair.cosine]


def _data(N=60, M=25, D=2, seed=0):
    rng = np.random.default_rng(seed)
    xd = rng.uniform(0, 10, size=(N, D))
    xm = rng.uniform(0, 10, size=(M, D))
    yd = np.sin(xd[:, :1]) + 0.1*rng.normal(size=(N, 1))
    return xd, xm, yd


# ---------------------------------------------------------------------
# Pair kernels and the CPU adaptor

@pytest.mark.parametrize("pair_fn", PAIR_KERNELS)
def test_adaptor_matches_distance_formula(pair_fn):
    xd, xm, _ = _data(D=3)
    eta, l = 1.7, 2.3
    r = cdist(xd, xm)
    expected = {
        'matern12': np.exp(-r/l),
        'matern32': (1 + np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l),
        'matern52': (1 + np.sqrt(5)*r/l + 5*r**2/(3*l**2))*np.exp(-np.sqrt(5)*r/l),
        'gaussian': np.exp(-0.5*r**2/l**2),
        'cosine': np.cos(2*np.pi*r/l),
    }[pair_fn.__name__] * eta**2

    got = as_base_cov(pair_fn)(xd, xm.T, np.array([eta, l]))   # also accepts array params
    assert got.shape == (len(xd), len(xm))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["matern12", "matern32", "matern52", "cosine"])
def test_adaptor_matches_existing_1d_kernels(name):
    x = np.linspace(0, 5, 30).reshape(-1, 1)
    params = (1.3, 0.9)
    old = getattr(cov, name + "_1d")(x, x.T, params)
    new = as_base_cov(getattr(cov_pair, name))(x, x.T, params)
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12)


def test_adaptor_single_point_for_toeplitz_style_call():
    x = np.linspace(0, 5, 30).reshape(-1, 1)
    f = as_base_cov(cov_pair.matern32)
    col = f(x, x[0, ...], (1.0, 1.0))
    np.testing.assert_allclose(col, f(x, x.T, (1.0, 1.0))[:, :1])


def test_adaptor_gives_base_classes_the_same_answer():
    xd, xm, yd = _data()
    base = GPtideScipy(xd, xm, 0.1, as_base_cov(cov_pair.matern32), (1.2, 2.0))
    # the same kernel written in base form
    def k(x, xpr, p):
        r = cdist(x, xpr.T)
        return p[0]**2*(1 + np.sqrt(3)*r/p[1])*np.exp(-np.sqrt(3)*r/p[1])
    ref = GPtideScipy(xd, xm, 0.1, k, (1.2, 2.0))
    np.testing.assert_allclose(base(yd), ref(yd), rtol=1e-10)


# ---------------------------------------------------------------------
# GPtideCUDA against GPtideScipy

def _pair(pair_fn=cov_pair.matern32, backend="numpy", params=(1.2, 2.0), sd=0.1, **kw):
    xd, xm, yd = _data()
    ref = GPtideScipy(xd, xm, sd, as_base_cov(pair_fn), params, **kw)
    gpu = GPtideCUDA(xd, xm, sd, pair_fn, params, backend=backend, **kw)
    return ref, gpu, yd


@pytest.mark.parametrize("pair_fn", [cov_pair.matern32, cov_pair.gaussian, cov_pair.matern12])
def test_factorisation_likelihood_and_mean_match_scipy(pair_fn, backend):
    ref, gpu, yd = _pair(pair_fn, backend)
    L = gpu._be.asnumpy(gpu.L)

    np.testing.assert_allclose(L, ref.L, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(L, np.tril(L))          # upper triangle is zero (matmuls rely on it)
    np.testing.assert_allclose(gpu.log_marg_likelihood(yd), ref.log_marg_likelihood(yd), rtol=1e-10)
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)


def test_in_sample_prediction(backend):
    xd, _, yd = _data()
    ref = GPtideScipy(xd, xd, 0.1, as_base_cov(cov_pair.matern32), (1.2, 2.0))
    gpu = GPtideCUDA(xd, xd, 0.1, cov_pair.matern32, (1.2, 2.0), backend=backend)
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)


def test_prediction_in_chunks(backend):
    ref, gpu, yd = _pair(backend=backend)
    gpu.chunk_size = 7   # does not divide M=25
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(gpu._calc_err(diag=True), ref._calc_err(diag=True), rtol=1e-8, atol=1e-9)


def test_posterior_covariance_matches_scipy(backend):
    ref, gpu, yd = _pair(backend=backend)
    V = gpu._be.asnumpy(gpu._calc_err(diag=False))
    np.testing.assert_allclose(V, ref._calc_err(diag=False), rtol=1e-8, atol=1e-9)


def test_update_params_matches_fresh_object(backend):
    ref, gpu, yd = _pair(backend=backend)
    gpu.update_params(cov_params=np.array([0.7, 1.3]), sd=0.25)   # arrays are fine
    xd, xm, _ = _data()
    fresh = GPtideScipy(xd, xm, 0.25, as_base_cov(cov_pair.matern32), (0.7, 1.3))

    np.testing.assert_allclose(gpu.log_marg_likelihood(yd), fresh.log_marg_likelihood(yd), rtol=1e-10)
    np.testing.assert_allclose(gpu(yd), fresh(yd), rtol=1e-8, atol=1e-9)


def _lin(x, params):
    return params[0] + params[1]*x[:, :1]


def test_mean_function_and_update_xm(backend):
    ref, gpu, yd = _pair(backend=backend, mean_func=_lin, mean_params=(1.0, 0.3))
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(gpu.log_marg_likelihood(yd), ref.log_marg_likelihood(yd), rtol=1e-10)

    gpu.update_params(mean_params=(2.0, -0.1))
    ref.update_params(mean_params=(2.0, -0.1))
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)

    xm_new = np.linspace(0, 10, 9).reshape(-1, 1) * np.ones((1, 2))
    gpu.update_xm(xm_new)
    ref.update_xm(xm_new)
    assert gpu(yd).shape == (9, 1)
    np.testing.assert_allclose(gpu(yd), ref(yd), rtol=1e-8, atol=1e-9)


def test_sample_shapes_and_prior_statistics(backend):
    ref, gpu, yd = _pair(backend=backend)
    assert gpu.prior(samples=4).shape == (60, 4)
    assert gpu.conditional(yd, samples=3).shape == (25, 3)

    np.random.seed(0)
    z = gpu.prior(samples=4000)
    # the sample covariance of the prior approaches Kdd + noise
    emp = np.cov(z[:5])
    exp = (ref.Kdd + (ref.sd**2 + 1e-7)*np.eye(60))[:5, :5]
    np.testing.assert_allclose(emp, exp, atol=0.15)


def _negative(x1, x2, params):
    return -params[0]


def test_not_positive_definite_raises(backend):
    xd = np.linspace(0, 1, 5).reshape(-1, 1)
    with pytest.raises(np.linalg.LinAlgError):
        GPtideCUDA(xd, xd, 0.1, _negative, (1.0,), backend=backend)


def test_mle_with_gptidecuda(backend):
    created = []

    class Counting(GPtideCUDA):
        def __init__(self, *a, **k):
            created.append(self)
            super().__init__(*a, **k)

    xd, _, yd = _data(N=40)
    res = mle(xd, yd, cov_pair.matern32, covparams_ic=[1.0, 2.0], noise_ic=0.2,
              GPclass=Counting, gp_kwargs=dict(backend=backend),
              method="Nelder-Mead", options={"maxiter": 25})

    assert len(created) == 1
    noise, eta, l = res.x
    ref = GPtideScipy(xd, xd, noise, as_base_cov(cov_pair.matern32), (eta, l))
    np.testing.assert_allclose(-ref.log_marg_likelihood(yd), res.fun, rtol=1e-8)
