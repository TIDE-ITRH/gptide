import numpy as np
import pytest

from gptide import GPtideScipy, GPtideToeplitz, mle
from gptide.cov import matern32_1d


def _mean(x, params):
    return params[0] + params[1] * x


@pytest.mark.parametrize("GPclass", [GPtideScipy, GPtideToeplitz])
def test_update_params_matches_fresh_object(GPclass):
    xd = np.linspace(0, 10, 40).reshape(-1, 1)
    xm = np.linspace(0, 10, 15).reshape(-1, 1)
    yd = np.sin(xd)

    gp = GPclass(xd, xm, 0.1, matern32_1d, (1.0, 2.0))
    gp.update_params(cov_params=(0.7, 1.3), sd=0.2)
    fresh = GPclass(xd, xm, 0.1 * 2, matern32_1d, (0.7, 1.3))

    assert gp.sd == 0.2
    assert gp.cov_params == (0.7, 1.3)
    np.testing.assert_allclose(gp.log_marg_likelihood(yd), fresh.log_marg_likelihood(yd))
    np.testing.assert_allclose(gp(yd), fresh(yd))


def test_update_params_only_sd_keeps_cov_params():
    xd = np.linspace(0, 10, 20).reshape(-1, 1)
    yd = np.cos(xd)
    gp = GPtideScipy(xd, xd, 0.1, matern32_1d, (1.0, 2.0))
    gp.update_params(sd=0.3)
    fresh = GPtideScipy(xd, xd, 0.3, matern32_1d, (1.0, 2.0))

    assert gp.cov_params == (1.0, 2.0)
    np.testing.assert_allclose(gp.log_marg_likelihood(yd), fresh.log_marg_likelihood(yd))


def test_update_params_updates_mean_function():
    xd = np.linspace(0, 10, 20).reshape(-1, 1)
    yd = 2 + 0.5 * xd + np.sin(xd)
    kw = dict(mean_func=_mean, mean_params=(0.0, 0.0))

    gp = GPtideScipy(xd, xd, 0.1, matern32_1d, (1.0, 2.0), **kw)
    gp.update_params(mean_params=(2.0, 0.5))
    fresh = GPtideScipy(xd, xd, 0.1, matern32_1d, (1.0, 2.0),
                        mean_func=_mean, mean_params=(2.0, 0.5))

    np.testing.assert_allclose(gp.mu_d, fresh.mu_d)
    np.testing.assert_allclose(gp.mu_m, fresh.mu_m)
    np.testing.assert_allclose(gp.log_marg_likelihood(yd), fresh.log_marg_likelihood(yd))


def test_mle_reuses_one_gp_object():
    created = []

    class CountingGP(GPtideScipy):
        def __init__(self, *a, **k):
            created.append(self)
            super().__init__(*a, **k)

    rng = np.random.default_rng(1)
    xd = np.linspace(0, 10, 30).reshape(-1, 1)
    yd = np.sin(xd) + 0.05 * rng.normal(size=xd.shape)

    res = mle(xd, yd, matern32_1d, covparams_ic=[1.0, 1.0], noise_ic=0.1,
              GPclass=CountingGP, method="Nelder-Mead", options={"maxiter": 60})

    assert len(created) == 1                 # built once, then update_params
    assert np.isfinite(res.fun)

    # same answer as a fresh object at the optimum
    noise, eta, l = res.x
    fresh = GPtideScipy(xd, xd, noise, matern32_1d, (eta, l))
    np.testing.assert_allclose(-fresh.log_marg_likelihood(yd), res.fun)
