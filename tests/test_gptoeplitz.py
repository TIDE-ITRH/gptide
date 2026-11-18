import numpy as np

from gptide import GPtideScipy, GPtideToeplitz
from gptide.cov import matern32_1d


def _pair(xm, sd=0.1, params=(1.3, 2.0)):
    xd = np.linspace(0, 10, 40).reshape(-1, 1)   # regular grid
    yd = np.sin(xd) + 0.1*np.cos(3*xd)
    return (GPtideScipy(xd, xm, sd, matern32_1d, params),
            GPtideToeplitz(xd, xm, sd, matern32_1d, params), yd)


def test_toeplitz_matches_scipy_out_of_sample():
    xm = np.linspace(-1, 11, 17).reshape(-1, 1)
    ref, toe, yd = _pair(xm)
    assert toe(yd).shape == (17, 1)
    np.testing.assert_allclose(toe(yd), ref(yd), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(toe.log_marg_likelihood(yd), ref.log_marg_likelihood(yd), rtol=1e-8)


def test_toeplitz_matches_scipy_in_sample():
    xd = np.linspace(0, 10, 40).reshape(-1, 1)
    ref, toe, yd = _pair(xd)
    assert toe(yd).shape == (40, 1)
    np.testing.assert_allclose(toe(yd), ref(yd), rtol=1e-6, atol=1e-8)


def test_toeplitz_posterior_variance_matches_scipy():
    xm = np.linspace(-1, 11, 17).reshape(-1, 1)
    ref, toe, yd = _pair(xm)
    np.testing.assert_allclose(toe._calc_err(diag=True), ref._calc_err(diag=True), rtol=1e-6, atol=1e-8)


def test_toeplitz_update_xm_matches_scipy():
    ref, toe, yd = _pair(np.linspace(0, 10, 5).reshape(-1, 1))
    xm_new = np.linspace(2, 8, 11).reshape(-1, 1)
    ref.update_xm(xm_new)
    toe.update_xm(xm_new)
    np.testing.assert_allclose(toe(yd), ref(yd), rtol=1e-6, atol=1e-8)
