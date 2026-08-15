import numpy as np

from gptide import GPtideScipy
from gptide.cov import matern32_1d


def _make_gp(xd, xm, sd=1e-4, cov_params=(1.0, 2.0)):
    return GPtideScipy(xd, xm, sd, matern32_1d, cov_params)


def test_prediction_recovers_data_at_low_noise():
    xd = np.linspace(0, 10, 15).reshape(-1, 1)
    yd = np.sin(xd)

    gp = _make_gp(xd, xd, sd=1e-4)
    ypred = gp(yd)

    np.testing.assert_allclose(ypred, yd, atol=1e-2)


def test_prediction_at_new_points_has_expected_shape():
    xd = np.linspace(0, 10, 10).reshape(-1, 1)
    yd = np.sin(xd)
    xm = np.linspace(0, 10, 25).reshape(-1, 1)

    gp = _make_gp(xd, xm, sd=0.1)
    ypred = gp(yd)

    assert ypred.shape == (25, 1)


def test_log_marg_likelihood_is_finite_scalar():
    xd = np.linspace(0, 10, 10).reshape(-1, 1)
    yd = np.sin(xd)

    gp = _make_gp(xd, xd, sd=0.1)
    ll = gp.log_marg_likelihood(yd)

    assert np.isscalar(ll) or ll.shape == ()
    assert np.isfinite(ll)


def test_prior_and_conditional_sample_shapes():
    xd = np.linspace(0, 10, 8).reshape(-1, 1)
    yd = np.sin(xd)

    gp = _make_gp(xd, xd, sd=0.1)

    prior_samples = gp.prior(samples=5)
    assert prior_samples.shape == (8, 5)

    cond_samples = gp.conditional(yd, samples=5)
    assert cond_samples.shape == (8, 5)


def test_update_xm_changes_output_shape():
    xd = np.linspace(0, 10, 10).reshape(-1, 1)
    yd = np.sin(xd)
    xm_initial = xd
    xm_new = np.linspace(0, 10, 3).reshape(-1, 1)

    gp = _make_gp(xd, xm_initial, sd=0.1)
    gp.update_xm(xm_new)
    ypred = gp(yd)

    assert ypred.shape == (3, 1)


def test_update_xm_uses_cov_kwargs_not_mean_kwargs():
    # A cov_kwargs-only kernel arg should still apply after update_xm; if update_xm
    # used mean_kwargs instead of cov_kwargs this would raise a TypeError.
    def scaled_matern32_1d(x, xpr, params, scale=1.0):
        return scale * matern32_1d(x, xpr, params)

    xd = np.linspace(0, 10, 10).reshape(-1, 1)
    xm_new = np.linspace(0, 10, 3).reshape(-1, 1)

    gp = GPtideScipy(
        xd, xd, 0.1, scaled_matern32_1d, (1.0, 2.0), cov_kwargs={"scale": 2.0}
    )
    gp.update_xm(xm_new)

    expected = scaled_matern32_1d(xm_new, xd.T, (1.0, 2.0), scale=2.0)
    np.testing.assert_allclose(gp.Kmd, expected)
