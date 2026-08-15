import numpy as np

from gptide import mle
from gptide.cov import matern32_1d


def test_mle_recovers_length_scale_on_synthetic_data():
    rng = np.random.default_rng(0)

    true_eta, true_l, true_noise = 1.5, 2.0, 0.05
    xd = np.linspace(0, 20, 60).reshape(-1, 1)

    from gptide import GPtideScipy

    gp_truth = GPtideScipy(xd, xd, true_noise, matern32_1d, (true_eta, true_l))
    yd = gp_truth.prior(samples=1, noise=true_noise)
    yd += rng.normal(scale=true_noise, size=yd.shape)

    res = mle(
        xd,
        yd,
        matern32_1d,
        covparams_ic=[1.0, 1.0],
        noise_ic=0.1,
        method="Nelder-Mead",
        options={"maxiter": 500},
    )

    assert res.success
    assert np.isfinite(res.fun)
