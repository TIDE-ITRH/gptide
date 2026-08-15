import numpy as np

from gptide.stats import (
    truncnorm,
    uniform,
    beta,
    normal,
    halfnorm,
    halfcauchy,
    gamma,
    invgamma,
    lognorm,
)


def test_truncnorm_samples_within_bounds():
    dist = truncnorm(mu=1.0, sigma=0.5, a=0, b=2)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples >= 0)
    assert np.all(samples <= 2)


def test_uniform_samples_within_bounds():
    dist = uniform(a=1.0, b=3.0)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples >= 1.0)
    assert np.all(samples <= 3.0)


def test_beta_samples_within_unit_interval():
    dist = beta(a=2.0, b=5.0)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_normal_samples_centered_on_mean():
    dist = normal(mu=3.0, sigma=1.0)
    samples = dist.rvs(size=5000, random_state=0)

    assert np.isclose(samples.mean(), 3.0, atol=0.1)


def test_halfnorm_samples_are_nonnegative():
    dist = halfnorm(sigma=1.5)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples >= 0)


def test_halfcauchy_samples_are_nonnegative():
    dist = halfcauchy(scale=1.0)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples >= 0)


def test_gamma_samples_are_positive():
    dist = gamma(shape=2.0, scale=1.0)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples > 0)


def test_invgamma_samples_are_positive():
    dist = invgamma(shape=3.0, scale=1.0)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples > 0)


def test_lognorm_samples_are_positive():
    dist = lognorm(mu=2.0, sigma=0.3)
    samples = dist.rvs(size=1000, random_state=0)

    assert np.all(samples > 0)


def test_priors_expose_logpdf_and_rvs_for_mcmc_and_mle():
    # gptide.mle/.mcmc only rely on .logpdf() and .rvs() (see mle._minfunc,
    # mcmc._minfunc_prior and mcmc.mcmc's p0 sampling), so any of these
    # wrappers must support that minimal interface.
    priors = [
        truncnorm(mu=1.0, sigma=0.5),
        uniform(a=0.0, b=1.0),
        beta(a=2.0, b=2.0),
        normal(mu=0.0, sigma=1.0),
        halfnorm(sigma=1.0),
        halfcauchy(scale=1.0),
        gamma(shape=2.0, scale=1.0),
        invgamma(shape=3.0, scale=1.0),
        lognorm(mu=1.0, sigma=0.5),
    ]

    for prior in priors:
        sample = prior.rvs(random_state=0)
        assert np.isfinite(prior.logpdf(sample))
