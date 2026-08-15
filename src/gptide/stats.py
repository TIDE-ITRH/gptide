"""
Stats functions

Thin wrappers around scipy.stats distributions, intended for use as priors
with gptide.mle and gptide.mcmc (both call `.logpdf()`; mcmc also calls
`.rvs()` to initialise walkers). Note that `mcmc.mh` additionally assumes
its priors are bounded (`.a`/`.b`/`.kwds["loc"]`/`.kwds["scale"]`), so only
truncnorm/uniform are suitable there.
"""

from scipy import stats

def truncnorm(mu, sigma, a = 0, b = 1e12):
    lower, upper = (a - mu) / sigma, (b - mu) / sigma
    return stats.truncnorm(
        lower, upper, loc=mu, scale=sigma)


def uniform(a, b):
    return stats.uniform(a, b - a)

def beta(a, b):
    return stats.beta(a, b)

def normal(mu, sigma):
    return stats.norm(loc=mu, scale=sigma)

def halfnorm(sigma):
    return stats.halfnorm(scale=sigma)

def halfcauchy(scale):
    return stats.halfcauchy(scale=scale)

def gamma(shape, scale):
    return stats.gamma(shape, scale=scale)

def invgamma(shape, scale):
    return stats.invgamma(shape, scale=scale)

def lognorm(mu, sigma):
    return stats.lognorm(s=sigma, scale=mu)