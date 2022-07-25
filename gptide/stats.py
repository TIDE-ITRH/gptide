"""
Stats functions
"""

from scipy import stats

def truncnorm(mu, sigma, a = 0, b = 1e12): 
    lower, upper = (a - mu) / sigma, (b - mu) / sigma
    return stats.truncnorm(
        lower, upper, loc=mu, scale=sigma)
