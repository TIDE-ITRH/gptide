"""
MCMC parameter estimation using blackjax

Main function

    
def mcmc_jax(
    xd, yd, 
    covfunc, 
    meanfunc, 
    priors,
    ncovparams,
    initvals,
    mean_kwargs={},
    cov_kwargs={},
    nwarmup=500, 
    niter=500,
    oi_kwargs={}):
"""

from gptide import GPtideJax
import jax.scipy.stats as jstats
import jax.numpy as jnp
import jax

import blackjax

from jax.experimental import host_callback


###################
# Main inference routines
###################
def mcmcjax(
    xd, 
    yd, 
    covfunc, 
    cov_priors,
    noise_prior,
    meanfunc=None,             
    mean_priors=[],
    mean_kwargs={},
    cov_kwargs={},
    nwarmup=500, 
    niter=500,
    nprior=500,
    gp_kwargs={},
    step_size = 1e-3,
    inverse_mass_matrix=None,
    initvals = None
    ):
    
    """
    Main MCMC function

    Run MCMC using BlackJax and return posterior samples, log probability of your 
    MCMC chain, samples from your priors and the actual emcee.EnsembleSampler [for testing].

    Parameters
    ----------
    xd: numpy.ndarray [N, D]
        Input data locations / predictor variable(s)

    yd: numpy.ndarray [N,1]
        Observed data

    covfunc: function
        Covariance function

    cov_priors: list of scipy.stats.rv_continuous objects
        List containing prior probability distribution for each parameter of the covfunc
     
    noise_prior: scipy.stats.rv_continuous object       
        Prior for I.I.D. noise

    Other Parameters
    ----------------
    meanfunc: function [None]
        Mean function 

    mean_priors: list of scipy.stats.rv_continuous objects
        List containing prior probability distribution for each parameter of the meanfunc

    mean_kwargs: dict
        Key word arguments for the mean function


    gp_kwargs: dict
        Key word arguments for the GPclass initialisation

    nwarmup: int
        see emcee.EnsembleSampler
    
    niter: int
        see emcee.EnsembleSampler.run_mcmc

    nprior: int 
        number of samples from the prior distributions to output


    Returns
    --------
    states:
        MCMC chains after burn in
 
    """
        
    priors  = [noise_prior] + cov_priors + mean_priors 
    ncovparams = len(cov_priors)
    nparams = len(priors)
    
    def logprob_fn(params):

        noise = params[0]
        covparams = params[1:ncovparams+1]
        meanparams = params[ncovparams+1:]

        ## Add on the priors
        log_prior = jnp.array([P.logpdf(val) for P, val in zip(priors, params)])
        sum_prior = jnp.sum(log_prior)
        #sum_prior = 0.
        
        #if jnp.any(jnp.isinf(log_prior)):
        #    return -np.inf

        myGP =GPtideJax(xd, xd, noise, covfunc, covparams, mean_func=meanfunc,
                            mean_params=meanparams, 
                            mean_kwargs=mean_kwargs, 
                            cov_kwargs=cov_kwargs,
                            **gp_kwargs)

        logp = myGP.log_marg_likelihood(yd)
        return logp + sum_prior
    
    # Build the kernel
    # Tunable
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.ones((nparams),) # Tunable
        
    nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
    
    # Initialise and do the warmup
    if initvals is None:
        initvals = jnp.array([pp.rvs() for pp in priors])
    
    if nwarmup>0:
        seed = jax.random.PRNGKey(1234)
        adapt = blackjax.window_adaptation(blackjax.nuts, logprob_fn, nwarmup, progress_bar=True)
        last_state, kernel, _ = adapt.run(seed, initvals)
        initial_state = nuts.init(last_state.position)

    else:
        initial_state = nuts.init(initvals)
        kernel = nuts.step

    # Use the warmed-up values to run the inference
    rng_key = jax.random.PRNGKey(0)
    states = inference_loop(rng_key, kernel, initial_state, niter)
    
    # Output priors
    p0 = jnp.array([jnp.array([pp.rvs() for pp in priors]) for i in range(nprior)])
    
    return states, p0

#####
# Prior class
# Need to create classes for the priors so they behave like scipy.stats.rvs
####
# class JaxPrior(object):
#     def __init__(self, statsclass, *args):
#         self.args = args
#         self.statsclass = statsclass
#     def logpdf(self, value):
#         return self.statsclass.logpdf(value, *self.args)
from jax.scipy import special as sc
import scipy.stats as ostats
from .stats import truncnorm

twopi = 2*jnp.pi

def psi(x):
    return 1/jnp.sqrt(twopi)*jnp.exp(-0.5*x*x)

def bigpsi(x):
    return 0.5*(1 + sc.erf(x/jnp.sqrt(2)))
    
def truncnorm_pdf(x, mu, sigma, lower, upper ): 
    
    cff0 = (x-mu)/sigma
    cff1 = (upper-mu)/sigma
    cff2 = (lower-mu)/sigma
    
    # This doesn't work...
    #     if (x<upper) & (x>lower):
    #         return 1/sigma * psi(cff0) / (bigpsi(cff1) - bigpsi(cff2))
    #     else:
    #         return 0.
    
    pdf = 1/sigma * psi(cff0) / (bigpsi(cff1) - bigpsi(cff2))
    return jnp.where((x<upper) & (x>lower), pdf, 0.)

def truncnorm_logpdf(x, mu, sigma, lower, upper ):
    
    return jnp.log(truncnorm_pdf(x, mu, sigma, lower, upper ))
    
    
def invgamma_logpdf( x, a):
    return -(a+1) * jnp.log(x) - sc.gammaln(a) - 1.0/x



class JaxPrior(object):
    
    def __init__(self, distname, *args):
        self.args = args
        self.distname = distname
        
        if self.distname in ['invgamma','truncnorm']:
            self.statsclass = None
        else:
            self.statsclass = getattr(jstats, distname)
        
        if self.distname is 'truncnorm':
            self._sp = truncnorm(*self.args)
        else:
            self._sp = getattr(ostats, distname)(*self.args)
        
    def logpdf(self, value):
        if self.distname is 'invgamma':
            return invgamma_logpdf(value, *self.args)
        
        elif self.distname is 'truncnorm':
            return truncnorm_logpdf(value, *self.args)
        else:
            return self.statsclass.logpdf(value, *self.args)
    
    def rvs(self):
        return self._sp.rvs()
    
####
# Progress bar routines
# from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/

def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(f"Iteration {iter_num:,} / {num_samples:,}")

@jax.jit
def progress_bar(arg, result):
    """
    Print progress of a scan/loop only if the iteration number is a multiple of the print_rate

    Usage: `carry = progress_bar((iter_num + 1, num_samples, print_rate), carry)`
    Pass in `iter_num + 1` so that counting starts at 1 and ends at `num_samples`

    """
    iter_num, num_samples, print_rate = arg
    result = jax.lax.cond(
        iter_num % print_rate==0,
        lambda _: host_callback.id_tap(_print_consumer, (iter_num, num_samples), result=result),
        lambda _: result,
        operand=None)
    return result

def progress_bar_scan(num_samples, print_int=10):
    #num_samples=len(keys)
    def _progress_bar_scan(func):
        print_rate = int(num_samples/print_int)
        def wrapper_progress_bar(carry, iter_num):
            iter_num = progress_bar((iter_num + 1, num_samples, print_rate), iter_num)
            return func(carry, iter_num)
        return wrapper_progress_bar
    return _progress_bar_scan

    
def inference_loop(rng_key, kernel, initial_state, num_samples, print_intervals=20):
    
        keys = jax.random.split(rng_key, num_samples)

        @jax.jit
        @progress_bar_scan(num_samples, print_int=print_intervals)
        def one_step(state, iter_num):
            rng_key = keys[iter_num]
            state, _ = kernel(rng_key, state)
            return state, state
        
        _, states = jax.lax.scan(one_step, initial_state, jnp.arange(num_samples))
        
        return states

    
"""
####
def mcmc_jax_debug(
    xd, yd, 
    covfunc, 
    meanfunc, 
    priors,
    ncovparams,
    mean_kwargs={},
    cov_kwargs={},
    niter=5,
    oi_kwargs={},
    step_size = 1e-3,
    inverse_mass_matrix=None,
    initvals = None):
    
    nparams = len(priors)
    
    def logprob_fn(params):

        noise = params[0]
        covparams = params[1:ncovparams]
        meanparams = params[ncovparams:]

        ## Add on the priors
        log_prior = jnp.array([P.logpdf(val) for P, val in zip(priors, params)])
        sum_prior = jnp.sum(log_prior)
        #sum_prior = 0.
        
        #if jnp.any(jnp.isinf(log_prior)):
        #    return -np.inf

        myOI = OptimalInterpJax(xd, xd, noise, covfunc, covparams, mean_func=meanfunc,
                            mean_params=meanparams, 
                            mean_kwargs=mean_kwargs, 
                            cov_kwargs=cov_kwargs,
                            **oi_kwargs)

        logp = myOI.log_marg_likelihood(yd)
        return logp + sum_prior
    
    # Build the kernel
    # Tunable
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.ones((nparams),) # Tunable
        
    nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
    
    # Intialise and do the warmup
    if initvals is None:
        initvals = jnp.array([pp.rvs() for pp in priors])
    

    state = nuts.init(initvals)
    
    # Iterate
    rng_key = jax.random.PRNGKey(0)
    for _ in range(niter):
        _, rng_key = jax.random.split(rng_key)
        state, _ = nuts.step(rng_key, state)
        print(state)
    
    return state
"""