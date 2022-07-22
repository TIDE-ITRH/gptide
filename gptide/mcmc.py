"""
MCMC parameter estimation using emcee
"""

import numpy as np
import emcee
from .gpdask import GPtideDask


    
def _minfunc_prior( params, x, Z, covfunc, meanfunc, 
           ncovparams, verbose, mean_kwargs, OIclass, oi_kwargs,
           priors):
    """
    This is the log_prob_fn in emcee speak. Takes a vector in the parameter space, and any additional arguments in the 
    args kwarg of the emcee.EnsembleSampler

    params:
        A sequence of parameters, 
            - The first is IID noise
            - Then there are ncovparams parameters for the covfunc
            - The rest are for the meanfunc 

    Zulberti - 02H p(data)?
    """
    

    noise = params[0]                   
    covparams = params[1:ncovparams]   # Zulberti this terminology is confusing. One would think ncovparams is the last 
                                       # number of covparams, not the number of the last covparam. I suggest changing this. 
    meanparams = params[ncovparams:]
    
    ## Add on the priors
    log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
    if np.any(np.isinf(log_prior)):
        return -np.inf
    sum_prior = np.sum(log_prior)
    #sum_prior = 0.
    
    myOI = OIclass(x, x, noise, covfunc, covparams, mean_func=meanfunc,
                        mean_params=meanparams, mean_kwargs=mean_kwargs, **oi_kwargs)
    
    logp = myOI.log_marg_likelihood(Z)  
    
    return logp + sum_prior             # Return the sume of the logs (log of the product)


def mcmc(
    xd, yd, 
    covfunc, 
    meanfunc, 
    priors,
    ncovparams,
    mean_kwargs={},
    OIclass=GPtideDask,
    verbose=False,
    nwalkers=200, nwarmup=200, niter=20, nprior=500,
    oi_kwargs={},
    parallel=True):
    """
    Main MCMC function
    """
    
    if parallel:
        import os 
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from multiprocessing import Pool

    ndim = len(priors)

    p0 = [np.array([pp.rvs() for pp in priors]) for i in range(nwalkers)]
    
    if parallel:
        with Pool() as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                _minfunc_prior, 
                                args=(xd, yd, covfunc, meanfunc, 
                                        ncovparams, verbose, mean_kwargs, 
                                        OIclass, oi_kwargs,
                                        priors),
                                pool=pool)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nwarmup, progress=True)
            sampler.reset()

            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                _minfunc_prior, 
                                args=(xd, yd, covfunc, meanfunc, 
                                        ncovparams, verbose, mean_kwargs, 
                                        OIclass, oi_kwargs,
                                        priors),
                                 )

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nwarmup, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    
    # Output priors
    p0 = np.array([np.array([pp.rvs() for pp in priors]) for i in range(nprior)])
    
    return samples, p0, sampler