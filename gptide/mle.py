from .gpscipy import GPtideScipy
import numpy as np
from scipy.optimize import minimize

##############
# Estimate parameters
##############
def minfunc( params, x, Z, covfunc, meanfunc, 
            ncovparams, verbose, mean_kwargs, GPclass, oi_kwargs,
           priors):
    
    if verbose:
        print(params)

    noise = params[0]
    covparams = params[1:ncovparams]
    meanparams = params[ncovparams:]
    
    ## Add on the priors
    sum_prior = 0.
    if priors is not None:
        log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
        if np.any(np.isinf(log_prior)):
            return 1e25
        sum_prior = np.sum(log_prior)
        
    
    myOI = GPclass(x, x, noise, covfunc, covparams, mean_func=meanfunc,
                        mean_params=meanparams, mean_kwargs=mean_kwargs, **oi_kwargs)
    nll = -myOI.log_marg_likelihood(Z)
    if verbose:
        print(nll)
    
    return nll - sum_prior


def mle(
    xd, yd, 
    covfunc, covparams_ic,
    meanfunc, meanparams_ic,
    noise_ic,
    mean_kwargs={},
    GPclass=GPtideScipy,
    verbose=False,
    priors=None,
    method = 'L-BFGS-B',
    bounds = None,
    options = None,
    callback = None,
    gp_kwargs={}):
    """
    Optimise the GP kernel parameters by minimising the negative log marginal likelihood
    """

    ncovparams = len(covparams_ic)+1
    myargs = (xd,  yd,  covfunc, meanfunc, ncovparams, verbose, mean_kwargs, GPclass, oi_kwargs, priors)
    myminfunc = minfunc
    
    params_ic = (noise_ic,)+covparams_ic+meanparams_ic
    
    return minimize(myminfunc, params_ic,
             args=myargs,
                method=method,
                bounds=bounds,
                options=options,
                callback=callback,
             ) 



