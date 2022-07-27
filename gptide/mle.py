from .gpscipy import GPtideScipy
import numpy as np
from scipy.optimize import minimize

##############
# Estimate parameters
##############

def mle(xd, 
        yd, 
        covfunc, 
        covparams_ic,
        noise_ic,
        meanfunc=None, 
        meanparams_ic=[],
        mean_kwargs={},
        GPclass=GPtideScipy,
        gp_kwargs={},
        priors=None,
        method = 'L-BFGS-B',
        bounds = None,
        options = None,
        callback = None,
        verbose=False):

    """
    Main MLE function

    Optimise the GP kernel parameters by minimising the negative log marginal likelihood/probability using 
    scipy.optimize.minimize. If priors are specified the log marginal probability is optimised, otherwise 
    the log marginal likelihood is minimised.

    Parameters
    ----------
    xd: numpy.ndarray [N, D]
        Input data locations / predictor variable(s)

    yd: numpy.ndarray [N,1]
        Observed data

    covfunc: function
        Covariance function

    covparams_ic: numeric
        Initial guess for the covariance function parameters
     
    noise_ic: scipy.stats.rv_continuous object       
        Initial guess for the I.I.D. noise

    Other Parameters
    ----------------
    meanfunc: function [None]
        Mean function 

    meanparams_ic: scipy.stats.rv_continuous object       
        Initial guess for the mean function parameters

    mean_kwargs: dict
        Key word arguments for the mean function

    GPclass: gptide.gp.GPtide class [GPtideScipy]
        The GP class used to estimate the log marginal likelihood

    gp_kwargs: dict
        Key word arguments for the GPclass initialisation
        
    priors:
        List containing prior probability distribution for each parameter of the noise, covfunc and meanfunc.
        If specified the log marginal probability is optimised, otherwise the log marginal likelihood is minimised.

    verbose: bool [False]
        Set to true for more output

    method: str ['L-BFGS-B']
        see scipy.optimize.minimize

    bounds: sequence or Bounds, optional [None]
        see scipy.optimize.minimize
        
    options: dict, optional [None]
        see scipy.optimize.minimize
        
    callback: callable, optional [None]
        see scipy.optimize.minimize
        
    Returns
    --------
    res: OptimizeResult
        Result of the scipy.optimize.minimize call

    """

    ncovparams = len(covparams_ic)+1
    myargs = (xd,  yd,  covfunc, meanfunc, ncovparams, verbose, mean_kwargs, GPclass, gp_kwargs, priors)
    myminfunc = _minfunc
    
    params_ic = [noise_ic,] + covparams_ic + meanparams_ic
    
    return minimize(myminfunc, params_ic,
             args=myargs,
                method=method,
                bounds=bounds,
                options=options,
                callback=callback,
             ) 

def _minfunc( params, 
             x, 
             Z, 
             covfunc, 
             meanfunc, 
             ncovparams, 
             verbose, 
             mean_kwargs, 
             GPclass, 
             gp_kwargs,
             priors):
    
    """
    Function to be minimised.
    """

    if verbose:
        print(params)

    noise      = params[0]
    covparams  = params[1:ncovparams]
    meanparams = params[ncovparams:]
    
    ## Add on the priors
    sum_prior = 0.
    if priors is not None:
        log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
        if np.any(np.isinf(log_prior)):
            return 1e25
        sum_prior = np.sum(log_prior)
        
    myOI = GPclass(x, x, noise, covfunc, covparams, mean_func=meanfunc,
                        mean_params=meanparams, mean_kwargs=mean_kwargs, **gp_kwargs)
    nll = -myOI.log_marg_likelihood(Z)
    if verbose:
        print(nll)
    
    return nll - sum_prior

