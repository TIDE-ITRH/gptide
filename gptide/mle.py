from .gpdask import GPtideDask
# from .oiscipy import OptimalInterpScipy

import numpy as np
from scipy.optimize import minimize

##############
# Estimate parameters
##############
def minfunc( params, x, Z, covfunc, meanfunc, 
            ncovparams, verbose, mean_kwargs, OIclass, oi_kwargs,
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
        
    
    myOI = OIclass(x, x, noise, covfunc, covparams, mean_func=meanfunc,
                        mean_params=meanparams, mean_kwargs=mean_kwargs, **oi_kwargs)
    nll = -myOI.log_marg_likelihood(Z)
    if verbose:
        print(nll)
    
    return nll - sum_prior


def optimise_oi(
    xd, yd, 
    covfunc, covparams_ic,
    meanfunc, meanparams_ic,
    noise_ic,
    mean_kwargs={},
    OIclass=GPtideDask,
    verbose=False,
    priors=None,
    method = 'L-BFGS-B',
    bounds = None,
    options = None,
    callback = None,
    oi_kwargs={}):
    """
    Optimise the OI parameters by minimising the negative log marginal likelihood
    """

    ncovparams = len(covparams_ic)+1
    myargs = (xd,  yd,  covfunc, meanfunc, ncovparams, verbose, mean_kwargs, OIclass, oi_kwargs, priors)
    myminfunc = minfunc
    
    params_ic = (noise_ic,)+covparams_ic+meanparams_ic
    
    return minimize(myminfunc, params_ic,
             args=myargs,
                method=method,
                bounds=bounds,
                options=options,
                callback=callback,
             ) 



# Functions to go in the oceanoi package
def run_oi_variable_xin(xin, yin, zin, Xout, Yout, cov_func, cov_params, sd, calc_err=False):
    """
    Spatial optimal interpolation on a time-series of input points with variable location
    e.g. drifters, satellite
    
    Inputs:
       - xin, yin: Arrays of input x and y data locations, shape [Nt, Nx]
       - zin: input data to be interpolated, shape [Nt, Nx]
       - Xout, Yout: vector of output data points, shape [Nxout]
       - cov_func: covariance function to parse to the OI class (function)
       - cov_params: covariance function parameters (tuple)
       - sd: OI error standard deviation (scalar or vector with shape [Nx])
       - calc_err: calculate OI error (standard deviation)
    
    Outputs:
        - zout: array [Nt, Nxout]: Interpolated output data
        - zerr: array [Nt, Nxout]: error standard deviation (only if calc_err is True)
    """
    # Check sizes are OK
    assert xin.shape == yin.shape
    assert zin.shape == xin.shape
    assert Xout.shape == Yout.shape
    
    Nt,Nx = zin.shape
    
    Zout = np.zeros((Nt,)+Xout.shape)
    if calc_err:
        Zerr = np.zeros((Nt,)+Xout.shape)

    for ii in range(Nt):
        # Flag bad data points
        myz = zin[ii,:]
        goodidx = ~np.isnan(myz)

        # Initialise the OI object
        OI = oi.OptimalInterp2D(xin[ii,goodidx], yin[ii,goodidx],\
                Xout.ravel(), Yout.ravel(), sd, cov_func, cov_params)
        
        Zout[ii,...] = OI(zin[ii,goodidx,None]).reshape(Xout.shape)
        if calc_err:
            Zerr[ii,...] = OI.calc_err().reshape(Xout.shape)
        
    if calc_err:
        return Zout, Zerr
    else:
        return Zout

####
# GP parameter optimization/estimation routines


####
# Old optimisation
def minfuncold( params, x, Z, covfunc):
    noise = params[0]
    covparams = params[1:]
    
    myOI = oi.OptimalInterp1D(x, x, noise, covfunc, covparams)
    nll = -myOI.log_marg_likelihood(Z)

    return nll

def minfunc_prior( params, priors, x, Z, covfunc):
    noise = params[0]
    covparams = params[1:]
    
    myOI = oi.OptimalInterp1D(x, x, noise, covfunc, covparams)
    nll = -myOI.log_marg_likelihood(Z)
    
    ## Add on the priors
    log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
    if np.any(np.isinf(log_prior)):
        return 1e25
    
    return nll - np.sum(log_prior)

def optimise_1d_oi(
    xd, yd, covfunc, covparams_ic,
    priors = None,
    scale=1,
    method = 'L-BFGS-B',
    bounds = None,
    options = None):
    """
    Optimise the OI parameters by minimising the negative log marginal likelihood
    """
    if priors is None:
        myargs = (xd*scale,  yd[:,None],  covfunc)
        myminfunc = minfunc
    else:
        myargs = (priors, xd*scale,  yd[:,None],  covfunc)
        myminfunc = minfunc_prior
        
    return minimize(myminfunc, covparams_ic,
             args=myargs,
                method=method,
                bounds=bounds,
                options=options,
             ) 
