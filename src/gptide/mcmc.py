"""
MCMC parameter estimation using emcee
"""

import numpy as np
import emcee
from .gpscipy import GPtideScipy


def mcmc(   xd, 
            yd, 
            covfunc, 
            cov_priors,
            noise_prior,
            meanfunc=None,             
            mean_priors=[],
            mean_kwargs={},
            GPclass=GPtideScipy,
            gp_kwargs={},
            nwalkers=None, 
            nwarmup=200, 
            niter=20, 
            nprior=500,
            parallel=False,
            verbose=False, 
            progress=True):
    """
    Main MCMC function

    Run MCMC using emcee.EnsembleSampler and return posterior samples, log probability of your 
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
     
    noise_priors: scipy.stats.rv_continuous object       
        Prior for I.I.D. noise

    Other Parameters
    ----------------
    meanfunc: function [None]
        Mean function 

    mean_priors: list of scipy.stats.rv_continuous objects
        List containing prior probability distribution for each parameter of the meanfunc

    mean_kwargs: dict
        Key word arguments for the mean function

    GPclass: gptide.gp.GPtide class [GPtideScipy]
        The GP class used to estimate the log marginal likelihood

    gp_kwargs: dict
        Key word arguments for the GPclass initialisation

    nwalkers: int or None
        see emcee.EnsembleSampler. If None it will be 20 times the number of parameters. 

    nwarmup: int
        see emcee.EnsembleSampler
    
    niter: int
        see emcee.EnsembleSampler.run_mcmc

    nprior: int 
        number of samples from the prior distributions to output

    parallel: bool [False]
        Set to true to run parallel

    verbose: bool [False]
        Set to true for more output

    Returns
    --------
    samples:
        MCMC chains after burn in

    log_prob:
        Log posterior probability for each sample in the MCMC chain after burn in

    p0:
        Samples from the prior distributions
    
    sampler: emcee.EnsembleSampler
        The actual emcee.EnsembleSampler used

    """
    
    if parallel:
        import os 
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        from multiprocessing import Pool

    priors  = [noise_prior] + cov_priors + mean_priors 
    ncovparams = len(cov_priors)

    ndim = len(priors)

    if nwalkers is None:
        nwalkers = 20*ndim

    p0 = [np.array([pp.rvs() for pp in priors]) for i in range(nwalkers)]
    
    if parallel:
        with Pool() as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                _minfunc_prior, 
                                args=(xd, yd, covfunc, meanfunc, 
                                        ncovparams, verbose, mean_kwargs, 
                                        GPclass, gp_kwargs,
                                        priors),
                                pool=pool)

            if progress:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nwarmup, progress=progress)
            sampler.reset()

            if progress:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                _minfunc_prior, 
                                args=(xd, yd, covfunc, meanfunc, 
                                        ncovparams, verbose, mean_kwargs, 
                                        GPclass, gp_kwargs,
                                        priors),
                                 )

        if progress:
                print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, nwarmup, progress=progress)
        sampler.reset()

        if progress:
                print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=progress)
        
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    log_prob = sampler.get_log_prob()[:, :].reshape((-1, 1))

    # Output priors
    p0 = np.array([np.array([pp.rvs() for pp in priors]) for i in range(nprior)])
    
    return samples, log_prob, p0, sampler
 
def _minfunc_prior( params, 
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
    Function to be maximised.
    
    This is the log_prob_fn in emcee speak. Takes a vector in the parameter space, and any additional arguments in the 
    args kwarg of the emcee.EnsembleSampler

    """
    

    noise = params[0]                   
    covparams = params[1:ncovparams+1]   # Zulberti this terminology was confusing. Now actually the number of cov params. 
    meanparams = params[ncovparams+1:]
    
    ## Add on the priors
    log_prior = np.array([P.logpdf(val) for P, val in zip(priors, params)])
    if np.any(np.isinf(log_prior)):
        return -np.inf
    sum_prior = np.sum(log_prior)
    #sum_prior = 0.
    
    myGP = GPclass(x, x, noise, covfunc, covparams, mean_func=meanfunc,
                        mean_params=meanparams, mean_kwargs=mean_kwargs, **gp_kwargs) # Zulberti - this is initialised on every iteration. Suspect significant gains by just updating params. 
    
    logp = myGP.log_marg_likelihood(Z)  
    
    return logp + sum_prior             # Return the sume of the logs (log of the product)


def _tune_scale(acc_rate, target_accept=0.3, max_log_step=0.5):
    """Bounded multiplicative step-size rescaling factor for a window acceptance rate.

    Moves the step size gradually towards `target_accept`, but the
    per-window change is capped at `exp(+/-max_log_step)` (~1.65x by
    default in either direction) regardless of how extreme the observed
    acceptance rate is. This matters because the window acceptance rate
    is noisy (small window, and possibly extra noise from re-randomized
    Vecchia orderings between evaluations) - an earlier, uncapped
    bucketed version of this could see one unlucky all-rejections window
    and shrink the step size 10x in a single update, which then tends to
    compound across further windows with no way back. Bounding the
    per-window change means correction still happens, just gradually
    enough that a single noisy window can't cause a runaway collapse.
    """
    delta = np.clip(acc_rate - target_accept, -max_log_step, max_log_step)
    return np.exp(delta)


def mh(
    xd,
    yd,
    covfunc,
    cov_priors,
    noise_prior,
    meanfunc=None,
    mean_priors=[],
    mean_kwargs={},
    steps=1/5,
    GPclass=GPtideScipy,
    gp_kwargs={},
    nwarmup=100,
    niter=100,
    tune=True,
    tune_interval=50,
    progress=True
):
    """ Metropolis Hasting (MH) sampler

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

    noise_priors: scipy.stats.rv_continuous object
        Prior for I.I.D. noise

    Other Parameters
    ----------------
    meanfunc: function [None]
        Mean function

    mean_priors: list of scipy.stats.rv_continuous objects
        List containing prior probability distribution for each parameter of the meanfunc

    mean_kwargs: dict
        Key word arguments for the mean function

    steps: float or list [1/5]
        Initial steps of the MH sampling. If float, step for infered parameter i is (up-low)*step
        where up and low are the lower and upper bounds of parameter. Rescaled during warmup if
        `tune=True` (in which case this just sets the relative starting scale between parameters).

    GPclass: gptide.gp.GPtide class [GPtideScipy]
        The GP class used to estimate the log marginal likelihood

    gp_kwargs: dict
        Key word arguments for the GPclass initialisation

    nwarmup: int
        Number of warmup steps. Step-size tuning (if enabled) only happens during these steps;
        `steps` is frozen for the remaining `niter` steps, since adapting during actual sampling
        would break the chain's stationarity guarantees.

    niter: int
        see emcee.EnsembleSampler.run_mcmc

    tune: bool [True]
        Adaptively rescale `steps` during warmup based on the rolling acceptance rate. The whole
        step vector is rescaled by a single shared factor each `tune_interval`, so the relative
        per-parameter scaling set by `steps` is preserved - this is a global scale search, not a
        per-parameter one, since accept/reject is a single joint decision over all parameters.

    tune_interval: int [50]
        Number of proposals between step-size rescalings during warmup.

    progress: bool [False]
        Show progress of sampling

    Returns
    --------
    samples: nd.array
        MCMC chains including burn in

    log_prob: nd.array
        Log posterior probability for each sample in the MCMC chain including burn in

    accept_samples: nd.array
        Samples from the prior distributions

    attrs: dict
        Dict of inference parameter configuration

    """
        
    n_mcmc = nwarmup+niter
    
    # number of parameters infered
    priors  =  cov_priors + mean_priors + [noise_prior]
    ncovparams = len(cov_priors)
    ndim = len(priors)

    # bounds and initializations - assumes truncnorm dist !
    lowers = np.array([p.kwds["loc"]+p.a*p.kwds["scale"] for p in priors])
    uppers = np.array([p.kwds["loc"]+p.b*p.kwds["scale"] for p in priors])
    initialisations = np.array([(up+low)*.5 for up, low in zip(lowers, uppers)])

    # step sizes 
    if isinstance(steps, float):
        step_sizes = np.array([ (up-low)*steps for low, up in zip(lowers, uppers)])

    # setup objects
    samples = [np.empty(n_mcmc) for _ in range(ndim)]
    accept_samples = np.empty(n_mcmc)
    lp_samples = np.empty(n_mcmc)
    lp_samples[:] = np.nan
    # init samples
    for i, s in enumerate(samples):
        s[0] =  initialisations[i]
    accept_samples[0] = 0
    
    # run mcmc once
    get_params = lambda p: (float(p[-1]), p[:ncovparams], p[ncovparams:-1])
    noise, covparams, meanparams = get_params(initialisations)
    gp_current = GPclass(
        xd, xd, 
        noise, 
        covfunc, covparams,
        mean_func=meanfunc, mean_params=meanparams, mean_kwargs=mean_kwargs,
        **gp_kwargs,
    )

    # Rolling window used to drive step-size tuning during warmup (see
    # _tune_scale). Reset every `tune_interval` proposals.
    window_accepts = 0
    window_count = 0

    for i in tqdm(np.arange(1, n_mcmc), disable=(not progress)):

        proposed = np.array([
            np.random.normal(s[i-1], step, 1)
            for s, step in zip(samples, step_sizes)
        ]).squeeze()

        if ((proposed.T <= lowers) | (proposed.T >= uppers)).any():
            # Out-of-bounds proposals are rejected without evaluating the
            # likelihood, but still count towards the acceptance-rate
            # window below - an oversized step size shows up here too.
            for s in samples:
                s[i] = s[i-1]
            lp_samples[i] = lp_samples[i-1]
            accept_samples[i] = 0
        else:
            if accept_samples[i-1] == True:
                gp_current = gp_proposed

            noise, covparams, meanparams = get_params(proposed)
            gp_proposed = GPclass(
                xd, xd,
                noise,
                covfunc, covparams,
                mean_func=meanfunc, mean_params=meanparams, mean_kwargs=mean_kwargs,
                **gp_kwargs,
            )

            lp_current = gp_current.log_marg_likelihood(yd)
            lp_proposed = gp_proposed.log_marg_likelihood(yd)

            alpha = np.exp(min(0.0, lp_proposed - lp_current))
            u = np.random.uniform()

            if alpha > u:
                for s, p in zip(samples, proposed):
                    s[i] = p
                accept_samples[i] = 1
                lp_samples[i] = lp_proposed
            else:
                for s, p in zip(samples, proposed):
                    s[i] = s[i-1]
                accept_samples[i] = 0
                lp_samples[i] = lp_samples[i-1]

        window_accepts += accept_samples[i]
        window_count += 1

        # Adaptive step-size tuning: rescale the whole step vector by a
        # single shared factor (preserving relative per-parameter scale)
        # based on the rolling acceptance rate. Only runs during warmup -
        # `steps` is frozen for the rest of sampling.
        if tune and i <= nwarmup and window_count >= tune_interval:
            step_sizes = step_sizes * _tune_scale(window_accepts / window_count)
            window_accepts = 0
            window_count = 0

    #samples = np.vstack([samples[-1]]+samples[:-1])
    samples = np.vstack(samples)

    attrs = dict(
        lowers=lowers, uppers=uppers,
        init=initialisations,
        steps=steps,
        final_step_sizes=step_sizes,
    )

    return samples, lp_samples, accept_samples, attrs
