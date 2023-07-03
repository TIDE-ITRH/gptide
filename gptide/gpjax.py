import jax.numpy as jnp
from jax import jit
from jax import random as jrandom
import jax.scipy.linalg as jla
from jax import config

from .gp import GPtide

config.update("jax_enable_x64", True)
twopi = 2*jnp.pi


class GPtideJax(GPtide):
    """
        Gaussian Process regression class

        Uses Jax to do the heavy lifting

        Parameters
        ----------
        xd: numpy.ndarray [N, D]
            Input data locations
        xm: numpy.ndarray [M, D]
            Output/target point locations
        sd: float
            Data noise parameter
        cov_func: callable function 
            Function used to compute the covariance matrices
        cov_params: tuple
            Parameters passed to `cov_func`

        Other Parameters
        ----------------
        key: uint32
            random key pair
        P: int (default=1)
            number of output dimensions
        cov_kwargs: dictionary, optional
            keyword arguments passed to `cov_func`
        mean_func: callable function
            Returns the mean function
        mean_params: tuple
            parameters passed to the mean function
        mean_kwargs: dict
            kwargs passed to the mean function

        """
    def __init__(self, xd, xm, sd, cov_func, cov_params, key=None, **kwargs):
        """
        Initialise GP object and evaluate mean and covatiance functions. 
        """
        
        GPtide.__init__(self, xd, xm, sd, cov_func, cov_params, **kwargs)
        
        # Update the random number key
        if key is None:
            key = jrandom.PRNGKey(42)
            self.key, subkey = jrandom.split(key)
        else:
            self.key = key
        
    def __call__(self, yd):
        """
        Predict the GP posterior mean given data

        Parameters
        ----------

        yd: numpy.ndarray [N,1]
            Observed data

        Returns
        --------

        numpy.ndarray
            Prediction

        """
        assert yd.shape[0] == self.N, ' first dimension in input data must equal '
        
        alpha = jla.cho_solve((self.L, True), yd - self.mu_d)
        
        return self.mu_m + self.Kmd.dot(alpha)
    
    def prior(self, samples=1, noise=0.):
        """
        Sample from the prior distribution

        Parameters
        ----------

        samples: int, optional (default=1)
            number of samples

        Returns
        -------

        prior_sample: numpy.ndarray [N,samples]
            array of output samples
             
        """
        return self._sample_prior(samples, noise=noise)
    
    def conditional(self, yd, samples=1):
        """
        Sample from the conidtional distribution

        Parameters
        ----------

        yd: numpy.ndarray [N,1]
            Observed data
        samples: int, optional (default=1)
            number of samples

        Returns
        -------

        conditional_sample: numpy.ndarray [N,samples]
            output array

        """
        return self._sample_posterior(yd, samples)
    
    def log_marg_likelihood(self, yd):
        """Compute the log of the marginal likelihood"""
        logdet = 2*jnp.sum(jnp.log(jnp.diagonal(self.L)))
        
        alpha = jla.cho_solve((self.L, True), yd - self.mu_d)
        
        qdist = jnp.dot( (yd-self.mu_d).T, alpha)[0,0]
        
        fac = self.N * jnp.log(twopi)
        
        return -0.5*(logdet + qdist + fac)      

         
    def _calc_cov(self, cov_func, cov_params):
        """Compute the covariance functions"""
        Kmd = cov_func(self.xm, self.xd.T, cov_params, **self.cov_kwargs)
        Kdd = cov_func(self.xd, self.xd.T, cov_params,  **self.cov_kwargs)
        
        return Kmd, Kdd
    
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """Calculate the cholesky factorization"""
        noise = (sd*sd + 1e-7)*jnp.eye(Kdd.shape[0])
        L = jla.cholesky(Kdd+noise, lower=True)
        w_md = None

        return L, w_md

    def _calc_err(self, diag=True): 
        """
        
        Compute the covariance of the conditional distribution

        Used by .conditional

        Not calculated with _calc_cov as it is not always needed.

        """
        Kmm = self.cov_func(self.xm, self.xm.T, self.cov_params, *self.cov_args, **self.cov_kwargs)
        Kdm = self.cov_func(self.xd, self.xm.T, self.cov_params, *self.cov_args, **self.cov_kwargs)
        
        v = jla.cho_solve((self.L, True),  Kdm)
        
        V = Kmm - v.T.dot(Kdm)
        
        if diag:
            return jnp.diag(V)
        else:
            return V

            
    def _sample_posterior(self, yd, samples):
        
        # Predict the mean
        ymu = self.__call__(yd)

        # Predict the covariance
        Σ = self._calc_err(diag=False)
        
        self.key, sub_key = self._update_key(self.key)

        myrand = jrandom.normal(sub_key, shape=(self.M, samples))
        
        jitter = 1e-7*jnp.eye(Σ.shape[0])
        noise =  (self.sd**2)*jnp.eye(self.M)
        L = jla.cholesky(Σ+jitter+noise, lower=True)
        

        return ymu + L.dot(myrand)

        
        
    def _sample_prior(self, samples, noise=0.):
        
        self.key, sub_key = self._update_key(self.key)

        myrand = jrandom.normal(sub_key, shape=(self.N,samples)) 
        
        

        return self.mu_d + self.L.dot(myrand) + noise*myrand

    def _update_key(self, old_key):
        # See https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
        
        new_key, sub_key = jrandom.split(old_key)
        return new_key, sub_key
    