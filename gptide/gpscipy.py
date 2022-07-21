import numpy as np
from scipy import linalg as la

from .gp import GPtide

class GPtideScipy(GPtide):
    
    def __init__(self, xd, xm, sd, cov_func, cov_params, **kwargs):
        """
        Gaussian Process regression class

        Uses scipy to do the heavy lifting

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
        
        GPtide.__init__(self, xd, xm, sd, cov_func, cov_params, **kwargs)
        
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
        
        alpha = la.cho_solve((self.L, True), yd - self.mu_d)
        
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
                
        logdet = 2*np.sum(np.log(np.diagonal(self.L)))
        
        alpha = la.cho_solve((self.L, True), yd - self.mu_d)
        
        qdist = np.dot( (yd-self.mu_d).T, alpha)[0,0] # original

        fac = self.N * np.log(2*np.pi)
        
        return -0.5*(logdet + qdist + fac)

         
    def _calc_cov(self, cov_func, cov_params):
        """Compute the covariance functions"""
        Kmd = cov_func(self.xm, self.xd.T, cov_params, **self.cov_kwargs)
        Kdd = cov_func(self.xd, self.xd.T, cov_params, **self.cov_kwargs)
        
        return Kmd, Kdd
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """Calculate the cholesky factorization"""
        L = la.cholesky(Kdd+(sd**2+1e-7)*np.eye(self.N), lower=True)
        w_md = None

        return L, w_md

    def _calc_err(self, diag=True):
        """
        Compute the covariance of the conditional distribution

        Used by .conditional
        """

        Kmm = self.cov_func(self.xm, self.xm.T, self.cov_params, **self.cov_kwargs)
        Kdm = self.cov_func(self.xd, self.xm.T, self.cov_params, **self.cov_kwargs)
        
        
        v = la.cho_solve((self.L, True),  Kdm)
        
        V = Kmm - v.T.dot(Kdm)
        
        if diag:
            return np.diag(V)
        else:
            return V
            
    def _sample_posterior(self, yd, samples):
        
        # Predict the mean
        ymu = self.__call__(yd)

        # Predict the covariance
        Σ = self._calc_err(diag=False)
        
        myrand = np.random.normal(size=(self.M,samples))
        L = la.cholesky(Σ+1e-7*np.eye(self.M), lower=True)
        
        return ymu + L.dot(myrand)
        
        
    def _sample_prior(self, samples, noise=0.):
        
        myrand = np.random.normal(size=(self.N,samples)) 
        
        return self.mu_d + self.L.dot(myrand) + noise*myrand

    
