"""
Classes for Gaussian Process regression
"""


class GPtide(object):
    """
    Gaussian Process base class 

    Intended as a placeholder for classes built with other libraries (scipy, jax)
    
    """
    
    mean_func = None
    mean_params = ()
    mean_kwargs = {}
    P=1 # Number of output dimensions
    
    cov_kwargs = {}
    cov_args = ()
    
    
    def __init__(self, xd, xm, sd, cov_func, cov_params, **kwargs):
        """
        Initialise GP object and evaluate mean and covatiance functions. 
        """
        self.__dict__.update(kwargs)
        
        assert xd.ndim==2
        
        self.N, self.D = xd.shape
        self.M, D = xm.shape
        
        self.N = self.N*self.P
        self.M = self.M*self.P

        self.xd = xd
        self.xm = xm
        
        self.sd = sd
        self.cov_func = cov_func
        self.cov_params = cov_params
        
        # Evaluate the covariance functions, mean function and the
        # factorisation of Kdd
        self._build()

    def _update_mean(self):
        """Evaluate the mean function at the data and output locations"""
        if self.mean_func is None:
            self.mu_d = 0.
            self.mu_m = 0.
        else:
            self.mu_d = self.mean_func(self.xd, self.mean_params, **self.mean_kwargs)
            self.mu_m = self.mean_func(self.xm, self.mean_params, **self.mean_kwargs)

    def _build(self):
        """Evaluate the covariances, mean and factorisation with the current parameters"""
        self.Kmd, self.Kdd = self._calc_cov(self.cov_func, self.cov_params)
        self._update_mean()
        # Calculate the cholesky of Kdd for later use
        self.L, self.w_md = self._calc_weights(self.Kdd, self.sd, self.Kmd)

    def update_params(self, cov_params=None, sd=None, mean_params=None):
        """
        Update the hyper-parameters and recompute the covariance and its factorisation

        The data (`xd`, `xm`), covariance function and mean function are kept
        as they are. This is cheaper than building a new object for every
        likelihood evaluation (e.g. in MLE/MCMC), and lets subclasses reuse memory
        (e.g. GPU buffers).

        Parameters
        ----------
        cov_params: tuple, optional
            New parameters passed to `cov_func`
        sd: float, optional
            New data noise parameter
        mean_params: tuple, optional
            New parameters passed to the mean function

        Parameters left as `None` are unchanged.
        """
        if cov_params is not None:
            self.cov_params = cov_params
        if sd is not None:
            self.sd = sd
        if mean_params is not None:
            self.mean_params = mean_params

        self._build()

    def prior(self, samples=1):
        """
        Placeholder
        """
        raise NotImplementedError
    
    def conditional(self, yd, samples=1):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def log_marg_likelihood(self, yd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def update_xm(self, xm):
        """Update the output locations and the covariance kernel"""

        # print('Updating xm')

        self.M, _ = xm.shape

        self.M = self.M*self.P
        
        self.xm = xm
        self.Kmd = self.cov_func(self.xm, self.xd.T, self.cov_params, **self.cov_kwargs)
        self._update_mean()  # mu_m depends on xm
        
        # print('Updated xm')

    def __call__(self, yd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def _calc_cov(self, cov_func, cov_params):
        """
        Placeholder
        """
        raise NotImplementedError
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def _calc_err(self, diag=True):
        """
        Placeholder
        """
        raise NotImplementedError
