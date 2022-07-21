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
        
        # Evaluate the covariance functions
        self.Kmd, self.Kdd = self._calc_cov(cov_func, cov_params)
        
        # Evaluate the mean function
        if self.mean_func is None:
            self.mu_d = 0.
            self.mu_m = 0.
        else:
            self.mu_d = self.mean_func(self.xd, self.mean_params, **self.mean_kwargs)
            self.mu_m = self.mean_func(self.xm, self.mean_params, **self.mean_kwargs)
        
        # Calculate the cholesky of Kdd for later use
        self.L, self.w_md = self._calc_weights(self.Kdd, self.sd, self.Kmd)
        
    def prior(self, samples=1):
        raise NotImplementedError
    
    def conditional(self, yd, samples=1):
        raise NotImplementedError
        
    def log_marg_likelihood(self, yd):
        raise NotImplementedError
        
    def update_xm(self, xm):
        self.M, _ = xm.shape
        self.xm = xm
        self.Kmd = self.cov_func(self.xm, self.xd.T, self.cov_params, **self.mean_kwargs) 
        
    def __call__(self, yd):
        raise NotImplementedError
        
    def _calc_cov(self, cov_func, cov_params):
        raise NotImplementedError
    
    def _calc_weights(self, Kdd, sd, Kmd):
        raise NotImplementedError
        
    def _calc_err(self, diag=True):
        raise NotImplementedError
