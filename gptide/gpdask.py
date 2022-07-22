import numpy as np
import dask
import dask.array as da

from .gp import GPtide

class GPtideDask(GPtide):
    """
    Optimal interpolation using scipy to do the heavy lifting
    
    """
    chunksize = None

    def __init__(self, xd, xm, sd, cov_func, cov_params, **kwargs):
        self.__dict__.update(kwargs)
        
        self.N, self.D = xd.shape
        self.M, D = xm.shape
        
        # P is the number of outputs
        self.N = self.N*self.P
        
        if self.chunksize is None:
            self.chunksize = self.N//2
            
        # Chunks must divide up the array exactly
        assert self.N%self.chunksize==0

        xd = da.from_array(xd, chunks=(self.chunksize,1))
        xm = da.from_array(xm, chunks=(self.chunksize,1))
        
        GPtide.__init__(self, xd, xm, sd, cov_func, cov_params, **kwargs)
        
        
    def __call__(self, yd):
        
        assert yd.shape[0] == self.N, ' first dimension in input data must equal '
        
        if isinstance(yd, np.ndarray):
            yd = da.from_array(yd, chunks=(self.chunksize,1))

        v = da.linalg.solve_triangular(self.L, yd - self.mu_d, lower=True)
        alpha = da.linalg.solve_triangular(self.L.T, v, lower=False)
        
        return self.mu_m + self.Kmd.dot(alpha)
        
    
    def _calc_cov(self, cov_func, cov_params):
        # Compute the covariance functions
        Kmd = cov_func(self.xm, self.xd.T, cov_params, **self.cov_kwargs)
        Kdd = cov_func(self.xd, self.xd.T, cov_params, **self.cov_kwargs)
        return Kmd, Kdd
    
    def _calc_weights(self, Kdd, sd, Kmd):
        
        # Calculate the cholesky factorization
        w_md = None
        
        sigI = da.eye(self.N,chunks=self.chunksize)*(sd**2+1e-7)
        
        L = da.linalg.cholesky(Kdd+sigI, lower=True).persist()

        return L, w_md

    def _calc_err(self, diag=True):

        Kmm = self.cov_func(self.xm, self.xm.T, self.cov_params, **self.cov_kwargs)
        Kdm = self.cov_func(self.xd, self.xm.T, self.cov_params, **self.cov_kwargs)
        
        v_tmp = da.linalg.solve_triangular(self.L, Kdm, lower=True)
        v = da.linalg.solve_triangular(self.L.T, v_tmp, lower=False)
        
        V = Kmm - v.T.dot(Kdm)
        
        if diag:
            return da.diag(V)
        else:
            return V.rechunk(self.M)
        
    def sample_posterior(self, yd, samples):
        
        # Predict the mean
        ymu = self.__call__(yd)

        # Predict the covariance
        Σ = self._calc_err(diag=False) # Add noise
        
        L = da.linalg.cholesky(Σ + 1e-7*da.eye(self.M, chunks=self.chunksize), lower=True).persist()
        myrand = da.random.normal(size=(self.M,samples))
        return ymu + L.dot(myrand)
        
        #return np.random.multivariate_normal(ymu.ravel(), Σ, samples).T
        
    def sample_prior(self, samples):
        
        myrand = da.random.normal(size=(self.N,samples)) 
        
        return self.mu_d + self.L.dot(myrand)

    def log_marg_likelihood(self, yd):
        
        if isinstance(yd, np.ndarray):
            yd = da.from_array(yd, chunks=(self.chunksize,1))
        
        logdet = 2*da.sum(da.log(da.diagonal(self.L)))
        
        v = da.linalg.solve_triangular(self.L, yd-self.mu_d, lower=True)
        alpha = da.linalg.solve_triangular(self.L.T, v, lower=False)
        
        qdist = da.dot( (yd-self.mu_d).T, alpha)[0,0]
        
        fac = self.N * da.log(2*np.pi)
        
        return -0.5*(logdet + qdist + fac).compute()
      
      
            