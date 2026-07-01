import numpy as np
from scipy import linalg as la

from .gp import GPtide

class GPtideToeplitz(GPtide):
    """
        Gaussian Process regression class

        Uses Toeplitz matrices for fast computation
        
        Note the sampling from the priors and conditionals is not implemented 
        for this class

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
    def __init__(self, xd, xm, sd, cov_func, cov_params, **kwargs):
        """
        Initialise GP object and evaluate mean and covatiance functions. 
        """
        
        GPtide.__init__(self, xd, xm, sd, cov_func, cov_params, **kwargs)
        
        assert self.D == 1, 'Only 1D inputs supported by the Toeplitz solver.'
        assert self.P == 1, 'Only univariate outputs supported by the Toeplitz solver.'

        
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
        
        alpha = la.solve_toeplitz(self.L, yd - self.mu_d)
        
        return self.mu_m + self.Kmd.dot(alpha)
  
    
    def log_marg_likelihood(self, yd):
        """Compute the log of the marginal likelihood"""
                
        _, logdet = modified_trench_helper(self.L)
        
        alpha = la.solve_toeplitz(self.L, yd - self.mu_d)
        
        qdist = np.dot( (yd-self.mu_d).T, alpha)[0,0] # original

        fac = self.N * np.log(2*np.pi)
        
        return -0.5*(logdet + qdist + fac)

         
    def _calc_cov(self, cov_func, cov_params):
        """Compute the covariance functions
        
        TODO: should just calculate the first row
        
        """
        #Kmd = cov_func(self.xm, self.xd.T, cov_params, **self.cov_kwargs)
        #Kdd = cov_func(self.xd, self.xd.T, cov_params, **self.cov_kwargs)
        Kmd = cov_func(self.xm, self.xd[0,...], cov_params, **self.cov_kwargs).T
        Kdd = cov_func(self.xd, self.xd[0,...], cov_params, **self.cov_kwargs)
        
        return Kmd, Kdd
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """Store the Toeplitz matrix first columnn"""
        L = Kdd[:,0]
        L[0] += (sd**2+1e-7)
        w_md = None

        return L, w_md

    def _calc_err(self, diag=True): 
        """
        Compute the covariance of the conditional distribution

        Used by .conditional

        Not calculated with _calc_cov as it is not always needed. 
        """

        Kmm = self.cov_func(self.xm, self.xm.T, self.cov_params, **self.cov_kwargs)
        Kdm = self.cov_func(self.xd, self.xm.T, self.cov_params, **self.cov_kwargs) 
        
        v = la.solve_toeplitz(self.L,  Kdm)
        
        V = Kmm - v.T.dot(Kdm)
        
        if diag:
            return np.diag(V)
        else:
            return V
            


    
"""
Toeplitz solvers from:
    https://github.com/a5a/toeplitz
    
We just use algs 1.2 and 1.3 to calculate the log-determinant
    
Reference:
    Zhang, Y., Leithead, W. E., & Leith, D. J. (2005). Time-series Gaussian process regression based on Toeplitz computation of O (N 2) operations and O (N)-level storage. Decision and Control, 2005 and 2005 European Control Conference. CDC-ECC’05. 44th IEEE Conference on, 3711–3716.
"""

def toeplitz_inverse(m):
    """
    Computes a fast inverse of a Toeplitz matrix (Alg. 1.1) and
    log det m
    mat = numpy.ndarray with Toeplitz form
    """
    c = m[:, 0]
    N = len(c)
    c_inv = np.zeros((N, N))  # intialising the matrix
    v, l = modified_trench_helper(c)

    c_inv[0, :] = v[::-1]
    c_inv[:, 0] = v[::-1]
    c_inv[-1, :] = v
    c_inv[:, -1] = v

    for ii in range(1, int(np.floor((N-1)/2)+1)):
        for jj in xrange(ii, N - ii):
            c_inv[ii, jj] = (c_inv[ii-1, jj-1]
                             + (v[N-jj-1] * v[N-ii-1] - v[ii-1] * v[jj-1])
                             / v[-1])
            c_inv[jj, ii] = c_inv[ii, jj]
            c_inv[N-ii-1, N-jj-1] = c_inv[ii, jj]
            c_inv[N-jj-1, N-ii-1] = c_inv[ii, jj]

    return c_inv, l


def modified_trench_helper(c):
    """
    The modified Trench algorithm (Alg. 1.2)
    c is a vector containing the first column of the matrix
    """
    N = len(c)
    wiggle = c[1:]/c[0]
    z, l = modified_durbin(len(wiggle), wiggle)
    l = l + N * np.log(c[0])

    v = np.zeros(N)  # initialising
    v[-1] = 1/((1 + wiggle.dot(z)) * c[0])
    v[:-1] = v[-1] * z[::-1]

    return v, l


def modified_durbin(m, wiggle):
    """
    Modified Durbin algorithm
    (Alg. 1.3)
    """
    z = np.zeros(m)

    z[0] = -wiggle[0]
    beta = 1
    alpha = -wiggle[0]
    l = 0

    for ii in range(m-1):
        beta = (1 - alpha**2) * beta
        l = l + np.log(beta)
        if ii == 0:  # there has to be a better way than this...?
            alpha = - (wiggle[ii+1] + wiggle[0] * z[0]) / beta
            z[0] = z[0] + alpha*z[0]
        else:
            # print alpha
            alpha = - (wiggle[ii+1] + wiggle[ii::-1].dot(z[:ii+1])) / beta
            z[:ii+1] = z[:ii+1] + alpha*z[ii::-1]
        z[ii+1] = alpha

    beta = (1 - alpha**2) * beta
    l = l + np.log(beta)

    return z, l