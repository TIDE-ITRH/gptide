"""
Covariance functions for optimal interpolation
"""

import numpy as np
from scipy.special import kv as K_nu
from scipy.special import gamma

####
# 2D Covariance models (placeholder here for N-dimensional non separable kernels [matrix inputs])

####
# 1D Covariance models (these are used as building blocks for separable 2d/3d functions)

def expquad_1d(x, xpr, params):
    eta, l = params
    return eta**2. * expquad(x, xpr, l)

def matern12_1d(x, xpr, params):
    eta, l = params
    return eta**2. * matern12(x, xpr, l)

def matern32_1d(x, xpr, params):
    eta, l = params
    return eta**2. * matern32(x, xpr, l)

def matern52_1d(x, xpr, params):
    eta, l = params
    return eta**2. * matern52(x, xpr, l)

def cosine_1d(x, xpr, params):
    eta, l = params
    return eta**2. * cosine(x, xpr, l)

def periodic_1d(x, xpr, params):
    eta, l, p = params
    return eta**2*periodic(x, xpr, l, p)


def matern_general_1d(x, xpr, params):
    eta, nu, l = params
    dx = np.sqrt((x-xpr)*(x-xpr))
    return matern_general(dx, eta, nu, l)

### Raw functions (these can be used to build higher dimensional separable kernels)

def matern52(x,xpr,l):
    """Matern 5/2 base function, -6 high frequency spectral slop"""
    fac1 = 5*(x-xpr)*(x-xpr)
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l + fac1/(3*l*l) )*np.exp(-fac2/l)

def matern32(x,xpr,l):
    """Matern 3/2 base function, -4 high frequency spectral slope"""
    fac1 = 3*(x-xpr)*(x-xpr)
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l)*np.exp(-fac2/l)

def matern12(x,xpr,l):
    """Matern 1/2 base function, -2 high frequency spectral slope """
    fac1 = (x-xpr)*(x-xpr)
    fac2 = np.sqrt(fac1)
    return np.exp(-fac2/l)

def periodic(x, xpr, l, p):
    """Periodic base function"""
    d = np.abs(x-xpr)
    sin1 = np.sin(np.pi*d/p)
    sin2 = sin1*sin1
    l2 = l*l
    cff = -2/l2
    return np.exp(cff*sin2)

def cosine(x, xpr, l):
    """Cosine base function"""
    return np.cos(2*np.pi*np.abs(x-xpr)/l)

def cosine_rw06(x, xpr, l):
    """Cosine base function"""
    return np.cos(np.pi*np.abs(x-xpr)/(l*l))

def se(x, xpr, l):
    return expquad(x, xpr, l)
    """Exponential quadration base function/Squared exponential/RBF"""
    
def rbf(x, xpr, l):
    return expquad(x, xpr, l)
    """Exponential quadration base function/Squared exponential/RBF"""
    
def expquad(x, xpr, l):
    """Exponential quadration base function/Squared exponential/RBF"""
    return np.exp(-(x-xpr)*(x-xpr)/(2*l*l))

def matern_general(dx, eta, nu, l):
    """General Matern base function, high frequency spectral slope is -(2\nu + 1)"""
    
    cff1 = np.sqrt(2*nu)*np.abs(dx)/l
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu)
    K *= np.power(cff1, nu)
    K *= K_nu(nu,cff1)
    
    K[np.isnan(K)] = np.power(eta, 2.)
    
    return K

### Matrix input kernels

def matern32_matrix(d,l):
    """
    Non scaled [var 1] Matern 3/2 that takes a distance martrix as an input (i.e. not a vector of coordinates as above)
    
    Parameters
    ==========
    l: Matern length scale
    d: distance matrix
    """

    fac1 = 3*d**2
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l)*np.exp(-fac2/l)

### Spectral estimation routines
def matern_spectra(f, eta, nu, l, n=1):
    """Helper routine to compute the power spectral density of a general Matern function"""
    
    S = np.power(eta,2.) * np.power(2.,n) * np.power(np.pi, 0.5*n) 
    S *= gamma(nu+0.5*n) * np.power(2*nu, nu)
    
    cff1 = gamma(nu)*np.power(l, 2*nu)
    cff2 = 2*nu/l**2 + 4*np.pi**2*f**2
    
    S /= cff1
    S *= np.power(cff2, -(nu+0.5*n))
    
    return S

