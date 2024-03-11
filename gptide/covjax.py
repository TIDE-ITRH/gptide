"""
Covariance functions using jax

There is probably a better way to do this without cut and pasting...
"""

import jax.numpy as np
from jax.scipy.special import gammaln

def calc_dist(x, xpr, eps=1e-14):
    dx2 = np.power(x-xpr, 2.)
    dx2 = np.where(dx2 < eps, eps, dx2)
    return np.sqrt(dx2)
    
###
# Special functions that are not in 
def gamma(x):
    return np.exp(gammaln(x))

# Modified bessel function of the second kind
#https://github.com/google/jax/issues/9956

def phi(t):
    return np.exp(np.pi / 2 * np.sinh(t))

def dphi(t):
    return np.pi / 2 * np.cosh(t) * np.exp(np.pi / 2 * np.sinh(t))

def bessel_k(nu, z):
    
    z = np.asarray(z)[..., None]
    t = np.linspace(-3, 3, 101)[None, :]
    integrand = 0.5*(0.5*z)**nu*np.exp(-phi(t)-z**2/(4*phi(t)))*phi(t)**(-nu-1)*dphi(t)

    return np.trapz(integrand, x=t, axis=-1)

###


def expquad_1d(x, xpr, params):
    eta, l = params
    return eta**2. * expquad(x, xpr, l)

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
    dx = calc_dist(x, xpr)
    return matern_general(dx, eta, nu, l)

### Raw functions

# def compute_dist(x1, x2, fac=1.):
#     d2 = fac*(x1-x2)*(x1-x2)
#     is_zero = np.allclose(d2, 0.)
#     d2 = np.where(is_zero, np.ones_like(d2), d2)  # replace d with ones if is_zero
#     d = np.sqrt(d2)
#     return np.where(is_zero, 0., d)  # replace sqrt(d2) with zero if is_zero

def matern52(x,xpr,l):
    """Matern 5/2 base function"""
    fac1 = 5*(x-xpr)*(x-xpr)
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l + fac1/(3*l*l) )*np.exp(-fac2/l)

def matern32(x,xpr,l):
    """Matern 3/2 base function"""
    fac1 = 3*(x-xpr)*(x-xpr)
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l)*np.exp(-fac2/l)

def matern12(x,xpr,l):
    """Matern 1/2 base function"""
    fac1 = (x-xpr)*(x-xpr)
    return np.exp(-fac1/l)

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
    
    cff1 = np.sqrt(2*nu)*dx/l
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu)
    K *= np.power(cff1, nu)
    K *= bessel_k(nu,cff1)
    #x = x.at[idx].set(y)
    idx = 1/dx > 1e12
    #return K.at[idx].set(np.power(eta, 2.))
    return np.where(idx, np.power(eta,2.), K)
    #return K