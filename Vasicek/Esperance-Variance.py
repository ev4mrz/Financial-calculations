import numpy as np 

# Vasicek parameters
gamma_star = 0.55
r_star = 0.04
sigma = 0.045
r0 = 0.08

def Vasicek_esperance(r0, gamma_star, r_star, t):
    """Compute theoretical mean E^Q[r(t)]."""
    return r_star + (r0 - r_star) * np.exp(-gamma_star * t)

def Vasicek_variance(t,sigma,gamma_star):
    """Compute theoretical variance Var[r(t)]."""
    return ((sigma^2)/(2*gamma_star))*(1-np.exp(-2*gamma_star*t))
