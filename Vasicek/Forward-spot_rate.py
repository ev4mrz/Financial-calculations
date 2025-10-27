import numpy as np
import matplotlib.pyplot as plt

# Vasicek model parameters
gamma = 0.4653     
r_star = 0.0634    
sigma = 0.0221   
t=0


def B_vasicek(t, T, gamma_star):
    tau = T - t
    if gamma_star == 0:
        return tau
    return (1 - np.exp(-gamma_star * tau)) / gamma_star

def A_vasicek(t, T, gamma_star, r_star, sigma):
    tau = T - t
    B = B_vasicek(t, T, gamma_star)
    term1 = (B - tau) * (r_star - sigma**2 / (2 * gamma_star**2))
    term2 = -sigma**2 * B**2 / (4 * gamma_star)
    return term1 + term2


def spot_rate(t, T, r0, gamma, r_star, sigma):
    """Calculate continuously compounded spot rate r(t,T)"""
    tau = T - t
    if tau == 0:
        return r0
    B = B_vasicek(t, T, gamma)
    A = A_vasicek(t, T, gamma, r_star, sigma)
    return -(A - B * r0) / tau

def forward_rate(t, T1, T2, r0, gamma, r_star, sigma):
    """Calculate continuously compounded forward rate f(r(t),t,T1,T2)"""
    if T2 <= T1:
        return np.nan
    A1 = A_vasicek(t, T1, gamma, r_star, sigma)
    A2 = A_vasicek(t, T2, gamma, r_star, sigma)
    B1 = B_vasicek(t, T1, gamma)
    B2 = B_vasicek(t, T2, gamma)
    return -(A2 - A1 - (B2 - B1) * r0) / (T2 - T1)

def forward_discount_factor(t, T1, T2, r0, gamma, r_star, sigma):
    """Calculate the forward discount factor Z(r(t),t,T1,T2)"""
    if T2 <= T1:
        return np.nan
    A1 = A_vasicek(t, T1, gamma, r_star, sigma)
    A2 = A_vasicek(t, T2, gamma, r_star, sigma)
    B1 = B_vasicek(t, T1, gamma)
    B2 = B_vasicek(t, T2, gamma)
    return np.exp((A2 - A1 - (B2 - B1) * r0))