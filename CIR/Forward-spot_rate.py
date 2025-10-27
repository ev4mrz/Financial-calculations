import numpy as np
import matplotlib.pyplot as plt

# if gamma*r_star > apha/2 interest rate never negative

# Vasicek model parameters
gamma_star = 0.4653     
r_star = 0.0634    
sigma = 0.0221  
alpha = 0.1 

def psi_1(gamma_star, alpha):
    return np.sqrt((gamma_star)**2 + 2 * alpha)


def B_CIR(t, T, gamma_star, alpha):
    tau = T - t
    exp_term = np.exp(psi_1(gamma_star, alpha) * tau)
    numerator = 2 * (exp_term - 1)
    denominator = (gamma_star + psi_1(gamma_star, alpha)) * (exp_term - 1) + 2 * psi_1(gamma_star, alpha)
    
    return numerator / denominator


def A_CIR(t, T, gamma_star, alpha, r_star):
    tau = T - t
    exp_term = np.exp(psi_1(gamma_star, alpha) * tau)
    coeff = 2 * r_star * gamma_star / alpha
    
    numerator = 2 * psi_1(gamma_star, alpha) * np.exp((psi_1(gamma_star, alpha) + gamma_star) * tau / 2)
    denominator = (gamma_star + psi_1(gamma_star, alpha)) * (exp_term - 1) + 2 * psi_1(gamma_star, alpha)
    
    return coeff * np.log(numerator / denominator)


# Spot

def discount_factor(t, T, gamma, r_star, sigma, r0):
    """discount factor Vasicek: Z(t,T) = exp(A(t,T) - B(t,T)*r(t))"""
    A = A_CIR(t, T, gamma, r_star, sigma)
    B = B_CIR(t, T, gamma)
    return np.exp(A - B * r0)

def spot_rate(t, T, r0, gamma, r_star, sigma):
    """Calculate continuously compounded spot rate r(t,T)"""
    tau = T - t
    if tau == 0:
        return r0
    B = B_CIR(t, T, gamma)
    A = A_CIR(t, T, gamma, r_star, sigma)
    return -(A - B * r0) / tau

# Forward 

def forward_rate(t, T1, T2, r0, gamma, r_star, sigma):
    """Calculate continuously compounded forward rate f(r(t),t,T1,T2)"""
    if T2 <= T1:
        return np.nan
    A1 = A_CIR(t, T1, gamma, r_star, sigma)
    A2 = A_CIR(t, T2, gamma, r_star, sigma)
    B1 = B_CIR(t, T1, gamma)
    B2 = B_CIR(t, T2, gamma)
    return -(A2 - A1 - (B2 - B1) * r0) / (T2 - T1)

def forward_discount_factor(t, T1, T2, r0, gamma, r_star, sigma):
    """Calculate the forward discount factor Z(r(t),t,T1,T2)"""
    if T2 <= T1:
        return np.nan
    A1 = A_CIR(t, T1, gamma, r_star, sigma)
    A2 = A_CIR(t, T2, gamma, r_star, sigma)
    B1 = B_CIR(t, T1, gamma)
    B2 = B_CIR(t, T2, gamma)
    return np.exp((A2 - A1 - (B2 - B1) * r0))