import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data
r0=0.06

maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 
                       2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0,
                       5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5,
                       7.75, 8.0, 8.25, 8.5, 8.75, 9.0])

yields = np.array([0.0617010, 0.0632000, 0.0652114, 0.0671270, 0.0676471, 
                   0.0679336, 0.0687412, 0.0692366, 0.0691518, 0.0689033,
                   0.0687555, 0.0686239, 0.0683940, 0.0679085, 0.0675003,
                   0.0677334, 0.0677255, 0.0674582, 0.0670663, 0.0669710,
                   0.0670661, 0.0669115, 0.0668679, 0.0665782, 0.0661633,
                   0.0660896, 0.0658362, 0.0656600, 0.0655917, 0.0656780,
                   0.0650526, 0.0649443, 0.0649143, 0.0648180, 0.0642875,
                   0.0640249])

def B_vasicek(t, T, gamma):
    """Calculate B(t,T) = (1 - exp(-gamma*(T-t))) / gamma"""
    tau = T - t
    if gamma == 0:
        return tau
    return (1 - np.exp(-gamma * tau)) / gamma

def A_vasicek(t, T, gamma, r_star, sigma):
    """Calculate A(t,T) according to Vasicek model formula"""
    tau = T - t
    B = B_vasicek(t, T, gamma)
    term1 = (B - tau) * (r_star - sigma**2 / (2 * gamma**2))
    term2 = -sigma**2 * B**2 / (4 * gamma)
    return term1 + term2

def vasicek_price(t, T, gamma, r_star, sigma, r0):
    """ZCB price according to Vasicek: P(t,T) = exp(A(t,T) - B(t,T)*r(t))"""
    A = A_vasicek(t, T, gamma, r_star, sigma)
    B = B_vasicek(t, T, gamma)
    return np.exp(A - B * r0)


# Market discount factors (ZCB prices)
market_prices = np.exp(-yields * maturities)

def objective(params, maturities, market_prices, r0):
    """Least squares method"""
    gamma, r_star, sigma = params
    
    if gamma <= 0 or sigma <= 0:
        return 1e10
    
    t = 0
    model_prices = np.array([
        vasicek_price(t, T, gamma, r_star, sigma, r0)
        for T in maturities
    ])
    
    sse = np.sum((model_prices - market_prices)**2)
    return sse

# Optimization
x0 = [0.3, 0.07, 0.01]
result = minimize(objective, x0, args=(maturities, market_prices, r0),
                 method='BFGS', 
                 options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

gamma_opt, r_star_opt, sigma_opt = result.x

print(f"γ (gamma)    = {gamma_opt:.6f}")
print(f"r* (r_star)  = {r_star_opt:.6f}")
print(f"σ (sigma)    = {sigma_opt:.6f}")
print(f"r₀ (r0)      = {r0:.6f}")