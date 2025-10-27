import numpy as np
import matplotlib.pyplot as plt

# Vasicek Parameters

#gamma_star = 0.486332
#r_star = 0.103604
#sigma = 0.144686
#r0 = 0.06

gamma_star = 0.55
r_star = 0.04
sigma = 0.045
r0 = 0.08

# Bond parameters
coupon_rate = 0.08
maturity = 10
frequency = 4 
notional = 100
t=0

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

def price_coupon_bond(discount_factors, maturity, coupon_rate, notional, frequency):
    
    maturities = np.arange(maturity,0,-1/frequency)
    maturities = maturities[::-1]
    
    periodic_coupon = (coupon_rate / frequency) * notional
    
    cash_flows = np.full(len(maturities), periodic_coupon)
    cash_flows[-1] += notional
    
    bond_price = np.sum(cash_flows * discount_factors)
    
    return bond_price

payment_time=np.arange(maturity,0,-1/frequency)
payment_time = payment_time[::-1]

vasicek_discount_factors = np.array([
    vasicek_price(0, T, gamma_star, r_star, sigma, r0)
    for T in payment_time
])

print( price_coupon_bond(vasicek_discount_factors, maturity,
                                       coupon_rate, notional, frequency))

