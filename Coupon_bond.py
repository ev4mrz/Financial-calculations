import numpy as np
import matplotlib.pyplot as plt


# Bond parameters
coupon_rate = 0.08
maturity = 8
frequency = 4 
notional = 100
t=0

# DATA
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

payment_yields = np.interp(payment_time, maturities, yields)

discount_factors = np.exp(-payment_yields * payment_time)

print( price_coupon_bond(discount_factors, maturity,
                                       coupon_rate, notional, frequency))

