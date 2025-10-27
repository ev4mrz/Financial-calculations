import numpy as np 

# Vasicek model with sigma =0

r0=0.08
r_mean=0.07
gamma=0.04
T=4

def short_rate(T):
    return r0+ r_mean*(1- np.exp(-gamma*T))

print( short_rate(T))