import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# t=0

# Market data

maturities = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,
                       2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,
                       5.25,5.5,5.75,6.0,6.25,6.5,6.75,7.0,7.25,7.5,
                       7.75,8.0,8.25,8.5,8.75,9.0])

yields = np.array([0.0617010,0.0632000,0.0652114,0.0671270,0.0676471,
                   0.0679336,0.0687412,0.0692366,0.0691518,0.0689033,
                   0.0687555,0.0686239,0.0683940,0.0679085,0.0675003,
                   0.0677334,0.0677255,0.0674582,0.0670663,0.0669710,
                   0.0670661,0.0669115,0.0668679,0.0665782,0.0661633,
                   0.0660896,0.0658362,0.0656600,0.0655917,0.0656780,
                   0.0650526,0.0649443,0.0649143,0.0648180,0.0642875,
                   0.0640249])


# Parameters

delta = 0.01
sigma = 0.045
r0 = 0.06

# call parameters
t=0
T0 = 1
TB = 6
K = 90
N=100


def fit_and_compute_theta(maturities, yields, sigma, delta=0.01, degree=10):
    coeffs = np.polyfit(maturities, yields, degree)
    poly = np.poly1d(coeffs)
    times = np.arange(0, maturities[-1], delta)
    forward_rates = np.array([poly(t) + t*(poly(t+delta)-poly(t))/delta for t in times])
    theta = np.array([(forward_rates[i+1]-forward_rates[i])/delta + sigma**2*times[i]
                      for i in range(len(forward_rates)-1)])
    return theta

def A_HoLee(T, theta, sigma, delta=0.01):
    n = int(T/delta)
    integral = np.sum([(T - j*delta)*theta[j]*delta for j in range(n)])
    return -integral + sigma**2 * T**3 / 6

def B_HoLee(T):
    return T

def discount_factor(T, theta, sigma, r0, delta=0.01):
    A = A_HoLee(T, theta, sigma, delta)
    B = B_HoLee(T)
    return np.exp(A - B*r0)

def sigmaT_HoLee(t, T0, TB, sigma):
    if T0 <= t or TB <= T0:
        return 0.0
    return (TB-T0)**2*sigma**2*T0

def call_price(t, T0, TB,theta, sigma, r0, K, N):
    sigmaT = sigmaT_HoLee(t, T0, TB, sigma)
    if sigmaT == 0:
        return 0.0

    Z_TB = discount_factor(TB, theta, sigma, r0, delta=0.01)
    Z_T0 = discount_factor(T0, theta, sigma, r0, delta=0.01)

    d1 = (np.log(Z_TB / (Z_T0 * K / N)) + 0.5 * sigmaT**2) / sigmaT
    d2 = d1 - sigmaT
    put =  N * Z_TB * norm.cdf(d1)-K * Z_T0 * norm.cdf(d2)
    return put

theta = fit_and_compute_theta(maturities, yields, sigma, delta=delta)
print (call_price(t, T0, TB,theta, sigma, r0, K, N))
