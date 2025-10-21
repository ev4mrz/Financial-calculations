import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Vasicek Parameters
gamma_star = 0.55
r_star = 0.04
sigma = 0.045
r0 = 0.08

# Cap parameters
t = 0
T = 5
RK = 0.035   # Cap rate (3.5%)
n = 2        # Semiannual payments
N = 100      # Notional


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

def vasicek_price(t, T, gamma_star, r_star, sigma, r0):
    A = A_vasicek(t, T, gamma_star, r_star, sigma)
    B = B_vasicek(t, T, gamma_star)
    return np.exp(A - B * r0)

def sigmaT_vasicek(t, T0, TB, gamma_star, sigma):
    if T0 <= t or TB <= T0:
        return 0.0
    tau = T0 - t
    B = B_vasicek(T0, TB, gamma_star)
    term = (1 - np.exp(-2 * gamma_star * tau)) / (2 * gamma_star)
    return sigma * np.sqrt(term) * B

def put_price(t, T0, TB, gamma_star, r_star, sigma, r0, K, N):
    sigmaT = sigmaT_vasicek(t, T0, TB, gamma_star, sigma)
    if sigmaT == 0:
        return 0.0

    Z_TB = vasicek_price(t, TB, gamma_star, r_star, sigma, r0)
    Z_T0 = vasicek_price(t, T0, gamma_star, r_star, sigma, r0)

    d1 = (np.log(Z_TB / (Z_T0 * K / N)) + 0.5 * sigmaT**2) / sigmaT
    d2 = d1 - sigmaT
    put = K * Z_T0 * norm.cdf(-d2) - N * Z_TB * norm.cdf(-d1)
    return put


def caplet_price(t, T0, TB, gamma_star, r_star, sigma, r0, RK, N):
    if T0 <= t or TB <= T0:
        return 0.0
    tau = TB - T0
    K_bond = 1.0 / (1.0 + tau * RK)
    put_zcb = put_price(t, T0, TB, gamma_star, r_star, sigma, r0, K_bond, 1.0)
    caplet = N * (1.0 + tau * RK) * put_zcb
    return caplet


def cap_price(t, T, gamma_star, r_star, sigma, r0, RK, N, n):
    maturities = np.arange(T,0,-1/n)
    maturities = maturities[::-1]
    caplets = np.array([caplet_price(t, T-1/n, T, gamma_star, r_star, sigma, r0, RK, N) for T in maturities])

    for i, c in enumerate(caplets):
        print(f"Caplet {i+1}: T0={maturities[i]-1/n:.4f}, TB={maturities[i]:.4f}, Price={c:.6f}")

    return np.sum(caplets)

print (cap_price(t, T, gamma_star, r_star, sigma, r0, RK, N, n))