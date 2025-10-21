import numpy as np
import matplotlib.pyplot as plt

# Market data
maturities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50,
                       2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00,
                       5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50,
                       7.75, 8.00, 8.25, 8.50, 8.75, 9.00])

yields = np.array([0.0617010, 0.0632000, 0.0652114, 0.0671270, 0.0676471, 
                   0.0679336, 0.0687412, 0.0692366, 0.0691518, 0.0689033,
                   0.0687555, 0.0686239, 0.0683940, 0.0679085, 0.0675003,
                   0.0677334, 0.0677255, 0.0674582, 0.0670663, 0.0669710,
                   0.0670661, 0.0669115, 0.0668679, 0.0665782, 0.0661633,
                   0.0660896, 0.0658362, 0.0656600, 0.0655917, 0.0656780,
                   0.0650526, 0.0649443, 0.0649143, 0.0648180, 0.0642875,
                   0.0640249])

# Ho-Lee parameters
delta = 0.01
sigma = 0.045
r0 = 0.06
T=3.5

def fit_and_compute_theta(maturities, yields, sigma, T, delta=0.01, degree=10):

    # Fit polynomial to market yields
    coeffs = np.polyfit(maturities, yields, degree)
    poly = np.poly1d(coeffs)

    times = np.arange(0, maturities[-1], delta)
    
    # Instantaneous forward rates
    forward_rates = np.array([poly(t) + t*(poly(t+delta)-poly(t))/delta for t in times])
    
    # Theta for Ho-Lee
    theta = np.array([(forward_rates[i+1]-forward_rates[i])/delta + sigma**2 * times[i]
                      for i in range(len(forward_rates)-1)])
    
    return theta


def ho_Lee_rates(theta, r0, sigma, delta=0.01):
    """
    Compute Ho-Lee discount rates, skipping time 0 to avoid division by zero.
    """
    n = len(theta) + 1
    times = np.arange(1, n) * delta 
    dfs = []
    for i, t in enumerate(times):
        integral = np.sum([(t - j*delta) * theta[j] * delta for j in range(i+1)])
        A = -integral + (sigma**2 * t**3)/6
        B = t
        df = np.exp(A - B * r0)
        dfs.append(df)
    rates = -np.log(dfs) / times
    return rates, times


def ho_lee_rate(maturity, theta, delta, r0, sigma):
    """Compute Ho-Lee rate for a specific maturity"""
    n = int(maturity / delta)
    integral = np.sum([(maturity - j*delta) * theta[j] * delta for j in range(n)])
    A = -integral + (sigma**2 * maturity**3)/6
    B = maturity
    df = np.exp(A - B * r0)
    rate = -np.log(df) / maturity
    return rate


def plot_yield_curves(times, ho_lee_rates, market_maturities, market_yields):
    """Plot Ho-Lee yield curve vs market yields"""
    plt.figure(figsize=(10,6))
    plt.plot(times, ho_lee_rates, label='Ho-Lee curve', color='red')
    plt.scatter(market_maturities, market_yields, label='Market data', color='blue')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title('Yield Curve - Ho-Lee Model vs Market')
    plt.legend()
    plt.grid(True)
    plt.show()



theta = fit_and_compute_theta(maturities, yields, sigma, delta)
rates_curve, times_curve = ho_Lee_rates(theta, r0, sigma, delta)

# Plot yield curve
plot_yield_curves(times_curve, rates_curve, maturities, yields)

# Rate for a specific maturity
rate = ho_lee_rate(T, theta, delta, r0, sigma)
print(f"Ho-Lee rate for maturity {T} years: {rate:.6f}")


