import numpy as np 
import matplotlib.pyplot as plt

# Vasicek parameters
gamma_star = 0.55
r_star = 0.04
sigma = 0.045
r0 = 0.08
delta = 0.01

# Bond parameters
coupon_rate = 0.08
maturity = 10
frequency = 4 
notional = 100
t = 0

def price_coupon_bond(discount_factors, maturity, coupon_rate, notional, frequency):
    
    maturities = np.arange(maturity, 0, -1/frequency)
    maturities = maturities[::-1]
    
    periodic_coupon = (coupon_rate / frequency) * notional
    
    cash_flows = np.full(len(maturities), periodic_coupon)
    cash_flows[-1] += notional
    
    bond_price = np.sum(cash_flows * discount_factors)
    
    return bond_price

def simulate_short_rate_euler(r0, gamma_star, r_star, sigma, T, delta, J):
    """Simulate short-rate paths under the Vasicek model using Euler discretization."""
    n_steps = int(T / delta)
    r = np.zeros((J, n_steps + 1))
    r[:, 0] = r0
    for i in range(n_steps):
        dW = np.sqrt(delta) * np.random.randn(J)
        r[:, i + 1] = r[:, i] + gamma_star * (r_star - r[:, i]) * delta + sigma * dW
    return r


def simulate_and_price_bond(r0, gamma_star, r_star, sigma, delta, J, maturity, 
                            coupon_rate, notional, frequency):
    maturities = np.arange(maturity, 0, -1/frequency)[::-1]
    n_steps = int(maturity / delta)
    time_grid = np.linspace(0, maturity, n_steps + 1)
    
    print(f"\nEULER SIMULATION: δ={delta}, J={J}")
    
    rate_paths = simulate_short_rate_euler(r0, gamma_star, r_star, sigma, maturity, delta, J)

    discount_factors_avg = []
    discount_factors_per_path = np.zeros((J, len(maturities)))
    
    for idx, t_pay in enumerate(maturities):
        n_steps_to_payment = int(t_pay / delta)
        integral_paths = np.sum(rate_paths[:, :n_steps_to_payment+1], axis=1) * delta
        df_paths = np.exp(-integral_paths)
        discount_factors_avg.append(np.mean(df_paths))
        discount_factors_per_path[:, idx] = df_paths
    
    discount_factors_avg = np.array(discount_factors_avg)
    
    bond_price_avg = price_coupon_bond(discount_factors_avg, maturity, coupon_rate, notional, frequency)
    
    bond_prices_per_path = np.zeros(J)
    for j in range(J):
        bond_prices_per_path[j] = price_coupon_bond(discount_factors_per_path[j, :], 
                                                    maturity, coupon_rate, notional, frequency)
    
    avg_price = bond_price_avg
    std_price = np.sqrt(np.sum((bond_prices_per_path - avg_price)**2) / J)
    se_price = std_price / np.sqrt(J)
    ci_price = 1.96 * se_price
    
    print(f"Average bond price: ${avg_price:.4f}")
    print(f"Standard error: ${se_price:.4f}")
    print(f"95% confidence interval: [${avg_price - ci_price:.4f}, ${avg_price + ci_price:.4f}]")
    

    avg_rates = np.mean(rate_paths, axis=0)
    std_rates = np.std(rate_paths, axis=0)
    ci_lower = avg_rates - 1.96 * std_rates / np.sqrt(J)
    ci_upper = avg_rates + 1.96 * std_rates / np.sqrt(J)
    
    return {
        "avg_price": avg_price,
        "ci_price": ci_price,
        "avg_rates": avg_rates,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "time_grid": time_grid,
        "discount_factors_avg": discount_factors_avg
    }

def simulate_and_price_ZCB(r0, gamma_star, r_star, sigma, delta, J, maturity, notional):
    n_steps = int(maturity / delta)
    time_grid = np.linspace(0, maturity, n_steps + 1)
    
    print(f"\nEULER SIMULATION: δ={delta}, J={J}")
    
    rate_paths = simulate_short_rate_euler(r0, gamma_star, r_star, sigma, maturity, delta, J)

    integral_paths = np.sum(rate_paths[:, :], axis=1) * delta
    df_paths = np.exp(-integral_paths)
    
    df_avg = np.mean(df_paths)
    
    avg_price = notional * df_avg
    
    # Standard error and confidence interval
    std_price = np.sqrt(np.sum((df_paths*notional - avg_price)**2) / J)
    se_price = std_price / np.sqrt(J)
    ci_price = 1.96 * se_price
    
    print(f"Average ZCB price: ${avg_price:.4f}")
    print(f"Standard error: ${se_price:.4f}")
    print(f"95% confidence interval: [${avg_price - ci_price:.4f}, ${avg_price + ci_price:.4f}]")
    
    # Short rate stats
    avg_rates = np.mean(rate_paths, axis=0)
    std_rates = np.std(rate_paths, axis=0)
    ci_lower = avg_rates - 1.96 * std_rates / np.sqrt(J)
    ci_upper = avg_rates + 1.96 * std_rates / np.sqrt(J)
    
    return {
        "avg_price": avg_price,
        "ci_price": ci_price,
        "avg_rates": avg_rates,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "time_grid": time_grid,
        "discount_factors_avg": df_avg
    }


def theoretical_mean(r0, gamma_star, r_star, t):
    """Calcule E^Q[r(t)]."""
    return r_star + (r0 - r_star) * np.exp(-gamma_star * t)
 
np.random.seed(42)

#for J in [100, 1000, 10000]:
#    simulate_and_price_bond(r0, gamma_star, r_star, sigma, delta=delta, J=J, maturity=maturity,
#                            coupon_rate=coupon_rate, notional=notional, frequency=frequency)

J_values = [100, 1000, 10000]
results = {}

for J in J_values:
    results[J] = simulate_and_price_bond(r0, gamma_star, r_star, sigma, delta=delta, J=J, maturity=maturity, coupon_rate=coupon_rate, 
                                         notional=notional, frequency=frequency)

#for J in J_values:
#    results[J] = simulate_and_price_ZCB(r0, gamma_star, r_star, sigma, delta, J, maturity, notional)


# Plot 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, J in enumerate(J_values):
    ax = axes[idx]
    result = results[J]
    
    theoretical = theoretical_mean(r0, gamma_star, r_star, result['time_grid'])
    
    ax.plot(result['time_grid'], result['avg_rates'], 'b-', 
            linewidth=2, label='Average simulated rate')
    
    # CI 95%
    ax.fill_between(result['time_grid'], result['ci_lower'], result['ci_upper'],
                    alpha=0.3, color='blue', label='95% CI')
    
    # Mean
    ax.plot(result['time_grid'], theoretical, 'r--', 
            linewidth=2, label=r'$\mathbb{E}^{\mathbb{Q}}_0[r(t)]$')
    
    ax.set_xlabel('Time t', fontsize=11)
    ax.set_ylabel('Short rate r(t)', fontsize=11)
    ax.set_title(f'J = {J}\nBond Price: ${result["avg_price"]:.4f} ± ${result["ci_price"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.03, 0.09])


plt.show()