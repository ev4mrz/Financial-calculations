import numpy as np 

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


def simulate_and_price_bond(r0, gamma_star, r_star, sigma, delta, J, maturity, coupon_rate, notional, frequency):
    """Run Vasicek Euler simulation and compute bond price statistics."""
    
    maturities = np.arange(maturity, 0, -1/frequency)
    maturities = maturities[::-1]
    
    print(f"\nEULER SIMULATION: δ={delta}, J={J}")
    
    # Simulate short-rate paths
    rate_paths = simulate_short_rate_euler(r0, gamma_star, r_star, sigma, maturity, delta, J)
    
    # Compute bond prices for each simulated path
    bond_prices = []
    for j in range(J):
        rates = rate_paths[j, :]
        discount_factors_sim = []
        
        for t_pay in maturities:
            # Integrate short rate from 0 to t_pay to get discount factor
            n_steps_to_payment = int(t_pay / delta)
            # Use trapezoidal rule to integrate: ∫₀ᵗ r(s)ds
            integral = np.trapz(rates[:n_steps_to_payment + 1], 
                               dx=delta)
            df = np.exp(-integral)
            discount_factors_sim.append(df)
            
        bond_price = price_coupon_bond(np.array(discount_factors_sim),
                                       maturity, coupon_rate, notional, frequency)
        bond_prices.append(bond_price)

    bond_prices = np.array(bond_prices)
    avg_price = np.mean(bond_prices)
    std_price = np.std(bond_prices, ddof=1)
    se_price = std_price / np.sqrt(J)
    ci = 1.96 * se_price

    print(f"Average bond price: ${avg_price:.4f}")
    print(f"Standard error: ${se_price:.4f}")
    print(f"95% CI: [${avg_price - ci:.4f}, ${avg_price + ci:.4f}]")

    avg_rates = np.mean(rate_paths, axis=0)
    std_rates = np.std(rate_paths, axis=0)

    return avg_price, std_price, avg_rates, std_rates

np.random.seed(42)

for J in [100, 1000, 10000]:
    simulate_and_price_bond(r0, gamma_star, r_star, sigma, delta=delta, J=J, maturity=maturity,
                            coupon_rate=coupon_rate, notional=notional, frequency=frequency)