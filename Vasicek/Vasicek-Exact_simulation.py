import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Vasicek parameters
gamma_star = 0.55
r_star = 0.04
sigma = 0.045
r0 = 0.08

# Time horizon
T = 10

def simulate_euler(r0, gamma_star, r_star, sigma, T, delta):
    """Simulate short-rate path using Euler scheme."""
    n_steps = int(T / delta)
    t = np.linspace(0, T, n_steps + 1)
    r = np.zeros(n_steps + 1)
    r[0] = r0
    
    for i in range(n_steps):
        dW = np.sqrt(delta) * np.random.randn()
        r[i + 1] = r[i] + gamma_star * (r_star - r[i]) * delta + sigma * dW
    
    return t, r


def simulate_exact(r0, gamma_star, r_star, sigma, T, delta, epsilons=None):
    """Simulate short-rate path using exact simulation scheme."""
    n_steps = int(T / delta)
    t = np.linspace(0, T, n_steps + 1)
    r = np.zeros(n_steps + 1)
    r[0] = r0
    
    for i in range(n_steps):
        if epsilons is not None:
            epsilon = epsilons[i]
        else:
            epsilon = np.random.randn()
        
        mean_term = r_star + np.exp(-gamma_star * delta) * (r[i] - r_star)
        variance_term = np.sqrt((sigma**2 / (2 * gamma_star)) * (1 - np.exp(-2 * gamma_star * delta)))
        
        r[i + 1] = mean_term + variance_term * epsilon
    
    return t, r

def Vasicek_mean(r0, gamma_star, r_star, t):
    """Compute theoretical mean E^Q[r(t)]."""
    return r_star + (r0 - r_star) * np.exp(-gamma_star * t)

# Step sizes to compare
deltas = [2, 0.5, 0.1, 0.01]

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, delta in enumerate(deltas):
    ax = axes[idx]
    
    # Generate one vector of iid standard normal random variables
    n_steps = int(T / delta)
    
    # Simulate both with same random seed
    np.random.seed(42 + idx)
    t_euler, r_euler = simulate_euler(r0, gamma_star, r_star, sigma, T, delta)
    
    np.random.seed(42 + idx)
    t_exact, r_exact = simulate_exact(r0, gamma_star, r_star, sigma, T, delta)
    
    # Compute theoretical mean
    t_theory = np.linspace(0, T, 1000)
    mean_theory = Vasicek_mean(r0, gamma_star, r_star, t_theory)
    
    # Plot trajectories
    ax.plot(t_euler, r_euler, 'b-', alpha=0.7, linewidth=1.5, label='Euler')
    ax.plot(t_exact, r_exact, 'r--', alpha=0.7, linewidth=1.5, label='Exact')
    ax.plot(t_theory, mean_theory, 'g-', linewidth=2, label='Mean Theory')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Short rate r(t)')
    ax.set_title(f'delta = {delta}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.12])

plt.tight_layout()
plt.show()