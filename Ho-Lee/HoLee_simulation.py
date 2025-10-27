import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 5                 
delta = 0.01          # time step
M = 5                 # number of trajectories

# Ho-Lee parameters
r0 = 0.03             
sigma = 0.01          
theta = 0.001         #constant drift


def simulate_ho_lee(r0, theta, sigma, T, delta):
    N = int(T / delta)
    t = np.linspace(0, T, N + 1)
    r = np.zeros(N + 1)
    r[0] = r0

    for i in range(N):
        dW = np.sqrt(delta) * np.random.randn()
        r[i + 1] = r[i] + theta * delta + sigma * dW

    return t, r

# plot multiple simulations
plt.figure(figsize=(10, 6))

for j in range(M):
    t, r = simulate_ho_lee(r0, theta, sigma, T, delta)
    plt.plot(t, r, lw=1.8, label=f"Trajectory {j+1}")

plt.title('Simulations of the Ho–Lee Short Rate Model (5 Trajectories)', fontsize=14, fontweight='bold')
plt.xlabel('Time (years)')
plt.ylabel('Short rate $r_t$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
