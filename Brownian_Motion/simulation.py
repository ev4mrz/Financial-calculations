import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(T, delta):

    N = int(T / delta)               # number of steps
    t = np.linspace(0, T, N + 1)    
    W = np.zeros(N + 1)              
    for i in range(N):
        W[i + 1] = W[i] + np.sqrt(delta) * np.random.randn()
    return t, W
    
T = 1.0        
delta = 0.01   
M = 5          

plt.figure(figsize=(10, 6))

for j in range(M):
    t, W = simulate_brownian_motion(T, delta)
    plt.plot(t, W, lw=1.5, label=f"Trajectory {j+1}")

plt.title("Simulated Brownian Motion", fontsize=14, fontweight='bold')
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()

