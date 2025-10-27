import numpy as np
import matplotlib.pyplot as plt

# inadequate for long term simulation, r(t) can be negative

def ho_lee_no_drift(r0, sigma, T, delta):
    "r(t) = r0 + sigma * X(t)"
    N = int(T / delta)
    t = np.linspace(0, T, N + 1)
    r = np.zeros(N + 1)
    r[0] = r0

    for i in range(N):
        r[i + 1] = r[i] + sigma * np.sqrt(delta) * np.random.randn()
    
    return t, r


r0 = 0.03
sigma = 0.01
T = 5
delta = 0.01
M = 5

plt.figure(figsize=(10,6))
for j in range(M):
    t, r = ho_lee_no_drift(r0, sigma, T, delta)
    plt.plot(t, r, label=f'Trajectory {j+1}')

plt.title("Ho–Lee Short Rate without Drift")
plt.xlabel("Time (years)")
plt.ylabel("r(t)")
plt.grid(True)
plt.legend()
plt.show()
