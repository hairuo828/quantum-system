import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
omega = 1  # Oscillator frequency
gamma = 1  # Damping rate
n_th = 5  # Mean thermal occupation
t = np.linspace(0, 10, 100)  # Time array

# Operator
a = destroy(50)  # Define operators with 100 basis states
H = omega * (a.dag() * a + 0.5)
C1 = np.sqrt(gamma * (1 + n_th)) * a
C2 = np.sqrt(gamma * n_th) * a.dag()

cat_state = (coherent(50, 2) - coherent(50, -2)).unit()

result = mesolve(H, cat_state, t, c_ops=[C1, C2], e_ops=[])

# Range for plotting
x = np.linspace(-5, 5, 100)

# Define the time points to plot
time = [0, 19 , 29, 39, 49, 59, 69, 79, 89, 99]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i, idx in enumerate(time):
    final_state = result.states[i]
    W = wigner(final_state, x, x)
    wlim = abs(W).max()
    wlim = (-wlim, wlim)
    axes[i].contourf(x, x, W, 100, cmap='viridis', vmin=wlim[0], vmax=wlim[1])
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('p')
    axes[i].set_aspect('equal')
plt.tight_layout()
plt.show()




