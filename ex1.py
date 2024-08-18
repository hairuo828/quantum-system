import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
N = 100  # Number of basis states
n_th = 5  # Mean thermal occupation

# Define states
vacuum_state = basis(N, 0)  # Vacuum state
coherent_state = coherent(N, 2)  # Coherent state
thermal_state = thermal_dm(N, n_th)  # Thermal state

# Range for plotting
x = np.linspace(-5, 5, 100)

# Plot Wigner function and quadrature probability distribution for each state
states = [vacuum_state, coherent_state, thermal_state]
titles = ['Vacuum State', 'Coherent State', 'Thermal State (n_th = 5)']

for i, state in enumerate(states):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Wigner function
    W = wigner(state, x, x)
    wlim = abs(W).max()
    wlim = (-wlim, wlim)
    axes[0].contourf(x, x, W, 100, cmap='viridis', vmin=wlim[0], vmax=wlim[1])
    axes[0].set_title(titles[i] + ' - Wigner Function')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('p')

    # Quadrature probability distribution
    Q = qfunc(state, x, x)
    qlim = Q.max()
    qlim = (0, qlim)
    axes[1].contourf(x, x, Q, 100, cmap='viridis', vmin=qlim[0], vmax=qlim[1])
    axes[1].set_title(titles[i] + ' - Quadrature Probability Distribution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p')

    plt.tight_layout()
    plt.show()

