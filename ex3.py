import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
N = 100  # Number of basis states
alpha = 2  # Coherent state parameter

# Define cat state
cat_state = (coherent(N, alpha) - coherent(N, -alpha)).unit()

# Range for plotting
x = np.linspace(-5, 5, 100)

# Plot Wigner function of cat state
W = wigner(cat_state, x, x)
wlim = abs(W).max()
wlim = (-wlim, wlim)
plt.contourf(x, x, W, 100, cmap='viridis', vmin=wlim[0], vmax=wlim[1])
plt.title('Cat State - Wigner Function')
plt.xlabel('x')
plt.ylabel('p')
plt.colorbar(label='Wigner function')
plt.show()
