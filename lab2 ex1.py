import numpy as np
import matplotlib.pyplot as plt
from qutip import *

alpha = 2  # coherent state parameter
r = 0.5  # squeezing parameter

a = destroy(50)

odd_cat_state = (coherent(50, 1j*alpha) - coherent(50, -1j*alpha)).unit()

x = np.linspace(-5, 5, 200)

W_odd_cat = wigner(odd_cat_state, x, x)

squeezed_cat_x = squeeze(50, r * np.exp(1j * 0))*odd_cat_state
W_squeezed_cat_x = wigner(squeezed_cat_x, x, x)

squeezed_cat_p = squeeze(50, r * np.exp(1j * np.pi))*odd_cat_state
W_squeezed_cat_p = wigner(squeezed_cat_p, x, x)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
wlim = abs(W_odd_cat).max()

axes[0].contourf(x, x, W_odd_cat, 100, cmap='PuBu', vmin=-wlim, vmax=wlim)
axes[0].set_title('odd cat state')
axes[0].set_xlabel('x')
axes[0].set_ylabel('p')
axes[0].set_aspect('equal')

axes[1].contourf(x, x, W_squeezed_cat_x, 100, cmap='PuBu', vmin=-wlim, vmax=wlim)
axes[1].set_title('X direction squeezing')
axes[1].set_xlabel('x')
axes[1].set_ylabel('p')
axes[1].set_aspect('equal')

axes[2].contourf(x, x, W_squeezed_cat_p, 100, cmap='PuBu', vmin=-wlim, vmax=wlim)
axes[2].set_title('P direction squeezing')
axes[2].set_xlabel('x')
axes[2].set_ylabel('p')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show()
