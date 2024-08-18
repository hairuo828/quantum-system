import numpy as np
import matplotlib.pyplot as plt
from qutip import *

alpha = 2  # coherent state parameter
r = 0.85  # squeezing parameter
omega = 50  # oscillator frequency
gamma = 1.5  # damping rate
N_th = 0  # vacuum environment
t = np.linspace(0, 2*np.pi/omega, 1000)  

a = destroy(50)
H = omega * (a.dag() * a + 0.5)
C1 = np.sqrt(gamma * (N_th + 1)) * a
C2 = np.sqrt(gamma * N_th) * a.dag()

x = np.linspace(-5, 5, 200)

odd_cat_state = (coherent(50, 1j*alpha) - coherent(50, -1j*alpha)).unit()

W_odd_cat = wigner(odd_cat_state, x, x)

squeezed_cat_x = squeeze(50, r * np.exp(1j * 0))*odd_cat_state
W_squeezed_cat_x = wigner(squeezed_cat_x, x, x)

squeezed_cat_p = squeeze(50, r * np.exp(1j * np.pi))*odd_cat_state
W_squeezed_cat_p = wigner(squeezed_cat_p, x, x)

result_odd_cat = mesolve(H, odd_cat_state, t, c_ops=[C1, C2], e_ops=[])
result_squeezed_cat_x = mesolve(H, squeezed_cat_x, t, c_ops=[C1, C2], e_ops=[])
result_squeezed_cat_p = mesolve(H, squeezed_cat_p, t, c_ops=[C1, C2], e_ops=[])

final_odd_cat = result_odd_cat.states[-1]
W_odd_cat = wigner(final_odd_cat, x, x)

final_squeezed_cat_x = result_squeezed_cat_x.states[-1]
W_squeezed_cat_x = wigner(final_squeezed_cat_x, x, x)

final_squeezed_cat_p = result_squeezed_cat_p.states[-1]
W_squeezed_cat_p = wigner(final_squeezed_cat_p, x, x)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
wlim = max(abs(W_odd_cat).max(), abs(W_squeezed_cat_x).max(), abs(W_squeezed_cat_p).max())

axes[0].contourf(x, x, W_odd_cat, 100, cmap='PuBu', vmin=-wlim, vmax=wlim)
axes[0].set_title('odd cat states')
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

min_W_odd_cat = W_odd_cat.min()
min_W_squeezed_cat_x = W_squeezed_cat_x.min()
min_W_squeezed_cat_p = W_squeezed_cat_p.min()

print("Wigner negativity:")
print(f"odd cat states: {min_W_odd_cat}")
print(f"X direction squeezing: {min_W_squeezed_cat_x}")
print(f"P direction squeezing: {min_W_squeezed_cat_p}")

inverse_squeezing_x = squeeze(50, -r * np.exp(1j * 0))
inverse_squeezing_p = squeeze(50, -r * np.exp(1j * np.pi))

unsqueezed_cat_x = inverse_squeezing_x * final_squeezed_cat_x * squeeze(50, r * np.exp(1j * 0))
unsqueezed_cat_p = inverse_squeezing_p * final_squeezed_cat_p * squeeze(50, r * np.exp(1j * np.pi))

fidelity_initial_cat_state = fidelity(odd_cat_state, final_odd_cat)
fidelity_x = fidelity(odd_cat_state, unsqueezed_cat_x)
fidelity_p = fidelity(odd_cat_state, unsqueezed_cat_p)

print("Fidelity with initial odd cat state:")
print(f"X direction squeezing: {fidelity_x}")
print(f"P direction squeezing: {fidelity_p}")
print(f"odd cat state: {fidelity_initial_cat_state}")
