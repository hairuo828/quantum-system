import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
omega = 1.0  # Oscillator frequency
gamma = 1.0  # Damping rate
n_th = 5  # Mean thermal occupation
t = np.linspace(0, 10, 1000)  # Time array

# Operator
a = destroy(50)
X = (a + a.dag()) / np.sqrt(2)
H = omega * (a.dag() * a + 0.5)
C1 = np.sqrt(gamma * (1 + n_th)) * a
C2 = np.sqrt(gamma * n_th) * a.dag()

# Initial state
psi0 = basis(50, 0)  # Ground state

# Time evolution
result = mesolve(H, psi0, t, c_ops=[C1, C2], e_ops=[X * X.dag()])

# Plot position quadrature variance as a function of time
plt.plot(t, result.expect[0])
plt.xlabel('Time')
plt.ylabel('<(X_M(t))^2>')
plt.title('Mechanical Position Quadrature Variance')
plt.show()

 