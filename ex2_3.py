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

# Compute derivative of <(X_M(t))^2> with respect to time
dt=10/1000
quad_variance = result.expect[0]
quad_variance_derivative = np.gradient(quad_variance,dt)

# Plot derivative of <(X_M(t))^2> with respect to time
plt.plot(t, quad_variance_derivative)
plt.xlabel('Time')
plt.ylabel('d<(X_M(t))^2>/dt')
plt.title('Rate of Mechanical Position Quadrature Variance')
plt.show()
