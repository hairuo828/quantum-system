import numpy as np
from qutip import basis, destroy

# Define the dimension of the Hilbert space
dim = 10

# Create a single-quanta Fock state in a 10-dimensional Hilbert space
fock_state = basis(dim, 1)

# Define the lowering operator
lowering_op = destroy(dim)

# Apply the lowering operator to the Fock state
ground_state = lowering_op * fock_state

print("Ground State Vector:")
print(ground_state)

