import numpy as np
import matplotlib.pyplot as plt

# Computation of bare propagators G_0
def bare_prop(t, U):
    # set energy states
    epsilon = -U / 2.0
    E = [0, epsilon, epsilon, 2 * epsilon + U]

    G_0 = np.zeros((4, len(t)), complex)
    for i in range(4):
        G_0[i] = (np.exp(-1j * E[i] * t))

    np.savez_compressed('barePropagators', t=t, G_0=G_0)

