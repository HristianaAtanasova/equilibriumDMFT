from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime


####################################
''' Impurity solver based on NCA '''
####################################
########## Define functions for the impurity solver ##########

def check(a, treshold):
    for i in np.nditer(a):
        if abs(i) > treshold:
            return True
    return False


def trapezConv(a, b, dt):
    return dt * (fftconvolve(a, b)[:len(a)] - 0.5 * a[:] * b[0] - 0.5 * a[0] * b[:])


def tdiff(D, t1, t2):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def delta(f, i):
    return 1 if f == i else 0


# Self energy for vertex functions
def Vertex(K, DeltaMatrix, V):
    for f in range(4):
        for j in range(4):
            V[f] += K[j] * DeltaMatrix[j, f]


# Integral equation for vertex functions
def Dyson(V, G, K, t):
    dt = t[1] - t[0]
    for f in range(4):
        Conv = np.zeros((len(t), len(t)), complex)
        for t2 in range(len(t)):
            Conv[:, t2] = trapezConv(V[f, :, t2], G[f, :], dt)
        for t1 in range(len(t)):
            K[f, t1, :] += trapezConv(np.conj(G[f, :]), Conv[t1], dt)


########## Impurity solver computes two-times correlation functions Green for a given hybridization Delta and interaction U ##########

def solve(Green, Delta, G_0, U, t, init, treshold):
    dt = t[1] - t[0]
    # Start with computation of NCA propagators
    # set energy states
    epsilon = -U / 2.0
    E = [0, epsilon, epsilon, 2 * epsilon + U]


    # fill in Delta Matrix elements for positive and negative times
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial and final states, in general two times object for the two branches
    # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            DeltaMatrix[0, 1, t1, t2] = tdiff(Delta[1, 0], t1, t2)
            DeltaMatrix[0, 2, t1, t2] = tdiff(Delta[1, 1], t1, t2)

            DeltaMatrix[1, 0, t1, t2] = tdiff(Delta[0, 0], t1, t2)
            DeltaMatrix[1, 3, t1, t2] = tdiff(Delta[1, 1], t1, t2)

            DeltaMatrix[2, 0, t1, t2] = tdiff(Delta[0, 1], t1, t2)
            DeltaMatrix[2, 3, t1, t2] = tdiff(Delta[1, 0], t1, t2)

            DeltaMatrix[3, 1, t1, t2] = tdiff(Delta[0, 1], t1, t2)
            DeltaMatrix[3, 2, t1, t2] = tdiff(Delta[0, 0], t1, t2)


    # Initialize one branch propagators
    G = np.zeros((4, len(t)), complex)  # indices are initial state, propagation time
    G_old = np.zeros((4, len(t)), complex)


    # Computation of bare propagators G_0
    for i in range(4):
        G_0[i] = (np.exp(-1j * E[i] * t))

        # plt.plot(t, np.real(G_0[i]), 'r--', t, np.imag(G_0[i]), 'b--')
        # plt.show()

    # Perform self consistent iteration to obtain bold propagators G
    G[:] = G_0
    while check(G - G_old, treshold):
        Sigma = np.sum(G[None] * DeltaMatrix[:, :, 0, :], 1)    # propagators for one branch only, so Delta needs only one time

        G_old[:] = G
        G[:] = G_0
        for i in range(4):
            Conv = trapezConv(G_0[i], Sigma[i], dt)
            G[i] -= trapezConv(Conv, G_old[i], dt)

    # for i in range(4):
        # plt.plot(t, np.real(G[i]), 'r--', t, np.imag(G[i]), 'b--')
        # plt.show()

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    # Initialization of Vertex functions
    K = np.zeros((4, 4, len(t), len(t)), complex)   # indices are initial | final states | upper/lower branch time
    K_0 = np.zeros((4, 4, len(t), len(t)), complex)

    # Computation of K_0
    # for i in range(4):
    i = init
    for f in range(4):
        K_0[i, f] = delta(i, f) * np.conj(G[i, None, :]) * G[i, :, None]

    # plt.plot(t, np.real(K_0[i, i, :, len(t)-1]), 'r--', t, np.imag(K_0[i, i, :, len(t)-1]), 'r--')
    # plt.plot(t, np.real(K_0[i, i, len(t) - 1]), 'b--', t, np.imag(K_0[i, i, len(t) - 1]), 'b--')
    # plt.show()

    # Perform self consistent iteration
    K[i] = K_0[i]
    K_old = np.zeros((4, len(t), len(t)), complex)
    counter = 0
    while check(K[i] - K_old, treshold):
        counter += 1
        V = np.zeros((4, len(t), len(t)), complex)
        Vertex(K[i], DeltaMatrix, V)
        K_old[:] = K[i]
        K[i] = K_0[i]
        Dyson(V, G, K[i], t)
        # K_old = K[i]
    print('')
    print('Finished calculation of K for initial state', i)
    err = np.abs(1 - np.abs(np.sum(K[i, :, len(t) - 1, len(t) - 1], 0)))
    print('Error for inital state =', i, 'is', err)
    print('')

    # plt.plot(t, np.real(K[1, 1, :, len(t)-1]), 'r--', t, np.imag(K[1, 1, :, len(t)-1]), 'b--')
    # plt.plot(t, np.real(K[1, 1, len(t)-1]), 'y--', t, np.imag(K[1, 1, len(t)-1]), 'k--')
    # plt.plot(t, np.real(K[1, 0].diagonal()), 'b--', t, np.real(K[1, 1].diagonal()), 'r', t, np.real(K[1, 2].diagonal()), 'g--', t, np.real(K[1, 3].diagonal()), 'b--')
    # plt.grid()
    # plt.show()

    ########## Computation of two-times Green's functions ##########
    for t1 in range(len(t)):
        for t_1 in range(t1+1):

            Green[0, 0, i, t_1, t1] = K[i, 0, t_1, t1] * G[1, (t1-t_1)] + K[i, 2, t_1, t1] * G[3, (t1-t_1)]
            Green[1, 0, i, t_1, t1] = K[i, 1, t_1, t1] * G[0, (t1-t_1)] + K[i, 3, t_1, t1] * G[2, (t1-t_1)]
            Green[0, 1, i, t_1, t1] = K[i, 0, t_1, t1] * G[2, (t1-t_1)] + K[i, 1, t_1, t1] * G[3, (t1-t_1)]
            Green[1, 1, i, t_1, t1] = K[i, 2, t_1, t1] * G[0, (t1-t_1)] + K[i, 3, t_1, t1] * G[1, (t1-t_1)]

    # output
    Vertexfunction = 'K_i={}'
    Greensfunction = 'Green_i={}'
    np.savez_compressed(Vertexfunction.format(init), t=t, K=K[i])
    np.savez_compressed(Greensfunction.format(init), t=t, Green=Green[:,:,i])

    # plt.plot(t, np.real(Green[1,0,1, :, len(t)-1]), 'y--', t, np.imag(Green[1, 0, 1, :, len(t)-1]), 'k--')
    # plt.plot(t, np.real(Green[0,1,1, :, len(t)-1]), 'r--', t, np.imag(Green[0, 1, 1, :, len(t)-1]), 'b--')
    # plt.grid()
    # plt.show()

    #print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t) - 1, len(t) - 1] + Green[1, 0, i, len(t) - 1, len(t) - 1]))
    #print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t) - 1, len(t) - 1] + Green[1, 1, i, len(t) - 1, len(t) - 1]))


    #print('Population for Spin Up les on site', i, 'is', Green[1, 0, i, len(t) - 1, len(t) - 1])
    #print('Population for Spin Down les on site', i, 'is', Green[1, 1, i, len(t) - 1, len(t) - 1])
    #print('')
    print('Population for Spin Up gtr on site', i, 'is', np.real(Green[0, 0, i, len(t) - 1, len(t) - 1]))
    print('Population for Spin Down gtr on site', i, 'is', np.real(Green[0, 1, i, len(t) - 1, len(t) - 1]))

    return Green

def main():
    parser = argparse.ArgumentParser(description = "run nca impurity solver")
    parser.add_argument("--U",    type=float, default = 2.0)
    # parser.add_argument("--hybfile",     default = "output.h5")
    # parser.add_argument("--hybsection",  default = "hybridization/hyb")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
                  
