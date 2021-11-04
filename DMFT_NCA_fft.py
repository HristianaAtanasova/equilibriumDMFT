from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime

### This is the same code as DMFT_NCA_fft.py with different order of time arguments in K ###
# set parameters
T = 1
beta = 1/T
mu = 0
wC = 10
t_param = 1
v = 10
treshold = 1e-6
tmax = 2
dt = 0.01
t = np.arange(0, tmax, dt)


########################################################################################################################
''' Calculate initial time-domain Hybridization function for given Density of States '''

dw = 0.01
wDOS = np.arange(-2*t_param, 2*t_param, dw)
Cut = np.pi/dt
w = np.arange(-Cut, Cut, dw)
fft_tmax = np.pi/dw
fft_tmin = -np.pi/dw
fft_dt = np.pi/Cut
fft_time = np.arange(fft_tmin, fft_tmax, fft_dt)

# fermi function
def fermi_function(w):
    return 1/(1+np.exp(beta*(w-mu)))

# flat band with soft cutoff
def A(w):
    return 1/((np.exp(v*(w-wC)) + 1) * (np.exp(-v*(w+wC)) + 1))

# semicircular density of states for bethe lattice
def semicircularDos(w):
    return 1/(2 * np.pi * t_param**2) * np.sqrt(4*t_param**2 - w**2)

# window function padded with zeros for semicircular DOS
N = int(2*Cut/dw)
a = int(N/2 + 2*t_param/dw)
b = int(N/2 - 2*t_param/dw)
DOS = np.zeros(N+1)
DOS[b:a] = semicircularDos(wDOS)

# frequency-domain Hybridization function
# Hyb_les = DOS * fermi_function(w)
# Hyb_gtr = DOS * (1 - fermi_function(w))

Hyb_les = A(w) * fermi_function(w)
Hyb_gtr = A(w) * (1 - fermi_function(w))

# ontain time-domain Hybridization function with fft
fDelta_les = np.conj((ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi)
fDelta_gtr = (ifftshift(fft(fftshift(Hyb_gtr)))) * dw/np.pi

# get real times from fft_times
Delta = np.zeros((2, 2, len(t)), complex)  # gtr/les | spin up/spin down
for t_ in range(len(t)):
    # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
    # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
    Delta[0, :, t_] = fDelta_gtr[int(N/2) + t_]
    Delta[1, :, t_] = fDelta_les[int(N/2) + t_]


########################################################################################################################
''' Impurity solver based on NCA '''

########## Define functions for the impurity solver ##########

def check(a):
    for i in np.nditer(a):
        if abs(i) > treshold:
            return True
    return False


def trapezConv(a, b):
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
def Dyson(V, G, K):
    for f in range(4):
        Conv = np.zeros((len(t), len(t)), complex)
        for t2 in range(len(t)):
            Conv[:, t2] = trapezConv(V[f, :, t2], G[f, :])
        for t1 in range(len(t)):
            K[f, t1, :] += trapezConv(np.conj(G[f, :]), Conv[t1])


########## Impurity solver computes two-times correlation functions Green for a given hybridization Delta and interaction U ##########

def Solver(Delta, U, init):

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
    G_0 = np.zeros((4, len(t)), complex)
    G = np.zeros((4, len(t)), complex)  # indices are initial state, propagation time
    G_old = np.zeros((4, len(t)), complex)


    # Computation of bare propagators G_0
    for i in range(4):
        G_0[i] = (np.exp(-1j * E[i] * t))

        # plt.plot(t, np.real(G_0[i]), 'r--', t, np.imag(G_0[i]), 'b--')
        # plt.show()

    # Perform self consistent iteration to obtain bold propagators G
    G[:] = G_0
    while check(G - G_old):
        Sigma = np.sum(G[None] * DeltaMatrix[:, :, 0, :], 1)    # propagators for one branch only, so Delta needs only one time

        G_old[:] = G
        G[:] = G_0
        for i in range(4):
            Conv = trapezConv(G_0[i], Sigma[i])
            G[i] -= trapezConv(Conv, G_old[i])

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
    while check(K[i] - K_old):
        counter += 1
        V = np.zeros((4, len(t), len(t)), complex)
        Vertex(K[i], DeltaMatrix, V)
        K_old[:] = K[i]
        K[i] = K_0[i]
        Dyson(V, G, K[i])
        # K_old = K[i]

    print('Finished calculation of K for initial state', i)
    err = np.abs(1 - np.abs(np.sum(K[i, :, len(t) - 1, len(t) - 1], 0)))
    print('Error for inital state =', i, 'is', err)
    print('                                                                                                                               ')

    # output
    file = 'Kfft_t={}_dt={}_i={}_f={}.out'
    for f in range(4):
        np.savetxt(file.format(tmax,dt,i,f), K[i, f].view(float), delimiter=' ')

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
    gtr_up = "Green_gtr_spinUp_U={}_T={}_t={}_i={}.out"
    les_up = "Green_les_spinUp_U={}_T={}_t={}_i={}.out"
    gtr_down = "Green_gtr_spinDown_U={}_T={}_t={}_i={}.out"
    les_down = "Green_les_spinDown_U={}_T={}_t={}_i={}.out"

    np.savetxt(les_up.format(U, T, tmax, i), Green[1, 0, i].view(float), delimiter=' ')
    np.savetxt(gtr_up.format(U, T, tmax, i), Green[0, 0, i].view(float), delimiter=' ')
    np.savetxt(les_down.format(U, T, tmax, i), Green[1, 1, i].view(float), delimiter=' ')
    np.savetxt(gtr_down.format(U, T, tmax, i), Green[0, 1, i].view(float), delimiter=' ')

    # plt.plot(t, np.real(Green[1,0,1, :, len(t)-1]), 'y--', t, np.imag(Green[1, 0, 1, :, len(t)-1]), 'k--')
    # plt.plot(t, np.real(Green[0,1,1, :, len(t)-1]), 'r--', t, np.imag(Green[0, 1, 1, :, len(t)-1]), 'b--')
    # plt.grid()
    # plt.show()

    print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t) - 1, len(t) - 1] + Green[1, 0, i, len(t) - 1, len(t) - 1]))
    print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t) - 1, len(t) - 1] + Green[1, 1, i, len(t) - 1, len(t) - 1]))

    print('                                                                                                                               ')

    print('Population for Spin Up les on site', i, 'is', Green[1, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down les on site', i, 'is', Green[1, 1, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Up gtr on site', i, 'is', Green[0, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down gtr on site', i, 'is', Green[0, 1, i, len(t) - 1, len(t) - 1])

    print('                                                                                                                               ')

########################################################################################################################
''' Main part starts here '''
n_loops = 10
Umax = 11
Umin = 10
init = 1   # chose initial state

######### perform loop over U #########

for U in np.arange(Umin, Umax, 2):
    print('Starting DMFT loop for U =', U, 'Temperature =', T, 'time = ', tmax)
    start = datetime.now()

    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  # gtr/les | spin up/spin down | initial state | time on upper/lower branch
    Green_old = np.zeros((2, 2, 4, len(t), len(t)), complex)
    Solver(Delta, U, 1)  # initial guess for the first DMFT loop
    Solver(Delta, U, 2)  # initial guess for the first DMFT loop
    Delta = np.zeros((2, 2, 4, len(t)), complex)  # gtr/les | spin up/spin down

    counter = 0
    while np.amax(np.abs(Green_old - Green)) > 0.001:
        start = datetime.now()
        counter += 1

        Delta[:, :, :] = t_param ** 2 * Green[:, :, :, ::-1, len(t) - 1]

        # average over initial states
        # Delta[:, :] = t_param**2 * np.sum(Green[:, :, :, ::-1, len(t)-1], 2)/4

        Green_old[:] = Green
        Solver(Delta[:,:,1], U, 2)
        Solver(Delta[:,:,2], U, 1)
        Diff = np.amax(np.abs(Green_old - Green))
        print('for U = ', U, ' and iteration Nr. ', counter, ' the Difference is ', Diff, ' after a calculation time ', datetime.now() - start)

print('Computation of Greens functions for U = ', U, 'Temperature = ', T, 'time = ', tmax, 'finished after', counter, 'iterations and', datetime.now() - start, 'seconds.')
