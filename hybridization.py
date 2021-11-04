from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

# fermi function
def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))

# flat band with soft cutoff
def wideBandDos(w, wC, v):
    """
    DOS for a flat band with soft cutoff
    """
    return 1 / ((np.exp(v * (w-wC)) + 1) * (np.exp(-v * (w + wC)) + 1))

# semicircular density of states for bethe lattice
def semicircularDos(w, v_0):
    """
    DOS for Bethe Lattice
    """
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 **2 - w ** 2)
 
def genSemicircularHyb(T, mu, v_0, tmax, dt, dw):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    Cut = np.pi/dt

    t = np.arange(0, tmax, dt)
    wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
    w = np.arange(-Cut, Cut, dw)

    # window function padded with zeros for semicircular DOS
    N = int(2*Cut/dw)
    a = int(N/2 + 2*v_0/dw)
    b = int(N/2 - 2*v_0/dw)
    DOS = np.zeros(N+1)
    DOS[b:a] = semicircularDos(wDOS, v_0)

    # frequency-domain Hybridization function
    Hyb_les = DOS * fermi_function(w, beta, mu)
    Hyb_gtr = DOS * (1 - fermi_function(w, beta, mu))
    
    # obtain time-domain Hybridization function with fft
    fDelta_les = np.conj((ifftshift(fft(fftshift(Hyb_les)))) * dw / np.pi)
    fDelta_gtr = (ifftshift(fft(fftshift(Hyb_gtr)))) * dw / np.pi
    
    # get real times from fft_times
    Delta = np.zeros((2, 2, 4, len(t)), complex)  # gtr/les | spin up/spin down
    for t_ in range(len(t)):
        # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
        # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
        Delta[0, :, :, t_] = fDelta_gtr[int(N/2) + t_]
        Delta[1, :, :, t_] = fDelta_les[int(N/2) + t_]
    
    np.savez_compressed('Delta', t=t, D=Delta)

def genWideBandHyb(T, mu, tmax, dt, dw, wC, v):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    Cut = np.pi/dt

    t = np.arange(0, tmax, dt)
    w = np.arange(-Cut, Cut, dw)
    N = int(2*Cut/dw)

    # frequency-domain Hybridization function
    Hyb_les = wideBandDos(w, wC, v) * fermi_function(w, beta, mu)
    Hyb_gtr = wideBandDos(w, wC, v) * (1 - fermi_function(w, beta, mu))
    
    # obtain time-domain Hybridization function with fft
    fDelta_les = np.conj((ifftshift(fft(fftshift(Hyb_les)))) * dw / np.pi)
    fDelta_gtr = (ifftshift(fft(fftshift(Hyb_gtr)))) * dw / np.pi
    
    # get real times from fft_times
    Delta = np.zeros((2, 2, 4, len(t)), complex)  # gtr/les | spin up/spin down
    for t_ in range(len(t)):
        # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
        # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
        Delta[0, :, :, t_] = fDelta_gtr[int(N/2) + t_]
        Delta[1, :, :, t_] = fDelta_les[int(N/2) + t_]
    
    np.savez_compressed('Delta', t=t, D=Delta)
