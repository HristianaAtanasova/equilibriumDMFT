from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse 
import toml

def tdiff(D, t1, t2):
    return D[:, :, t2 - t1] if t2 >= t1 else np.conj(D[:, :, t1 - t2])

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
 
def genSemicircularHyb(T, mu, v_0, tmax, dt, wmax, dw):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    Cut = np.pi / dt

    t = np.arange(0, tmax, dt)
    w = np.arange(-2 * v_0, 2 * v_0, dw)
    fw = np.arange(-Cut, Cut, dw)

    # window function padded with zeros for semicircular DOS
    N = int(2 * Cut / dw)
    a = int(N/2 + 2 * v_0 / dw)
    b = int(N/2 - 2 * v_0 / dw)
    dos = np.zeros(N + 1)
    dos[b:a] = semicircularDos(w, v_0)

    # frequency-domain Hybridization function
    Hyb_les = dos * fermi_function(fw, beta, mu)
    Hyb_gtr = dos * (1 - fermi_function(fw, beta, mu))
    
    # obtain time-domain Hybridization function with fft
    fDelta_les = np.conj((ifftshift(fft(fftshift(Hyb_les)))) * dw / np.pi)
    # fDelta_les = ifftshift(fft(fftshift(Hyb_les))) * dw / (np.pi)
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw / (np.pi)
    
    # get real times from fft_times
    Delta = np.zeros((2, 2, len(t)), complex)  # gtr/les | spin up/spin down
    for t_ in range(len(t)):
        # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
        # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
        Delta[0, :, t_] = fDelta_gtr[int(N/2) + t_]
        Delta[1, :, t_] = fDelta_les[int(N/2) + t_]

    # spin = 0 
    # plt.plot(w, Hyb_les[b:a], label = 'Hyb_les')
    # plt.plot(w, Hyb_gtr[b:a], label = 'Hyb_gtr')
    # plt.legend()
    # plt.savefig('Hyb_w.pdf')
    # plt.close()
    # plt.plot(t, np.real(Delta[1, spin]), '-', t, np.imag(Delta[1, spin]), '--', label = 'Delta_les')
    # plt.plot(t, np.real(Delta[0, spin]), '-', t, np.imag(Delta[0, spin]), '--', label = 'Delta_gtr')
    # plt.legend()
    # plt.savefig('Hyb_t.pdf')
    # plt.close()

    # # fft parameters
    # Cut_w = np.pi/dt
    # dw = dt
    # Cut_t = np.pi/dw
    # 
    # ft = np.arange(-Cut_t, Cut_t, dt)
    # fw = np.arange(-Cut_w, Cut_w, dw)
    # 
    # N_w = len(fw)
    # N_t = len(ft)
    # 
    # w = np.arange(-wmax, wmax, dw)
    # w_start = int(N_w/2 - int(wmax/dw))
    # w_end = int(N_w/2 + int(wmax/dw))
    # 
    # t = np.arange(-tmax, tmax, dt)
    # t_start = int((N_t/2 - int(tmax/dt)))
    # t_0 = int(N_t/2)
    # t_end = int(N_t/2 + int(tmax/dt))

    # D = np.zeros((2, N_t), complex)
    # wD = np.zeros((2, len(w)), complex)

    # D[:, t_0:t_end] = Delta[:, spin]
    # D[:, t_start:t_0] = np.real(Delta[:, spin, ::-1]) - 1j*np.imag(Delta[:, spin, ::-1])

    # plt.plot(t, np.imag(D[1,t_start:t_end]), label='D_les')
    # plt.plot(t, np.imag(D[0,t_start:t_end]), label='D_gtr')
    # plt.legend()
    # plt.savefig('D.pdf')
    # plt.close()

    # for comp in range(2):
    #     fD = ifftshift(fft(fftshift(D[comp]))) * (dt/2)
    #     # wD[comp] = fD[-w_end:-w_start]
    #     wD[comp] = fD[w_start:w_end]

    # plt.plot(w, np.real(wD[1]), '-', w, np.imag(wD[1]), '--', label = 'Hyb_les')
    # plt.plot(w, np.real(wD[0]), '-', w, np.imag(wD[0]), '--', label = 'Hyb_gtr')
    # plt.legend()
    # plt.savefig('Hyb_w_2.pdf')
    # plt.close()

    # dos = np.zeros((2, N_w), complex)
    # dos[:, w_start:w_end] = wD
    # 
    # Delta_new = np.zeros((2, len(t)), complex)  # gtr/les | spin up/spin down
    # for comp in range(2):
    #     fDelta = ifftshift(fft(fftshift(dos[comp]))) * dw / np.pi
    #     Delta_new[comp] = np.real(fDelta[t_start:t_end]) - 1j * np.imag(fDelta[t_start:t_end]) 

    # plt.plot(t[int(len(t)/2):], np.real(Delta_new[0, int(len(t)/2):]), '-', t[int(len(t)/2):], np.imag(Delta_new[0, int(len(t)/2):]), '--', label = 'Delta_gtr')
    # plt.plot(t[int(len(t)/2):], np.real(Delta_new[1, int(len(t)/2):]), '-', t[int(len(t)/2):], np.imag(Delta_new[1, int(len(t)/2):]), '--', label = 'Delta_les')
    # plt.legend()
    # plt.savefig('Hyb_t_2.pdf')
    # plt.close()

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
    Delta = np.zeros((2, 2, len(t)), complex)  # gtr/les | spin up/spin down
    for t_ in range(len(t)):
        # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
        # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
        Delta[0, :, t_] = fDelta_gtr[int(N/2) + t_]
        Delta[1, :, t_] = fDelta_les[int(N/2) + t_]
    
    np.savez_compressed('Delta', t=t, D=Delta)

def main():
    parser = argparse.ArgumentParser(description = "run dmft")
    # parser.add_argument("--output",   default = "output.h5")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    genSemicircularHyb(params['T'], params['mu'], params['v_0'], params['tmax'], params['dt'], params['wC'], params['dw'])

if __name__ == "__main__":
    main()


