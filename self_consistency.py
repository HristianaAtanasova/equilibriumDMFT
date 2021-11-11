from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse 
import toml


def FT_time_to_freq(Green, tmax, dt, wmax, dw):
    """
    Generate a Green's function in frequency space 
    """
    # fft parameters
    Cut_w = np.pi/dt
    dw = dt
    Cut_t = np.pi/dw
    
    ft = np.arange(-Cut_t, Cut_t, dt)
    fw = np.arange(-Cut_w, Cut_w, dw)
    
    N_w = len(fw)
    N_t = len(ft)
    
    w = np.arange(-wmax, wmax, dw)
    w_start = int(N_w/2 - int(wmax/dw))
    w_end = int(N_w/2 + int(wmax/dw))
    
    t = np.arange(-tmax, tmax, dt)
    t_start = int((N_t/2 - int(tmax/dt)))
    t_0 = int(N_t/2)
    t_end = int(N_t/2 + int(tmax/dt))

    G_N = np.zeros((2, 2, N_t), complex)
    G_w = np.zeros((2, 2, len(w)), complex)

    G_N[:, :, t_0:t_end] = Green
    G_N[:, :, t_start:t_0] = np.real(Green[:, :, ::-1]) - 1j*np.imag(Green[:, :, ::-1])

    # plt.plot(t, np.imag(G_N[1, 0, t_start:t_end]), label='D_les')
    # plt.plot(t, np.imag(G_N[0, 0, t_start:t_end]), label='D_gtr')
    # plt.legend()
    # plt.savefig('Green_time.pdf')
    # plt.close()

    for comp in range(2):
        for spin in range(2):
            fG = ifftshift(fft(fftshift(G_N[comp, spin]))) * (dt/2)
            # G_w[comp, spin] = fG[-w_end:-w_start]
            G_w[comp, spin] = fG[w_start:w_end]

    # plt.plot(w, np.real(G_w[1, 0]), '-', w, np.imag(G_w[1, 0]), '--', label = 'Hyb_les')
    # plt.plot(w, np.real(G_w[0, 0]), '-', w, np.imag(G_w[0, 0]), '--', label = 'Hyb_gtr')
    # plt.legend()
    # plt.savefig('Green_freq.pdf')
    # plt.close()

    return G_w


def FT_freq_to_time(Green, tmax, dt, wmax, dw):
    """
    Generate a Green's function in frequency space 
    """
    # fft parameters
    Cut_w = np.pi/dt
    dw = dt
    Cut_t = np.pi/dw
    
    ft = np.arange(-Cut_t, Cut_t, dt)
    fw = np.arange(-Cut_w, Cut_w, dw)
    
    N_w = len(fw)
    N_t = len(ft)
    
    w = np.arange(-wmax, wmax, dw)
    w_start = int(N_w/2 - int(wmax/dw))
    w_end = int(N_w/2 + int(wmax/dw))
    
    t = np.arange(-tmax, tmax, dt)
    t_start = int((N_t/2 - int(tmax/dt)))
    t_0 = int(N_t/2)
    t_end = int(N_t/2 + int(tmax/dt))

    G_N = np.zeros((2, 2, N_w), complex)
    G_t = np.zeros((2, 2, len(t)), complex)  

    G_N[:, :, w_start:w_end] = Green
    for comp in range(2):
        for spin in range(2):
            fG= ifftshift(fft(fftshift(G_N[comp, spin]))) * dw / np.pi
            G_t[comp, spin] = fG[t_start:t_end]

    return G_t[:, :, int(len(t)/2):]


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


if __name__ == "__main__":
    main()


