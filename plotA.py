import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import argparse
import toml

def plotGreen(U, T, v_0, mu, wC, v, tmax, dt, dw, tol, treshold, n_loops, **kwargs):
    wC = 8.0
    t = np.arange(0, tmax, dt)
    Cut = np.pi/dt
    dw = 2*np.pi/Cut
    # dw = dt
    ft = np.arange(0, Cut, dt)
    fw = np.arange(-Cut, Cut, dw)
    w = np.arange(-wC, wC, dw)
        
    spin = 0
    init = 1

    Greensfunct = 'Green_i={}.npz'
    loaded = np.load(Greensfunct.format(init))
    t = loaded['t']
    Green = loaded['Green']
 
    
    Gles = 1j*Green[1, spin, ::-1, len(t)-1]
    Ggtr = - 1j*Green[0, spin, ::-1, len(t)-1]
    
    # Gles = Green[1, spin, ::-1, len(t)-1]
    # Ggtr = np.real(Green[0, spin, ::-1, len(t)-1]) - 1j * np.imag(Green[0, spin, ::-1, len(t)-1])
    # Ggtr = Green[0, spin, ::-1, len(t)-1] 

    N = int(Cut/dt)
    Gadv = np.zeros(len(ft), complex)
    Gadv[0:int(len(t))] = (Gles + np.conj(Ggtr))
    # Gadv[0:int(len(t))] = Gles
    # Gadv[0:int(len(t))] = Ggtr 
    
    fGadv = fftshift(fft(Gadv)) * (dt) / (np.pi)
    a = int((N-len(w))/2)
    b = int((N+len(w))/2)
    
    # plt.plot(t, np.imag(Gles), 'b--', t, np.real(Gles), 'r--')
    # plt.plot(t, np.imag(Ggtr), 'b', t, np.real(Ggtr), 'r')
    plt.plot(w, np.imag(fGadv[a:b]), '--', w, np.real(fGadv[b:a:-1]), '-')
    #plt.legend(loc='best')
    plt.ylabel('A($\omega$)')
    plt.xlabel('$\omega$')
    plt.grid()
    #plt.show()
    plt.savefig('A.pdf')

def main():
    parser = argparse.ArgumentParser(description = "plot Green's function ")
    parser.add_argument("--params",   default = "run.toml")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    plotGreen(**params)

if __name__ == "__main__":
    main()
