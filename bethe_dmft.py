from scipy.signal import fftconvolve
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
import argparse
import toml 

import nca 
import bareprop
import hybridization 
import self_consistency

def run_dmft(U, T, v_0, mu, wC, v, tmax, dt, dw, tol, treshold, n_loops, output, **kwargs):
    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw) 

    # gtr/les | spin up/spin down | initial state | time on upper/lower branch
    Green = np.zeros((2, 2, len(t), len(t)), complex)  
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t)), complex)  
    Delta_freq = np.zeros((2, 2, len(w)), complex)  

    # calculate and load bare propagators
    bareprop.bare_prop(t, U)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, wC, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw, wC, v)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']

    start = datetime.now()
    msg = 'Starting DMFT loop for U = {} | Temperature = {} | time = {}'.format(U, T, tmax)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    counter = 0
    diff = np.inf
    iteration = 0 
    while diff > tol:
        start_innerloop = datetime.now()
        counter += 1

        msg = 'starting iteration {}'.format(counter)
        print(msg)
        print('-'*len(msg))

        Green_old[:] = Green

        Green = nca.solve(Green, Delta, G_0, U, t, 1, treshold)

        plt.plot(t, np.real(Green[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[0, 0, ::-1, len(t)-1]), '--', label = 'Green_gtr')
        plt.plot(t, np.real(Green[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[1, 0, ::-1, len(t)-1]), '--', label = 'Green_les')
        plt.legend()
        plt.savefig('Greens_t.pdf')
        plt.close()

        Green_freq = self_consistency.FT_time_to_freq(Green[:, :, ::-1, len(t)-1], tmax, dt, wC, dw)
        
        Delta_freq[0] = Green_freq[1]
        Delta_freq[1] = Green_freq[0]

        # Delta_freq = Green_freq

        Delta = self_consistency.FT_freq_to_time(Delta_freq, tmax, dt, wC, dw)

        # plt.plot(w, np.real(Green_freq[0, 0]), '-', w, np.imag(Green_freq[0, 0]), '--', label = 'Green_w_gtr')
        # plt.plot(w, np.real(Green_freq[1, 0]), '-', w, np.imag(Green_freq[1, 0]), '--', label = 'Green_w_les')
        # plt.legend()
        # plt.savefig('Greens_w.pdf')
        # plt.close()
 
        plt.plot(t, np.real(Delta[0, 0]), '-', t, np.imag(Delta[0, 0]), '--', label = 'Delta_gtr')  
        plt.plot(t, np.real(Delta[1, 0]), '-', t, np.imag(Delta[1, 0]), '--', label = 'Delta_les')  
        plt.legend()
        plt.savefig('Delta_t.pdf')
        plt.close()      

        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0
        
        print('')
        msg = 'for iteration {} the difference is {} after a calculation time {}'.format(counter, diff, datetime.now() - start_innerloop)
        print(msg)
        print('-'*len(msg))
        msg = 'Computation of Greens functions for U = {}, Temperature = {} and time = {} finished after {} iterations and {} seconds.'.format(U, T, tmax, counter, datetime.now()-start)
        print('-'*len(msg))
        print(msg)
        print('-'*len(msg))

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

    Green = run_dmft(**params)

if __name__ == "__main__":
    main()

