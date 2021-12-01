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
import k_greensfunction
import self_consistency

def run_dmft(U, T, v_0, mu, wC, v, tmax, dt, dw, tol, treshold, n_loops, output, **kwargs):
    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw) 

    # gtr/les | spin up/spin down | initial state | time on upper/lower branch
    Green = np.zeros((2, 2, len(t), len(t)), complex)  
    Green_w = np.zeros((2, 2, len(w)), complex)  
    Green_local = np.zeros((2, len(w)), complex)  
    Green_dot = np.zeros((2, 2, len(w)), complex)  
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t)), complex)  
    Delta_old = np.zeros((2, 2, len(t)), complex)  
    Delta_w = np.zeros((2, 2, len(w)), complex)  
    Sigma_w = np.zeros((2, len(w)), complex)

    # calculate and load bare propagators
    bareprop.bare_prop(t, U)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    k_greensfunction.gen_k_dep_Green(T, v_0, U, mu, tmax, dt, wC, dw)
    loaded = np.load('Green_k.npz')
    Green_d = loaded['G']

    # Green_dot_w = self_consistency.FT_time_to_freq(Green_d[:, :, ::-1, len(t)-1], tmax, dt, wC, dw)
    # Green_dot = Green_dot_w

    # plt.plot(w, np.real(Green_dot[0, 0]), '-', w, np.imag(Green_dot[0, 0]), '--', label = 'Green_dot_gtr')  
    # plt.plot(w, np.real(Green_dot[1, 0]), '-', w, np.imag(Green_dot[1, 0]), '--', label = 'Green_dot_les')  
    # plt.legend()
    # plt.savefig('Green_dot.pdf')
    # plt.close()      

    hybridization.genGaussianHyb(T, mu, v_0, tmax, dt, wC, dw)
    # hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, wC, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw, wC, v)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']
    dos = loaded['dos']

    plt.plot(w, np.real(dos), '-', w, np.imag(dos), '--')
    plt.savefig('noninteracting_dos.pdf')
    plt.close()

    start = datetime.now()
    msg = 'Starting DMFT loop for U = {} | Temperature = {} | time = {} | dt = {}  '.format(U, T, tmax, dt)
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
        Delta_old[:] = Delta

        Green = nca.solve(Green, Delta, G_0, U, t, 1, treshold)

        # plt.plot(t, np.real(Delta[0, 0]), '-', t, np.imag(Delta[0, 0]), '--', label = 'Delta_gtr')  
        # plt.plot(t, np.real(Delta[1, 0]), '-', t, np.imag(Delta[1, 0]), '--', label = 'Delta_les')  
        # plt.legend()
        # plt.savefig('Delta_t.pdf')
        # plt.close()      

        # plt.plot(t, np.real(Green[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[0, 0, ::-1, len(t)-1]), '--', label = 'Green_gtr')
        # plt.plot(t, np.real(Green[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[1, 0, ::-1, len(t)-1]), '--', label = 'Green_les')
        # plt.legend()
        # plt.savefig('Greens_t.pdf')
        # plt.close()

        # transform Greens function from time to frequancy domain and extract impurity self-energy 
        Green_w = self_consistency.FT_time_to_freq(Green[:, :, ::-1, len(t)-1], tmax, dt, wC, dw)
        Delta_w = self_consistency.FT_time_to_freq(Delta, tmax, dt, wC, dw)

        # plt.plot(w, np.real(Green_w[0, 0]), '-', w, np.imag(Green_w[0, 0]), '--', label = 'Green_w_gtr')  
        # plt.plot(w, np.real(Green_w[1, 0]), '-', w, np.imag(Green_w[1, 0]), '--', label = 'Green_w_les')  
        # plt.legend()
        # plt.savefig('Green_w.pdf')
        # plt.close()      

        # plt.plot(w, np.real(Delta_w[0, 0]), '-', w, np.imag(Delta_w[0, 0]), '--', label = 'Delta_w_gtr')  
        # plt.plot(w, np.real(Delta_w[1, 0]), '-', w, np.imag(Delta_w[1, 0]), '--', label = 'Delta_w_les')  
        # plt.legend()
        # plt.savefig('Delta_w.pdf')
        # plt.close()      
        
        Delta_ret = Delta_w[0] + Delta_w[1]
        Green_ret = Green_w[0] + Green_w[1,:, ::-1]

        plt.plot(w, np.real(Delta_ret[0]), '-', w, np.imag(Delta_ret[ 0]), '--', label = 'Delta_ret')  
        plt.legend()
        plt.savefig('Delta_ret.pdf')
        plt.close()      

        plt.plot(w, np.real(Green_ret[0]), '-', w, np.imag(Green_ret[ 0]), '--', label = 'Green_ret')  
        plt.legend()
        plt.savefig('Green_ret.pdf')
        plt.close()      

        for spin in range(2):
            for w_ in range(len(w)):
                Sigma_w[spin, w_] = w[w_] + mu - Delta_ret[spin, w_] - (1.0 / Green_ret[spin, w_])
                # Sigma_w[comp, spin, w_] = 1.0 / Green_dot[comp, spin, w_] - Delta_w[comp, spin, w_] - (1.0 / Green_w[comp, spin, w_])
                # Sigma_w[comp, spin, w_] = - Delta_w[comp, spin, w_] - (1 / Green_w[comp, spin, w_])

        # sum over k and extract hybridization from mdyson equation  
        for spin in range(2):
            for w_ in range(len(w)):
                # I = dos / (w[w_] + mu - Sigma_w[comp, spin, w_] - w)
                I = dos / (Delta_ret[spin, w_] + (1.0 / Green_ret[spin, w_]) - w)
                Green_local[spin, w_] = np.trapz(I, x=w) 
                Delta_ret[spin, w_] = w[w_] + mu - (1.0 / Green_local[spin, w_]) - Sigma_w[spin, w_]
                # Delta_w[comp, spin, w_] = Green_dot[comp, spin, w_] - (1 / Green_w[comp, spin, w_]) - Sigma_w[comp, spin, w_]
                # Delta_w[comp, spin, w_] = - (1 / Green_w[comp, spin, w_]) - Sigma_w[comp, spin, w_]

        # convert Delta from frequency to time domain 
        Delta_w[0] = (1 - hybridization.fermi_function(w, 1.0/T, mu)) * Delta_ret
        Delta_w[1] = hybridization.fermi_function(w, 1.0/T, mu) * Delta_ret
        Delta = self_consistency.FT_freq_to_time(Delta_w, tmax, dt, wC, dw)

        # plt.plot(w, np.real(Sigma_w[0]), '-', w, np.imag(Sigma_w[0]), '--', label = 'Sigma_w')  
        # plt.legend()
        # plt.savefig('Sigma_w.pdf')
        # plt.close()      

        # plt.plot(w, np.real(Green_local[0]), '-', w, np.imag(Green_local[0]), '--', label = 'Green_local')  
        # plt.legend()
        # plt.savefig('Green_local.pdf')
        # plt.close()      

        # plt.plot(t, np.real(Delta[0, 0]), '-', t, np.imag(Delta[0, 0]), '--', label = 'Delta_gtr')  
        # plt.plot(t, np.real(Delta[1, 0]), '-', t, np.imag(Delta[1, 0]), '--', label = 'Delta_les')  
        # plt.legend()
        # plt.savefig('Delta_new.pdf')
        # plt.close()      
    
        # Delta = (Delta + Delta_old) / 2

        Delta[0] = Delta[1]
        Delta[1] = Delta[0]
 
        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0
        
        print('')
        msg = 'for iteration {} the difference is {} after a calculation time {}'.format(counter, diff, datetime.now() - start_innerloop)
        print(msg)
        print('-'*len(msg))
        msg = 'Computation of Greens functions for U = {}, temperature = {}, time = {} and dt = {} finished after {} iterations and {} seconds.'.format(U, T, tmax, dt, counter, datetime.now()-start)
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

