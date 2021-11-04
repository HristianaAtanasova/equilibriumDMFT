from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
import argparse
import toml 

import nca 
import bareprop
import hybridization 

def run_dmft(U, T, v_0, mu, wC, v, tmax, dt, dw, tol, treshold, n_loops, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    # gtr/les | spin up/spin down | initial state | time on upper/lower branch
    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  
    Green_old = np.zeros((2, 2, 4, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, 4, len(t)), complex)  

    # calculate and load bare propagators
    bareprop.bare_prop(t, U)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
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
        start = datetime.now()
        counter += 1

        Delta[:, :, :] = v_0 ** 2 * Green[:, :, :, ::-1, len(t) - 1]

        Green_old[:] = Green
        Green = nca.solve(Green, Delta[:,:,1], G_0, U, t, 2, treshold)
        Green = nca.solve(Green, Delta[:,:,2], G_0, U, t, 1, treshold)
        diff = np.amax(np.abs(Green_old - Green))
        
        print('')
        msg = 'for iteration {} the difference is {} after a calculation time {}'.format(counter, diff, datetime.now() - start)
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

