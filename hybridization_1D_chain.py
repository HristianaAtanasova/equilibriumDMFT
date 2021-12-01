#!/usr/bin/env python
# Tight binding for a 1D noninteracting bath coupled to 1D chain.
from general import *
from scipy.signal import hilbert


def calculate_green_function(H, omega):
    dim = H.shape[0]
    nomega = omega.shape[0]

    # Diagonalize Hamiltonian:
    E, T = np.linalg.eigh(H)

    # Calculate GF:
    gf_k_w = np.zeros((dim, dim, nomega), complex)
    for k in range(dim):
        xi_k = E[k]
        gf_k_w[k, k,:] = 1.0 / (omega - xi_k + 1j * p.eta)
    gf_ij_w = np.swapaxes(np.tensordot(np.tensordot((T), gf_k_w, ([1], [0])), dagger(T), ([1], [0])), 1, 2)
    return gf_ij_w, E, T


def inverse_calculation(**pdict):
    p = Struct(**pdict)
    res = {}
    omega = np.linspace(-p.omega_c, p.omega_c, p.n_omega)

    # Build Hamiltonians:
    Himp = np.zeros((1, 1))
    Himp[0, 0] = p.epsilon

    H = np.zeros((p.n_bath + 1, p.n_bath + 1))
    H[0, 0] = p.epsilon
    H[0, 1] = H[1, 0] =  p.td
    for i in range(1, p.n_bath):
        H[i, i] = p.epsilon
        H[i, i + 1] = H[i + 1, i] = p.tb

    # Evaluate gfs:
    gf_ij_w_imp, Eimp, Timp = calculate_green_function(Himp, omega)
    gf_ij_w, E, T = calculate_green_function(H, omega)

    # Calculate Hybridization self energy:
    Delta_ij_w = np.zeros((1, 1, p.n_omega), complex)
    Delta_ij_w[0, 0, :] = (1.0 / (gf_ij_w_imp[0,0,:]) - 1.0 / (gf_ij_w[0,0,:]))
    print(Delta_ij_w[0, 0, :])

    res['omega'] = omega
    res['gf_ij_w_imp'] = gf_ij_w_imp
    res['Delta_ij_w'] = Delta_ij_w
    res['gf_ij_w'] = gf_ij_w

    return res


def block_calculation(**pdict):
    p = Struct(**pdict)
    res = {}
    omega = np.linspace(-p.omega_c, p.omega_c, p.n_omega)

    # Build Hamiltonians:
    Himp = np.zeros((1, 1))
    Himp[0, 0] = p.epsilon

    Hbath = np.zeros((p.n_bath, p.n_bath))
    for i in range(0, p.n_bath - 1):
        Hbath[i, i] = p.epsilon
        Hbath[i, i + 1] = Hbath[i + 1, i] = p.tb

    # Evaluate gfs:
    gf_ij_w_imp, Eimp, Timp = calculate_green_function(Himp, omega)
    gf_ij_w_bath, Ebath, Tbath = calculate_green_function(Hbath, omega)

    # Calculate Hybridization self energy:
    Delta_ij_w = np.zeros((1, 1, p.n_omega), complex)
    Delta_ij_w[0, 0, :] = p.td * p.td * gf_ij_w_bath[0,0,:]

    # Solve for full gf (in impurity subspace) using the Hybridization self energy:
    gf_ij_w = np.zeros((1, 1, p.n_omega), complex)
    for i in range(p.n_omega):
        gf_ij_w[:,:, i] = np.linalg.inv(np.linalg.inv(gf_ij_w_imp[:,:, i]) - Delta_ij_w[:,:, i])

    res['omega'] = omega
    res['gf_ij_w_imp'] = gf_ij_w_imp
    res['gf_ij_w_bath'] = gf_ij_w_bath
    res['Delta_ij_w'] = Delta_ij_w
    res['gf_ij_w'] = gf_ij_w

    return res


if __name__ == '__main__':
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    params = {
        'epsilon': 0.0,
        'td': 3.162 * 2, # *2 for two leads
        'tb': 10.0,
        # 'td': 0.3162 * 2, # *2 for two leads
        # 'tb': 1.0,
        'n_bath': 300,
        'n_omega': 500,
        'omega_c': 200,
        # 'omega_c': 4,
        'eta': 0.1,
    }
    p = Struct(**params)

    # Numerical calculation:
    res_blocks = block_calculation(**params)
    Delta_direct = res_blocks["Delta_ij_w"]
    omega = res_blocks["omega"]

    # Inverse calculation:
    res_inverse = inverse_calculation(**params)
    Delta_inverse = res_inverse["Delta_ij_w"]

    # Analytical calculation:
    Gamma_analytical = 0.5 * p.td ** 2 / p.tb ** 2 * np.lib.scimath.sqrt((2 * p.tb) ** 2 - omega ** 2).real
    Lambda_analytical = hilbert(Gamma_analytical).imag

    # Plot Hybridizations:
    plt.figure()

    plt.plot(omega, -Delta_direct[0, 0, :].imag, color='black', linewidth=2.0, label=r'$-\Gamma$ numerical')
    plt.plot(omega, Delta_direct[0, 0, :].real, color='black', linewidth=2.0, linestyle='--', label=r'$\Lambda$ numerical')

    plt.plot(omega, Gamma_analytical, color='red', linewidth=3.0, label=r'$-\Gamma$ analytical')
    plt.plot(omega, Lambda_analytical, color='red', linewidth=3.0, linestyle='--', label=r'$\Lambda$ from Hilbert transform')

    plt.plot(omega, -Delta_inverse.imag[0,0,:], color='blue', linewidth=3.0, alpha=0.5, linestyle='-', label=r'$-\Gamma$ from inverse')
    plt.plot(omega, Delta_inverse.real[0,0,:], color='blue', linewidth=3.0, alpha=0.5, linestyle='--', label=r'$\Lambda$ from inverse')

    hfreq  = omega[np.abs(omega) >= 20.0]
    plt.plot(hfreq, 40.0 / hfreq, color='black', linewidth=1.0, linestyle=':', label=r'$40/\omega$')

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta$')
    plt.legend(loc='best')
    # plt.tight_layout()
    plt.savefig('hybridization_1D_chain.pdf')
    # plt.show()
