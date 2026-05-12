import numpy as np
import matplotlib.pyplot as plt
from zeta import couple, ζ_p as zeta_p
import Coeff as Co

def lorentzian_single(ν, ν0, Γ):
    return np.sqrt(Γ)/(ν - ν0 + 1j * Γ)

def power_complex_lorentzians(ν, νν, ΓΓ, AA):
    # Take |.|^2 of the sum
    V = ν[None, :]
    V0 = np.asarray(νν)[:, None]
    Γ = np.asarray(ΓΓ)[:, None]
    A = np.asarray(AA)[:, None]

    L = A * lorentzian_single(V, V0, Γ)
    
    return np.abs(np.sum(L, axis=0))**2

    # return np.abs(sum(A * lorentzian_single(ν, ν0, Γ)
    #            for ν0, Γ, A in zip(νν, ΓΓ, AA)))**2

def power_real_lorentzians(ν, νν, ΓΓ, AA):
    # Take sum of |.|^2
    V = ν[None, :]
    V0 = np.asarray(νν)[:, None]
    Γ = np.asarray(ΓΓ)[:, None]
    A = np.asarray(AA)[:, None]

    L = A * lorentzian_single(V, V0, Γ)

    return(np.sum(np.abs(L)**2, axis = 0))
    
    # return sum(np.abs(A * lorentzian_single(ν, ν0, Γ))**2
    #            for ν0, Γ, A in zip(νν, ΓΓ, AA))

if __name__ == '__main__':
    Γ_p = 0.1
    q = 1
    Δν = 10
    ΔΠ = 0.15
    f_range = np.arange(-20, 20, 0.001)

    g_freq = np.concatenate((np.arange(1, 11), np.arange(0.3, 10.7, 0.6)))
    p_freq = np.asarray([5.6, 15.6])


    νmp, νmg = couple(p_freq, g_freq, q, q)

    nu_mixed = np.concatenate((νmp, νmg))

    matrix = Co.approx_coeffs(p_freq, g_freq, νmp, νmg, zeta_p(νmp, q, ΔΠ, Δν, p_freq), zeta_p(νmg, q, ΔΠ, Δν, p_freq))

    A_mixed = np.concatenate((np.diag(matrix[:len(p_freq), :len(p_freq)]), np.sum(matrix[:, len(p_freq):], axis=0)))

    Gamma_mixed = np.abs(A_mixed)**2 * Γ_p

    P_mod_complex = power_complex_lorentzians(f_range, nu_mixed, Gamma_mixed, A_mixed)

    P_mod_real = power_real_lorentzians(f_range, nu_mixed, Gamma_mixed, A_mixed)

    ax = plt.subplots(2,1, sharex = True, sharey = True)[1]

    plt.plot(f_range, power_complex_lorentzians(f_range, p_freq, np.asarray(Gamma_mixed[:2]), np.asarray(A_mixed[:2])))
    plt.show()

    ax[0].plot(f_range, P_mod_complex, lw = 3)
    ax[1].plot(f_range, P_mod_real, lw = 3)
    ax[0].vlines(nu_mixed, 0, 10, colors = "green")
    # plt.xlim(4, 8)
    plt.show()