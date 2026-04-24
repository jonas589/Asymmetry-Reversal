import numpy as np

def lorentzian_single(ν, ν0, Γ):
    return np.sqrt(Γ)/(ν - ν0 + 1j * Γ)

def power_complex_lorentzians(ν, νν, ΓΓ, AA):
    # Take |.|^2 of the sum
    V = ν[None, :]
    V0 = np.asarray(νν)[:, None]
    Γ = np.asarray(ΓΓ)[:, None]
    A = np.asarray(AA)[:, None]

    L = A * np.sqrt(Γ) / (V - V0 + 1j * Γ)
    
    return np.abs(np.sum(L, axis=0))**2

    # return np.abs(sum(A * lorentzian_single(ν, ν0, Γ)
    #            for ν0, Γ, A in zip(νν, ΓΓ, AA)))**2

def power_real_lorentzians(ν, νν, ΓΓ, AA):
    # Take sum of |.|^2
    V = ν[None, :]
    V0 = np.asarray(νν)[:, None]
    G = np.asarray(ΓΓ)[:, None]
    A = np.asarray(AA)[:, None]

    L = A * np.sqrt(G) / (V - V0 + 1j * G)

    return(np.sum(np.abs(L)**2, axis = 0))
    
    # return sum(np.abs(A * lorentzian_single(ν, ν0, Γ))**2
    #            for ν0, Γ, A in zip(νν, ΓΓ, AA))
