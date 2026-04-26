import numpy as np
from zeta import couple, ζ_p as zeta_p
import lorentzians as lor
import mode_coeffs as Co

def neg_lnL(arguments, P_obs, ν_range):

    q_1, a, Δν, ΔΠ, ε_p_0, δν1, δν2, ε_g_1, ε_g_2, ν_max, Γ_p, N, ν_core, ν_env, i_deg, Vis_1, Vis_2, *_ = arguments

    i = np.radians(i_deg)
    eps = 1e-8

    ε_p_1 = ε_p_0 - 1/2 - (δν1/Δν)
    ε_p_2 = ε_p_0 - 1 - (δν2/Δν)

    # σ_gran = 100 * (ν_max/30)**(-0.7) # Gran. amplitude
    # τ_gran = 3000 * (ν_max / 30)**(-1.0) # Gran. timescale
    # W_noise = 2.0 # Whitenoise

    # P_back = W_noise + (4 * σ_gran**2 * τ_gran * 1e-6) / (1 + (2 * np.pi * ν_sub * τ_gran * 1e-6)**2)

    # May change delta nu factor for stars other than KIC 7341231
    ν_mask = (ν_range >= ν_max-4*Δν) & (ν_range <= ν_max+4*Δν)
    ν_sub = ν_range[ν_mask]
    P_obs_sub = P_obs[ν_mask]

    σ_env = 0.66*(ν_max)**(0.88)/np.sqrt(8*np.log(2)) # σ, W_env = 0.66*(ν_max)**(0.88)
    gauss = np.exp((-1/2)*((ν_sub-ν_max)/(σ_env))**2)

    νp_0 = pure_modes(Δν, ΔΠ, ε_p_0 + (1/2), ε_g_1, ν_max)[0]
    P_l0_mod = sum_l0_lorentzians(ν_sub, νp_0, Γ_p)

    νp_1, νg_1 = pure_modes(Δν, ΔΠ, ε_p_1 + (1/2), ε_g_1, ν_max)
    P_l1_mod = sum(sum_mixed_lorentzians(ν_sub, q_1, Δν, ΔΠ, Γ_p, m*ν_core, m*ν_env, νp_1, νg_1)*E[(1,abs(m))](i) for m in range(-1, 2)) * Vis_1

    νp_2, νg_2 = pure_modes(Δν, ΔΠ/np.sqrt(3), ε_p_2 + 1, ε_g_2, ν_max)
    P_l2_mod = sum(sum_mixed_lorentzians(ν_sub, q_1, Δν, ΔΠ/np.sqrt(3), Γ_p, m*ν_core, m*ν_env, νp_2, νg_2)*E[(2,abs(m))](i) for m in range (-2, 3)) * Vis_2

    P_mod = np.clip(((P_l0_mod + P_l1_mod + P_l2_mod) * N * gauss), eps, None)
    lnL = float(np.sum((P_obs_sub/P_mod)-np.log(P_obs_sub/P_mod)))
    
    return lnL if np.isfinite(lnL) else 1e15

# Dispatch dictionary
E = {
        (1,0): lambda i: np.cos(i)**2,
        (1,1): lambda i: (1/2)*np.sin(i)**2,
        (2,0): lambda i: (1/4)*(3*np.cos(i)**2-1)**2,
        (2,1): lambda i: (3/8)*np.sin(2*i)**2,
        (2,2): lambda i: (3/8)*np.sin(i)**4
    }

def pure_modes(Δν, ΔΠ, ε_p, ε_g, ν_max):
    # May change Δν factor for stars other than KIC 7341231

    min_n = np.round(np.floor((ν_max - 4*Δν) / Δν))
    max_n = np.round(np.ceil((ν_max + 4*Δν) / Δν))
    νp = Δν * (np.arange(min_n, max_n) + ε_p)
    
    min_n_g = np.round(np.floor(1 / (np.max(νp) * ΔΠ)))
    max_n_g = np.round(np.ceil(1 / (np.min(νp) * ΔΠ)))
    νg = 1 / (ΔΠ * (np.arange(min_n_g, max_n_g) + ε_g))[::-1]
    return νp, νg

def sum_l0_lorentzians(ν_range, νp, Γ_p):
    A_l0 = np.ones(len(νp))
    return lor.power_complex_lorentzians(ν_range, νp, Γ_p * A_l0**2, A_l0)

def sum_mixed_lorentzians(ν_range, q, Δν, ΔΠ, Γ_p, ν_core, ν_env, νp, νg):

    νp_env = νp + ν_env
    νg_core = νg + ν_core

    if not np.all(np.isfinite([q, Δν, ΔΠ, Γ_p, ν_core, ν_env])):
        return np.ones_like(ν_range) * 1e-10

    νmp, νmg = couple(νp_env, νg_core, q, q)

    nu_mixed = np.concatenate([νmp, νmg])

    matrix = Co.approx_coeffs(νp_env, νg_core, νmp, νmg, zeta_p(νmp, q, ΔΠ, Δν, νp_env), zeta_p(νmg, q, ΔΠ, Δν, νp_env))

    A_mixed = np.concatenate((np.diag(matrix[:len(νp_env), :len(νp_env)]), np.sum(matrix[:, len(νp_env):], axis=0)))

    Gamma_mixed = np.abs(A_mixed)**2 * Γ_p

    P_mod = lor.power_complex_lorentzians(ν_range, nu_mixed, Gamma_mixed, A_mixed)

    return P_mod