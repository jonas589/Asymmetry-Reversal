import numpy as np
from zeta import couple, ζ_p as zeta_p
import lorentzians as lor
import Coeff as Co

def neg_lnL(arguments, P_obs, ν_range, Γ_p, i_deg, Vis_1, Vis_2):

    q_1, a, Δν, ΔΠ, ε_p_0, δν1, δν2, ε_g_1, ε_g_2, ν_max, N, ν_core, ν_env, α, β_1, β_2, *_ = arguments

    q_2 = a*((1-np.sqrt(1-((4*q_1)/(q_1+1)**2)**np.sqrt(3)))/((1+np.sqrt(1-((4*q_1)/(q_1+1)**2)**np.sqrt(3)))))

    i = np.radians(i_deg)
    eps = 1e-8

    ε_p_1 = ε_p_0 + 1/2 - (δν1/Δν)
    ε_p_2 = ε_p_0 + 1 - (δν2/Δν)

    σ_gran = 100 * (ν_max/30)**(-0.61) # Gran. amplitude
    τ_gran = 3000 * (ν_max / 30)**(-0.81) # Gran. timescale
    W_noise = 2.0

    # May change delta nu factor for stars other than KIC 7341231
    νp_0 = pure_modes(Δν, ΔΠ, ε_p_0, ε_p_0, ε_g_1, ν_max, α, 0)[0]
    νp_1, νg_1 = pure_modes(Δν, ΔΠ, ε_p_1, ε_p_0, ε_g_1, ν_max, α, β_1)
    νp_2, νg_2 = pure_modes(Δν, ΔΠ/np.sqrt(3), ε_p_2, ε_p_0, ε_g_2, ν_max, α, β_2)

    if any(len(arr) == 0 for arr in [νp_0, νp_1, νg_1, νp_2, νg_2]):
        return 1e15

    if not (np.all(np.diff(νp_0) > 0) and 
        np.all(np.diff(νp_1) > 0) and 
        np.all(np.diff(νp_2) > 0)):
        return 1e15

    P_back = W_noise + (4 * σ_gran**2 * τ_gran * 1e-6) / (1 + (2 * np.pi * ν_range * τ_gran * 1e-6)**2)

    σ_env = 0.66*(ν_max)**(0.88)/np.sqrt(8*np.log(2)) # σ, W_env = 0.66*(ν_max)**(0.88)
    gauss = np.exp((-1/2)*((ν_range-ν_max)/(σ_env))**2)


    P_l0_mod = sum_l0_lorentzians(ν_range, νp_0, Γ_p)
    P_l1_mod = sum(sum_mixed_lorentzians(ν_range, q_1, Δν, ΔΠ, Γ_p, m*ν_core, m*ν_env, νp_1, νg_1)[0]*E[(1,abs(m))](i) for m in range(-1, 2)) * Vis_1
    P_l2_mod = sum(sum_mixed_lorentzians(ν_range, q_2, Δν, ΔΠ/np.sqrt(3), Γ_p, m*ν_core, m*ν_env, νp_2, νg_2)[0]*E[(2,abs(m))](i) for m in range (-2, 3)) * Vis_2

    P_mod = np.clip(((P_l0_mod + P_l1_mod + P_l2_mod) * N * gauss)+P_back, eps, None)
    lnL = float(np.sum((P_obs/P_mod)-np.log(P_obs/P_mod)))
    
    return lnL if np.isfinite(lnL) else 1e15

# Dispatch dictionary
E = {
        (1,0): lambda i: np.cos(i)**2,
        (1,1): lambda i: (1/2)*np.sin(i)**2,
        (2,0): lambda i: (1/4)*(3*np.cos(i)**2-1)**2,
        (2,1): lambda i: (3/8)*np.sin(2*i)**2,
        (2,2): lambda i: (3/8)*np.sin(i)**4
    }

def pure_modes(Δν, ΔΠ, ε_p, ε_p_0, ε_g, ν_max, α, β):
    # May change Δν factor for stars other than KIC 7341231

    min_n = np.round(np.floor((ν_max - 4*Δν) / Δν))
    max_n = np.round(np.ceil((ν_max + 4*Δν) / Δν))
    n_range = np.arange(min_n, max_n)
    n_max_ref = (ν_max / Δν) - ε_p_0
    νp = Δν * (n_range + β*(n_range-n_max_ref) + (α/2)*(n_range-n_max_ref)**2 + ε_p)
    
    min_n_g = np.round(np.floor(1 / (np.max(νp) * ΔΠ)))
    max_n_g = np.round(np.ceil(1 / (np.min(νp) * ΔΠ)))
    νg = 1 / (ΔΠ * (np.arange(min_n_g, max_n_g) + 1/2 + ε_g))[::-1]
    return νp, νg

def sum_l0_lorentzians(ν_range, νp, Γ_p):
    A_l0 = np.ones(len(νp))
    return lor.power_complex_lorentzians(ν_range, νp, Γ_p * A_l0**2, A_l0)

def sum_mixed_lorentzians(ν_range, q_L, Δν, ΔΠ, Γ_p, ν_core, ν_env, νp, νg):

    νp_env = νp + ν_env
    νg_core = νg + ν_core

    νmp, νmg = couple(νp_env, νg_core, q_L, q_L)

    nu_mixed = np.concatenate((νmp, νmg))

    matrix = Co.approx_coeffs(νp_env, νg_core, νmp, νmg, zeta_p(νmp, q_L, ΔΠ, Δν, νp_env), zeta_p(νmg, q_L, ΔΠ, Δν, νp_env))

    A_mixed = np.concatenate((np.diag(matrix[:len(νp_env), :len(νp_env)]), np.sum(matrix[:, len(νp_env):], axis=0)))

    Gamma_mixed = np.abs(A_mixed)**2 * Γ_p

    P_mod = lor.power_complex_lorentzians(ν_range, nu_mixed, Gamma_mixed, A_mixed)

    return P_mod, nu_mixed

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import lightkurve as lk
    import nifty_ls
    from astropy import units as u
    from os.path import isfile
    import dill
    ex = 60
    star = "KIC 7341231"
    filename = f"{star}_{ex}.npy"

    if not isfile(filename):
        print(f"Downloading lightcurve for {star}...")
        search_result = lk.search_lightcurve(star, mission="Kepler", exptime=ex)
        lcs = search_result.download_all()
        lc = lcs.stitch().flatten().remove_outliers()
        with open(filename, "wb") as f:
            dill.dump(lc, f)
    else:
        print(f"Loading cached lightcurve: {filename}")
        with open(filename, "rb") as f:
            lc = dill.load(f)

    a_sort = np.argsort(lc.time.value)
    periodogram = nifty_ls.lombscargle(
        t=lc.time[a_sort].value,
        y=lc.flux[a_sort].value,
        dy=lc.flux_err[a_sort].value,
        normalization='psd',
        nthreads=4,
        nyquist_factor=1
    )

    Ny_freq = (1 / (2 * ex)) * (10**6)
    print(f"Nyquist frequency: {Ny_freq}")

    ν_range_full = np.asarray((periodogram.freq() / u.d).to(u.uHz))
    P_obs_full = periodogram.power[:len(ν_range_full)]

    q_1 = 0.379
    a = 0.145434
    Δν = 29
    ΔΠ = 111.18 / 1e6
    ε_p_0 = 0.256
    δν1 = -0.118
    δν2 = 3.413
    ε_g_1 = 0.31
    ε_g_2 = 0.138
    ν_max = 408
    Γ_p = 0.05
    N = 400
    ν_core = (781)/1000 # Write into parenthesis in nanohertz, converted to microhertz
    ν_env = (53)/1000 # Write into parenthesis in nanohertz, converted to microhertz
    inc_angle = 85
    Vis_1 = 1.54
    Vis_2 = 0.58

    initial_guesses = [q_1, a, ΔΠ, δν1, δν2, ε_g_1, ε_g_2, ν_max, Γ_p, N, ν_core, ν_env]

    start_values = [q_1, a, Δν, ΔΠ, ε_p_0, δν1, δν2, ε_g_1, ε_g_2, ν_max, Γ_p, N, ν_core, ν_env, inc_angle, Vis_1, Vis_2]

    bounds = [
        (0.35, 0.4), # q_1
        (0.01, 0.8), # a
        (105/1e6, 120/1e6), # ΔΠ
        (-0.160, -0.060), # δν1
        (3.100, 3.600), # δν2
        (0.260, 0.340), # ε_g_1
        (0.110, 0.160), # ε_g_2
        (360, 460), # ν_max
        (0.001, 0.5), # Γ_p
        (100, 1000), # N
        (0.777, 0.785), # ν_core
        (0.047, 0.059) # ν_env
    ]

    ν_mask = (ν_range_full >= ν_max-5*Δν) & (ν_range_full <= ν_max+5*Δν)
    P_obs = P_obs_full[ν_mask]
    ν_range = ν_range_full[ν_mask]

    restore = True
    if restore:
        print("Reloading previously fitted parameters...")
        variables = [np.float64(0.37189833447914633),
                    np.float64(0.07119841918770088),
                    29,
                    np.float64(0.00010876160653607039),
                    0.256,
                    np.float64(-0.07886551056351797),
                    np.float64(3.56384684822003),
                    np.float64(0.2649615695134452),
                    np.float64(0.15565378298342722),
                    np.float64(377.97440833520415),                              
                    0.02,                                             # np.float64(0.4990476416961482),
                    80,                                            # np.float64(232.7444734345702),
                    np.float64(0.7778279543597688),
                    np.float64(0.051844795032554236),
                    85,
                    1.54,
                    0.58
                    ]
        q_1, a, Δν, ΔΠ, ε_p_0, δν1, δν2, ε_g_1, ε_g_2, ν_max, Γ_p, N, ν_core, ν_env, inc_angle, Vis_1, Vis_2 = variables

    if not restore:
        print("Loading values from Liagre et al. (2026)...")

    versus = False
    if versus:
        print("Plotting observed vs. fitted PSD...")

        # ax = plt.subplots(3, 1, sharey = True, sharex = True, figsize=(10, 8))[1]
        ax = plt.subplots(1, 1, sharey = True, sharex = True, figsize=(10, 8))[1]
        i_rad = np.radians(inc_angle)
        q_2 = a*((1-np.sqrt(1-((4*q_1)/(q_1+1)**2)**np.sqrt(3)))/((1+np.sqrt(1-((4*q_1)/(q_1+1)**2)**np.sqrt(3)))))
        ε_p_1 = ε_p_0 + 1/2 - (δν1/Δν)
        ε_p_2 = ε_p_0 + 1 - (δν2/Δν)

        σ_gran = 100 * (ν_max/30)**(-0.61)
        τ_gran = 3000 * (ν_max / 30)**(-0.81)
        W_noise = 2.0

        ν_mask = (ν_range >= ν_max-4*Δν) & (ν_range <= ν_max+4*Δν)
        ν_sub = ν_range[ν_mask]
        P_obs_sub = P_obs[ν_mask]

        σ_env = 0.66*(ν_max)**(0.88)/np.sqrt(8*np.log(2))
        gauss = np.exp((-1/2)*((ν_sub-ν_max)/(σ_env))**2)

        νp_0 = pure_modes(Δν, ΔΠ, ε_p_0, ε_g_1, ν_max)[0]
        P_l0_mod = sum_l0_lorentzians(ν_sub, νp_0, Γ_p)

        νp_1, νg_1 = pure_modes(Δν, ΔΠ, ε_p_1, ε_g_1, ν_max)
        P_l1_mod = sum(sum_mixed_lorentzians(ν_sub, q_1, Δν, ΔΠ, Γ_p, m*ν_core, m*ν_env, νp_1, νg_1)[0]*E[(1,abs(m))](i_rad) for m in range(-1, 2)) * Vis_1
        νmixed_1_all = [sum_mixed_lorentzians(ν_sub, q_1, Δν, ΔΠ, Γ_p, m*ν_core, m*ν_env, νp_1, νg_1)[1] for m in range(-1, 2)]

        νp_2, νg_2 = pure_modes(Δν, ΔΠ/np.sqrt(3), ε_p_2, ε_g_2, ν_max)
        P_l2_mod = sum(sum_mixed_lorentzians(ν_sub, q_2, Δν, ΔΠ/np.sqrt(3), Γ_p, m*ν_core, m*ν_env, νp_2, νg_2)[0]*E[(2,abs(m))](i_rad) for m in range (-2, 3)) * Vis_2
        νmixed_2_all = [sum_mixed_lorentzians(ν_sub, q_2, Δν, ΔΠ/np.sqrt(3), Γ_p, m*ν_core, m*ν_env, νp_2, νg_2)[1] for m in range(-2, 3)]

        P_back = W_noise + (4 * σ_gran**2 * τ_gran * 1e-6) / (1 + (2 * np.pi * ν_sub * τ_gran * 1e-6)**2)
        P_mod = ((P_l0_mod + P_l1_mod + P_l2_mod) * N * (gauss**2)) + P_back

        # ax[0].plot(ν_sub, P_obs_sub, c = '#000000', lw = .6, label='Observed', alpha = 0.5)
        # ax[1].plot(ν_sub, P_mod, c="#e31a1c",  label='Model', alpha=0.9)
        # ax[2].plot(ν_sub, P_obs_sub, c = '#000000', lw = .6, label='Observed')
        # ax[2].plot(ν_sub, P_mod, c="C1",  label='Model')
        
        # for v_list, color in zip([νp_0, νmixed_1_all, νmixed_2_all], ['C0', 'C1', 'C2']):
        #         v_visible = v_list[(v_list >= ν_max-4*Δν) & (v_list <= ν_max+4*Δν)]
        #         ax[1].vlines(v_list, 0, 4000, color=color, lw=0.5, alpha = 0.8)
        #         ax[2].vlines(v_list, 0, 4000, color=color, lw=0.5, alpha = 0.8)

        # ax[1].set_xlabel(r'Frequency $[\mu Hz]$')
        # ax[0].set_ylabel(r'Power $[ppm^{2}/\mu Hz]$')
        ax.plot(ν_sub, P_obs_sub, c = '#000000', lw = .6, label='Observed')
        ax.plot(ν_sub, P_mod, c="C1",  label='Model')
        for v_list, color in zip([νp_0, νmixed_1_all[0], νmixed_1_all[1], νmixed_1_all[2], νmixed_2_all[0], νmixed_2_all[1], νmixed_2_all[2], νmixed_2_all[3], νmixed_2_all[4]], ['Green', 'Cyan', 'Blue', 'Cyan', 'Red', 'Orange', 'Yellow', 'Orange', 'Red']):
                ax.vlines(v_list, 0, 4000, color=color, lw=0.5, alpha = 0.8)
        plt.xlim(ν_max-4*Δν, ν_max+4*Δν)
        plt.tight_layout()
        plt.show()
    