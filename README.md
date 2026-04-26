# Asymmetry-Reversal

Hello dear user!

I'm Jonas Káral, a Physics Undergraduate at the University of Sydney (as of writing this). This repository was created for my 3rd year individual research project (see SCDL3991 on USYD website), where I worked closely with Joel Ong from the Sydney Institute for Astronomy (SIfA).

Our project was based on fitting the power spectra of mixed mode oscillating stars using the lorentzian approximation of mixed modes (see f.ex Ong et al. (2022)), and testing the goodness of fit of power spectra created using either the sum of squares and square of sums approaches.

Therefore, the functionality of this repository is designed in three main steps:

1.- The file 'lorentzians.py' creates the lorentzians corresponding to the mixed mode frequency, the frequency range and the FWHM. Using these, the two remaining functions create power spectra from these lorentzians, either with the sum of squares method ('power_real_lorentzians') or the square of sums ('power_complex_lorentzians'). 

2.- The functions in 'spectrum.py' - apart from 'neg_lnL' - generate mixed mode frequencies, and from there use the function 'approx_coeffs' and the repository 'zeta', both created by Joel Ong, to approximate mixed mode coupling for l = 1 and l = 2 modes; l = 0 modes do not couple with g-modes, so these are generated on their own. These mixed mode frequencies, along with a frequency range given by the Nyquist frequency of the power spectrum being fitted for, are passed into the functions from step (1), and rotational splitting and mode visibility is applied where relevant. With the 'power vs. frequency' functions for each degree, we apply an overall Gaussian to our data.

3.- Lastly, we utilize the log-likelihood (see f.ex Liagre et al. (2026)) to approximate the goodness of fit. The code is designed such that we can use either scipy.optimize.differential_evolution or scipy.optimize.minimize, where initial guesses and bounds for the following stellar parameters must be provided (in the same order):

q_1 (Coupling between p- and g-modes for l = 1)
a (A percentage value of an upper bound of q_2 found from q_1) # Strictly 0<a<=1
Δν (Large frequency separation of p-modes)
ΔΠ (Large period separation of g-modes)
ε_p_0 (Phase shift for p-modes of l = 0)
δν1 (Value which allows us to find phase shift for p-modes of l = 1)
δν2 (Value which allows us to find phase shift for p-modes of l = 2)
ε_g_1 (Phase shift for g-modes of l = 1)
ε_g_2 (Phase shift for g-modes of l = 2)
ν_max (Frequency of maximum power)
Γ_p (FWHM of lorentzians, perhaps fixed to some low value such as 0.1)
N (Coefficient for overall Gaussian applied in step (2))
ν_core(Rotational frequency in the core for rotational splitting in step (2)) Write in nanohertz, converted to microhertz
ν_env (Rotational frequency in the envelope for rotational splitting in step (2)) # Write in nanohertz, converted to microhertz
inc_angle (Star's inclination angle)
Vis_1 (Visibility of l = 1 modes, as in table (4) of Mosser et al. (2012))
Vis_2 (Visibility of l = 2 modes, as in table (4) of Mosser et al. (2012))