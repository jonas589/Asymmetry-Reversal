import numpy as np
from zeta import couple, quality_check
from zeta import ζ_p as zeta_p

def approx_coeffs(ν_p, ν_g, νm_p, νm_g, ζ_p, ζ_g):

	
	# Compute approximate mixing coefficients for expressing each
	# mixed mode as a linear combination of p- and g-modes.

	# We are in particular interested in specifically the p-mode
	# coefficients, although the subsequent lines also compute
	# the g-mode coefficients.

	# These coefficients ought to be such that the sum of squares
	# of p-mode coefficients sum to 1-ζ, and the sum of squares of
	# g-mode coefficients sum to ζ.
	

	assert len(ν_p) == len(νm_p)
	assert len(ν_g) == len(νm_g)

	N_p = len(ν_p)
	N_g = len(ν_g)

	C_approx = np.zeros((N_p, N_p + N_g))
	C_approx[:N_p, :N_p] = np.identity(N_p)

	C_approx[:N_p, :] *= np.sqrt(1-ζ_p)[:, None]

	diff_g = 1 / (ν_p[:, None]**2 - νm_g[None, :]**2)
	C_approx[:N_p, N_p:] = np.sqrt(1-ζ_g)[None, :] * diff_g / np.sqrt(np.sum(diff_g**2, axis=0))

	return C_approx