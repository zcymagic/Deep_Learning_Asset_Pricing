"""Synthetic panel with a known pricing structure, used to validate every tool before real data.

Ground truth (all switchable through SyntheticSpec):
	* a priced "market" factor f_t with conditional price of risk lambda_t = lam * (1 + kappa * credit_spread_t)
	* a priced value factor v_t loading on BEME only, not spanned by the market: the SDF must tilt
	  toward value (without it the market alone prices everything and SDF weights on characteristics
	  are not identified -- a CAPM world)
	* exposures beta(z, s) that are additive, partly nonlinear (U-shaped ST_REV), state-dependent
	  (momentum and investment loadings move with credit_spread) and contain one interaction (r12_2 x LME)
	* CF2P is a near-duplicate of BEME but has no role in beta -> the SDF is only weakly identified
	  between the two (the Rashomon test case)
	* mispricing: alpha in small / illiquid stocks (LME low, Spread high), a limits-to-arbitrage region
	* missing risk: a second factor loading on momentum that is priced only in high-VIX months (vix > 0.8 sd)
The oracle SDF prices both factors exactly, so auditing it must find exactly the two planted failures.
"""
from dataclasses import dataclass

import numpy as np

from .data import Panel

CHAR_NAMES = ['LME', 'Spread', 'IdioVol', 'LTurnover', 'r12_2', 'ST_REV', 'BEME', 'CF2P', 'OP', 'Investment']
MACRO_NAMES = ['credit_spread', 'term_spread', 'vix', 'inflation', 'unemployment', 'ip_growth']


@dataclass
class SyntheticSpec:
	T: int = 600
	N: int = 1000
	seed: int = 0
	p_obs: float = 0.85
	rho_char: float = 0.95
	corr_value: float = 0.95		# latent correlation BEME / CF2P
	lam: float = 0.012				# mean price of factor risk (monthly)
	kappa: float = 0.5				# lambda_t = lam * (1 + kappa * credit_spread_t)
	sigma_f: float = 0.04
	value_premium: float = 0.006	# premium of the value factor (loading = z_BEME); 0 disables
	sigma_v: float = 0.02
	sigma_e: float = 0.10
	interaction: float = -2.0		# beta += interaction * z_r12_2 * z_LME
	mispricing: float = 0.008		# alpha in the LME < -0.25 & Spread > 0.25 region (0 disables)
	missing_factor: float = 0.2	# premium of the unpriced momentum factor when vix > 0.8 (0 disables)
	sigma_g: float = 0.02


def true_beta(Z, S, spec):
	"""Exposure to the priced factor. Z: T x N x K, S: T x 1 (credit_spread)."""
	z = {name: Z[:, :, k] for k, name in enumerate(CHAR_NAMES)}
	s = S[:, :1]
	beta = 1.0 \
		+ 1.0 * z['BEME'] \
		+ 0.8 * z['r12_2'] * (1.0 - 0.8 * s) \
		+ 0.6 * z['OP'] \
		+ 2.0 * (z['ST_REV'] ** 2 - 1.0 / 12) \
		- 0.6 * z['Investment'] * (1.0 + 0.5 * s) \
		+ spec.interaction * z['r12_2'] * z['LME']
	return beta


def _rank_normalize(x, mask):
	out = np.zeros_like(x)
	for t in range(x.shape[0]):
		m = mask[t]
		v = x[t, m]
		r = v.argsort().argsort()
		out[t, m] = (r + 0.5) / len(v) - 0.5
	return out


def _ar1(T, rho, rng, shape=()):
	x = np.zeros((T,) + shape)
	x[0] = rng.standard_normal(shape)
	for t in range(1, T):
		x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(shape)
	return x


def make_synthetic_panel(spec=SyntheticSpec()):
	rng = np.random.default_rng(spec.seed)
	T, N, K = spec.T, spec.N, len(CHAR_NAMES)
	mask = rng.random((T, N)) < spec.p_obs

	# latent characteristics with persistence and economically plausible correlations
	lat = _ar1(T, spec.rho_char, rng, (N, K))
	ix = {name: k for k, name in enumerate(CHAR_NAMES)}
	lat[:, :, ix['Spread']] = -0.6 * lat[:, :, ix['LME']] + 0.8 * lat[:, :, ix['Spread']]
	lat[:, :, ix['IdioVol']] = -0.5 * lat[:, :, ix['LME']] + 0.87 * lat[:, :, ix['IdioVol']]
	c = spec.corr_value
	lat[:, :, ix['CF2P']] = c * lat[:, :, ix['BEME']] + np.sqrt(1 - c ** 2) * lat[:, :, ix['CF2P']]
	Z = np.stack([_rank_normalize(lat[:, :, k], mask) for k in range(K)], axis=2)

	# macro series
	macro = np.stack([_ar1(T, 0.9, rng) for _ in MACRO_NAMES], axis=1)
	macro[:, 2] = 0.3 * macro[:, 0] + np.sqrt(1 - 0.09) * macro[:, 2]	# vix mildly related to credit
	macro = (macro - macro.mean(0)) / macro.std(0)
	S = macro[:, [0]]

	beta = true_beta(Z, S, spec)
	lam_t = spec.lam * (1.0 + spec.kappa * S[:, 0])
	f = spec.sigma_f * rng.standard_normal(T)
	R = beta * (lam_t + f)[:, None] + spec.sigma_e * rng.standard_normal((T, N))
	v = spec.sigma_v * rng.standard_normal(T)
	R += Z[:, :, ix['BEME']] * (spec.value_premium + v)[:, None]

	z = {name: Z[:, :, k] for k, name in enumerate(CHAR_NAMES)}
	alpha_region = (z['LME'] < -0.25) & (z['Spread'] > 0.25)
	R += spec.mispricing * alpha_region
	high_vix = macro[:, 2] > 0.8
	g = spec.sigma_g * rng.standard_normal(T)
	R += z['r12_2'] * (spec.missing_factor * high_vix + g * (spec.missing_factor != 0))[:, None]

	R = np.where(mask, R, 0.0).astype(np.float32)
	Z = np.where(mask[:, :, None], Z, 0.0).astype(np.float32)

	# oracle SDF prices both (independent) factors conditionally: E_t[M (lam_t + f)] = E_t[M (premium + v)] = 0
	b_t = lam_t / (spec.sigma_f ** 2 + lam_t ** 2)
	b_v = spec.value_premium / (spec.sigma_v ** 2 + spec.value_premium ** 2)
	M_oracle = 1.0 - b_t * (lam_t + f) - b_v * (spec.value_premium + v)

	panel = Panel(R=R, Z=Z, mask=mask, macro=macro.astype(np.float32),
		char_names=list(CHAR_NAMES), macro_names=list(MACRO_NAMES), dates=np.arange(T))
	truth = {
		'beta': np.where(mask, beta, 0.0), 'M_oracle': M_oracle, 'lam_t': lam_t, 'f': f, 'v': v,
		'alpha_region': alpha_region & mask, 'high_vix': high_vix,
	}
	return panel, truth


def split_panel(panel, n_train, n_valid):
	T = panel.T
	return (panel.subset_time(np.arange(0, n_train)),
		panel.subset_time(np.arange(n_train, n_train + n_valid)),
		panel.subset_time(np.arange(n_train + n_valid, T)))
