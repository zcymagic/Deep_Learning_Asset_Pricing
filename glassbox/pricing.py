"""SDF construction, the no-arbitrage moment loss and evaluation metrics.

Conventions follow Chen-Pelger-Zhu: the SDF is M_t = 1 - sum_i w_{t,i} R_{t,i}; the tradable SDF
factor is F_t = sum_i w_{t,i} R_{t,i} with weights normalized to sum |w_t| = 1 for evaluation; Sharpe
ratios are monthly unless marked annualized.
"""
import numpy as np
import torch


def sdf_from_weights(w, R, mask):
	"""M_t = 1 - sum_i w_ti R_ti (torch)."""
	return 1.0 - (w * R * mask).sum(dim=1)


def moment_loss(M, R, mask, h=None, weighted=True):
	"""Average squared pricing error of the conditional moments E[M_{t} R_{t,i} h_{t,i}] = 0.

	h: C x T x N instruments (None = unconditional, h = 1). Matches FeedForwardModelWithNA_GAN._add_loss.
	"""
	maskf = mask.float()
	T_i = maskf.sum(dim=0).clamp(min=1.0)
	base = R * maskf * M[:, None]
	if h is None:
		h = torch.ones((1,) + R.shape, device=R.device)
	em = (base[None] * h).sum(dim=1) / T_i	# C x N
	if weighted:
		wt = T_i / T_i.max()
		return (em ** 2 * wt).mean()
	return (em ** 2).mean()


def normalized_weights(w, mask):
	"""Normalize each month so that sum_i |w_ti| = 1 (numpy or torch)."""
	if isinstance(w, torch.Tensor):
		w = w * mask
		return w / w.abs().sum(dim=1, keepdim=True).clamp(min=1e-12)
	w = np.where(mask, w, 0.0)
	return w / np.maximum(np.abs(w).sum(axis=1, keepdims=True), 1e-12)


def sdf_factor(w, R, mask):
	"""Tradable SDF factor F_t with normalized weights (numpy)."""
	wn = normalized_weights(np.asarray(w), mask)
	return (wn * R).sum(axis=1)


def sharpe(x, annualize=False):
	x = np.asarray(x, dtype=float)
	sr = x.mean() / x.std()
	return sr * np.sqrt(12) if annualize else sr


def newey_west_se(x, lags=None):
	"""HAC standard error of the mean of a series."""
	x = np.asarray(x, dtype=float)
	T = len(x)
	if lags is None:
		lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
	e = x - x.mean()
	s = e @ e / T
	for l in range(1, lags + 1):
		s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / T
	return np.sqrt(max(s, 1e-20) / T)


def tstat(x, lags=None):
	return np.mean(x) / newey_west_se(x, lags)


def explained_variation(beta, R, mask):
	"""EV and (weighted) cross-sectional R2 using beta as loadings, as in src/model/model_utils.py."""
	beta = np.where(mask, beta, 0.0)
	num = (beta * R).sum(axis=1)
	den = np.maximum((beta * beta).sum(axis=1), 1e-12)
	R_hat = beta * (num / den)[:, None]
	res = np.where(mask, R - R_hat, 0.0)
	Rm = np.where(mask, R, 0.0)
	N_t = np.maximum(mask.sum(axis=1), 1)
	T_i = np.maximum(mask.sum(axis=0), 1)
	ev = 1 - np.mean((res ** 2).sum(axis=1) / N_t) / np.mean((Rm ** 2).sum(axis=1) / N_t)
	xs = 1 - np.mean((res.sum(axis=0) / T_i) ** 2 * T_i) / np.mean((Rm.sum(axis=0) / T_i) ** 2 * T_i)
	return ev, xs


def spanning_test(F_big, F_small):
	"""Regress F_big on F_small (with intercept). Returns alpha, its NW t-stat and the R2.

	A significant alpha means the bigger (more complex) SDF earns Sharpe ratio the smaller one cannot
	span: the price of interpretability.
	"""
	X = np.column_stack([np.ones_like(F_small), F_small])
	coef, *_ = np.linalg.lstsq(X, F_big, rcond=None)
	res = F_big - X @ coef
	r2 = 1 - res.var() / F_big.var()
	return coef[0], coef[0] / newey_west_se(res + coef[0]), r2


def factor_sdf_coefficients(F):
	"""Unconditional SDF M = 1 - b'F that prices the traded factors F (T x J): b = E[FF']^-1 E[F]."""
	F = np.atleast_2d(np.asarray(F, dtype=float).T).T
	return np.linalg.solve(F.T @ F / len(F), F.mean(axis=0))


def factor_sdf(F_est, F_eval=None):
	b = factor_sdf_coefficients(F_est)
	F_eval = F_est if F_eval is None else F_eval
	F_eval = np.atleast_2d(np.asarray(F_eval, dtype=float).T).T
	return 1.0 - F_eval @ b
