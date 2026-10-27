"""The Rashomon set of glass-box SDFs: which conclusions does the data actually identify?

For a linear-in-parameters SDF (LinearSDF, AdditiveSDF) and fixed instruments h, the GMM objective

	J(theta) = sum_rows omega (a - b' theta)^2 + theta' P theta
	           (unconditional moments + conditional moments E[M R h] = 0, smoothness penalty P)

is exactly quadratic, so it has a closed-form minimizer theta* and the Rashomon set of all SDFs within
a factor (1 + eps) of the best objective is an ellipsoid

	E = {theta : (theta - theta*)' A (theta - theta*) <= r^2},   A = B' Omega B + P.

Two complementary notions are reported:
	* Rashomon multiplicity (method='ellipsoid'): the radius is a relative tolerance r^2 = eps J*, the
	  usual Rashomon definition -- how different can SDFs be that give up at most eps of pricing fit?
	  (eps=None calibrates r^2 as the `level` bootstrap quantile of (theta*_b - theta*)' A (theta*_b - theta*),
	  a simultaneous confidence region; with hundreds of parameters it is very conservative.)
	* Statistical identification (method='bootstrap'): marginal stationary-block-bootstrap intervals of
	  each quantity from closed-form refits on resampled months, Bonferroni-adjusted across features.

Consequences used below:
	* any linear functional g' theta (a stock's SDF weight, a shape-function value, a slope) has the
	  exact range g' theta* +/- r sqrt(g' A^-1 g);
	* any quadratic functional theta' G theta (the variance a characteristic contributes to SDF weights)
	  has exact bounds from a trust-region subproblem;
	* stock-level SDF ambiguity is the half-width of the weight range, scaled by the cross-sectional
	  dispersion of weights that month.
"""
import numpy as np
import pandas as pd
import torch

from .sdf import AdditiveSDF, LinearSDF, _hat_coords


# ------------------------------------------------------------------------------------------------
# quadratic objective
# ------------------------------------------------------------------------------------------------
def penalty_matrix(model):
	"""P such that model.penalty() == theta' P theta (requires lam_group == 0)."""
	D = model.theta().numel()
	P = torch.zeros(D, D, dtype=torch.float64)
	if isinstance(model, LinearSDF):
		return P
	if not isinstance(model, AdditiveSDF):
		raise TypeError('penalty_matrix needs a LinearSDF or AdditiveSDF')
	if model.lam_group > 0:
		raise ValueError('the Rashomon ellipsoid needs a quadratic penalty: set lam_group = 0')
	nk = model.n_knots
	D2 = torch.zeros(nk - 2, nk, dtype=torch.float64)
	for i in range(nk - 2):
		D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
	block = model.lam_smooth * D2.T @ D2
	starts = [1 + k * nk for k in range(model.K)]
	if model.state_varying:
		off = 1 + model.K * nk + model.P
		starts += [off + q * nk for q in range(model.P * model.K)]
	for st in starts:
		P[st:st + nk, st:st + nk] += block
	return P


def quadratic_objective(model, batch, h=None, weighted=True, time_weights=None):
	"""Return (A, b_vec, c0) with J(theta) = theta' A theta - 2 b_vec' theta + c0 (float64).

	Rows: unconditional moments weighted 1/N and conditional moments weighted 1/(C N), as in
	train.fit_moment_loss, each times the CPZ observation weight T_i / max T_i. time_weights (T) are
	bootstrap counts of each month.
	"""
	Psi = model._cached_design(batch).double()
	R, maskf = batch['R'].double(), batch['mask'].double()
	if time_weights is not None:
		maskf = maskf * torch.as_tensor(time_weights, dtype=torch.float64)[:, None]
	T_i = maskf.sum(0).clamp(min=1.0)
	wt = T_i / T_i.max() if weighted else torch.ones_like(T_i)
	N = R.shape[1]
	inst = [(torch.ones_like(R), 1.0 / N)]
	if h is not None:
		C = h.shape[0]
		inst += [(h[c].double(), 1.0 / (C * N)) for c in range(C)]
	D = Psi.shape[1]
	A = torch.zeros(D, D, dtype=torch.float64)
	bv = torch.zeros(D, dtype=torch.float64)
	c0 = 0.0
	for hc, scale in inst:
		Rh = R * maskf * hc / T_i								# T x N
		a = Rh.sum(0)											# N
		Bc = Rh.T @ Psi											# N x D
		om = wt * scale
		A += Bc.T @ (Bc * om[:, None])
		bv += Bc.T @ (a * om)
		c0 += float((om * a * a).sum())
	return A + penalty_matrix(model), bv, c0


def select_shrinkage(model, batch_tr, h, batch_va, grid=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), weighted=True,
		ridge=1e-8):
	"""Pick the L2 shrinkage of the closed-form SDF by validation Sharpe ratio of F = 1 - M = Psi theta.

	Returns (best shrink, DataFrame with train / valid Sharpe ratios for every value).
	"""
	A, bv, _ = quadratic_objective(model, batch_tr, h, weighted)
	D = A.shape[0]
	scale = torch.trace(A) / D
	Psi_tr = model._cached_design(batch_tr).double()
	Psi_va = model._cached_design(batch_va).double()
	rows = []
	for lam in grid:
		th = torch.linalg.solve(A + (ridge + lam) * scale * torch.eye(D, dtype=torch.float64), bv)
		f_tr, f_va = (Psi_tr @ th).numpy(), (Psi_va @ th).numpy()
		rows.append({'shrink': lam, 'sharpe_train': f_tr.mean() / f_tr.std(), 'sharpe_valid': f_va.mean() / f_va.std()})
	df = pd.DataFrame(rows)
	return float(df.loc[df['sharpe_valid'].idxmax(), 'shrink']), df


# ------------------------------------------------------------------------------------------------
# trust-region subproblem: extremes of a quadratic over the unit ball
# ------------------------------------------------------------------------------------------------
def _trs_max(Mq, c, tol=1e-12):
	"""max_{||u|| <= 1} u' Mq u + 2 c' u (Mq symmetric). Returns (value, u)."""
	lam, V = np.linalg.eigh(Mq)
	ch = V.T @ c
	lmax = lam[-1]
	if lmax < 0:
		u = -np.linalg.solve(Mq, c)
		if np.linalg.norm(u) <= 1:
			return float(u @ Mq @ u + 2 * c @ u), u

	def norm_u(l):
		return np.sqrt(np.sum((ch / (l - lam)) ** 2))

	lo = max(lmax, 0.0) + 1e-14 * max(1.0, abs(lmax))
	if norm_u(lo) < 1.0:	# hard case: move along the top eigenvector
		u = V @ np.where(np.isclose(lam, lmax), 0.0, ch / (lmax - lam + 1e-300))
		tau = np.sqrt(max(0.0, 1.0 - u @ u))
		u = u + tau * V[:, -1]
		return float(u @ Mq @ u + 2 * c @ u), u
	hi = lo + np.linalg.norm(c) + abs(lmax) + 1.0
	while norm_u(hi) > 1.0:
		hi *= 2.0
	for _ in range(200):
		mid = 0.5 * (lo + hi)
		if norm_u(mid) > 1.0:
			lo = mid
		else:
			hi = mid
		if hi - lo < tol * max(1.0, hi):
			break
	u = V @ (ch / (hi - lam))
	return float(u @ Mq @ u + 2 * c @ u), u


# ------------------------------------------------------------------------------------------------
# the Rashomon ellipsoid
# ------------------------------------------------------------------------------------------------
class RashomonEllipsoid:
	def __init__(self, model, batch, h=None, eps=0.01, level=0.95, n_boot=200, mean_block=6, seed=0,
			ridge=1e-8, shrink=0.0, weighted=True):
		"""Closed-form refit of `model` on (batch, h) and the Rashomon ellipsoid around it.

		eps given: relative tolerance, r^2 = eps J*. eps None: simultaneous bootstrap calibration at `level`.
		n_boot > 0 also stores bootstrap refits for the marginal (method='bootstrap') tables.
		The model's parameters are overwritten with theta*. ridge (relative to the mean diagonal) only
		regularizes directions that do not move any weight (e.g. the level of a centered shape function).
		shrink adds an L2 penalty shrink * mean-diag(A) * |theta|^2 to the objective (a shrinkage SDF in
		the spirit of Kozak-Nagel-Santosh). With hundreds of parameters the unpenalized optimum overfits
		badly (CPZ panel, additive_state: train Sharpe 4.2, test ~0), so choose it with select_shrinkage.
		The objective stays quadratic, so every exact bound below still holds for the penalized J.
		"""
		A, bv, c0 = quadratic_objective(model, batch, h, weighted)
		D = A.shape[0]
		ridge_I = (ridge + shrink) * torch.trace(A) / D * torch.eye(D, dtype=torch.float64)
		self.shrink = shrink
		A = A + ridge_I
		theta = torch.linalg.solve(A, bv)
		J_star = float(c0 - bv @ theta)
		self.model = model
		self.A = A.numpy()
		self.theta = theta.numpy()
		self.J_star = J_star
		self.boot_thetas = None
		if n_boot > 0:
			from .audit import stationary_bootstrap_indices
			T = batch['R'].shape[0]
			idx = stationary_bootstrap_indices(T, n_boot, mean_block, np.random.default_rng(seed))
			thetas = []
			for b in range(n_boot):
				k = np.bincount(idx[b], minlength=T).astype(float)
				Ab, bb, _ = quadratic_objective(model, batch, h, weighted, time_weights=k)
				thetas.append(torch.linalg.solve(Ab + ridge_I, bb).numpy())
			self.boot_thetas = np.array(thetas)
			d = self.boot_thetas - self.theta
			self.boot_dist = ((d @ self.A) * d).sum(1)
		if eps is None:
			self.r2 = float(np.quantile(self.boot_dist, level))
			self.eps = self.r2 / J_star
		else:
			self.eps = eps
			self.r2 = eps * J_star
		self.Ainv = np.linalg.inv(self.A)
		self.Ainv = 0.5 * (self.Ainv + self.Ainv.T)
		Lc = np.linalg.cholesky(self.Ainv)
		self.L = Lc											# theta = theta* + r L u, ||u|| <= 1
		model.set_theta(theta.float())
		self._feature_moments(batch)

	def _feature_moments(self, batch):
		"""Mean and covariance of the feature vector over all observed stock-months."""
		D = self.theta.size
		S1, S2, n = np.zeros(D), np.zeros((D, D)), 0
		with torch.no_grad():
			for t in range(batch['R'].shape[0]):
				ph = self.model.features(batch, t).double().numpy()
				S1 += ph.sum(0)
				S2 += ph.T @ ph
				n += ph.shape[0]
		self.phi_mean = S1 / n
		self.phi_cov = S2 / n - np.outer(self.phi_mean, self.phi_mean)

	def set_eps(self, eps):
		"""Change the relative tolerance in place (the fit and bootstrap draws are unchanged)."""
		self.eps, self.r2 = eps, eps * self.J_star
		return self

	# --- generic functionals ----------------------------------------------------------------------
	def objective(self, theta):
		d = theta - self.theta
		return self.J_star + d @ self.A @ d

	def linear_range(self, g):
		g = np.asarray(g, dtype=float)
		mid = g @ self.theta
		half = np.sqrt(self.r2 * g @ self.Ainv @ g)
		return mid, mid - half, mid + half

	def quadratic_range(self, G):
		"""Exact (min, max) of theta' G theta over the ellipsoid, and the reference value."""
		r = np.sqrt(self.r2)
		LtG = self.L.T @ G
		Mq = r * r * LtG @ self.L
		Mq = 0.5 * (Mq + Mq.T)
		c = r * LtG @ self.theta
		base = self.theta @ G @ self.theta
		vmax, _ = _trs_max(Mq, c)
		vmin_neg, _ = _trs_max(-Mq, -c)
		return base, base - vmin_neg, base + vmax

	def sample(self, n=1000, seed=0, boundary=True):
		rng = np.random.default_rng(seed)
		u = rng.standard_normal((n, self.theta.size))
		u /= np.linalg.norm(u, axis=1, keepdims=True)
		if not boundary:
			u *= rng.random((n, 1)) ** (1.0 / self.theta.size)
		return self.theta[None] + np.sqrt(self.r2) * u @ self.L.T

	def _boot_interval(self, values_boot, ref, alpha):
		"""Basic (pivotal) bootstrap interval at level 1 - alpha."""
		qlo, qhi = np.quantile(values_boot, [alpha / 2, 1 - alpha / 2])
		return 2 * ref - qhi, 2 * ref - qlo

	def _require_boot(self):
		if self.boot_thetas is None:
			raise ValueError('construct with n_boot > 0 for method="bootstrap"')

	# --- economic quantities ----------------------------------------------------------------------
	def _group_matrix(self, idx):
		G = np.zeros_like(self.phi_cov)
		idx = np.asarray(idx)
		G[np.ix_(idx, idx)] = self.phi_cov[np.ix_(idx, idx)]
		return G

	def importance_table(self, names, groups=None, rel_tol=0.01, method='ellipsoid', alpha=0.05):
		"""Variance each characteristic (or group of them) contributes to SDF weights, with bounds.

		groups: optional {label: [characteristic names]} for joint importance (e.g. value measures).
		method='ellipsoid': exact min / max over the Rashomon set. method='bootstrap': marginal interval,
		Bonferroni over the rows. identified = lower bound exceeds rel_tol of the reference total.
		"""
		pg = self.model.param_groups()
		total = self.theta @ self.phi_cov @ self.theta
		rows = []
		items = [(names[k], pg[k]) for k in range(len(names))]
		items += [('pair:%s x %s' % (names[u], names[v]), pg[('pair', q)]) for q, (u, v) in enumerate(getattr(self.model, 'pairs', []))]
		for label, members in (groups or {}).items():
			items.append(('group:' + label, sum((pg[names.index(m)] for m in members), [])))
		if method == 'bootstrap':
			self._require_boot()
		for label, idx in items:
			G = self._group_matrix(idx)
			if method == 'bootstrap':
				# percentile interval: importance is a non-negative quadratic, the pivotal one can go negative
				ref = self.theta @ G @ self.theta
				vb = ((self.boot_thetas @ G) * self.boot_thetas).sum(1)
				a = alpha / len(items)
				lo, hi = np.quantile(vb, [a / 2, 1 - a / 2])
			else:
				ref, lo, hi = self.quadratic_range(G)
			rows.append({'feature': label, 'importance': ref / total, 'min': max(lo, 0.0) / total, 'max': hi / total,
				'identified': lo > rel_tol * total})
		return pd.DataFrame(rows)

	def shape_band(self, k, s=None, grid=None):
		"""Shape function of characteristic k at state s with its exact Rashomon band (lo, hi)."""
		m = self.model
		grid = np.linspace(-0.5, 0.5, 51) if grid is None else np.asarray(grid)
		ref, lo, hi = [], [], []
		for z in grid:
			g = self._shape_functional(k, z, s)
			a, b, c = self.linear_range(g)
			ref.append(a); lo.append(b); hi.append(c)
		return grid, np.array(ref), np.array(lo), np.array(hi)

	def _shape_functional(self, k, z, s=None):
		"""g with g' theta = f_k(z; s) (centered shape value)."""
		m = self.model
		D = self.theta.size
		g = np.zeros(D)
		if isinstance(m, LinearSDF):
			g[1 + k] = z
			if m.use_state and s is not None:
				for p in range(m.P):
					g[1 + m.K + m.P + p * m.K + k] = s[p] * z
			return g
		nk = m.n_knots
		j, a = _hat_coords(torch.tensor([float(z)]), nk)
		basis = np.zeros(nk)
		basis[int(j)] += 1 - float(a)
		basis[int(j) + 1] += float(a)
		basis -= 1.0 / nk
		g[1 + k * nk:1 + (k + 1) * nk] = basis
		if m.state_varying and s is not None:
			off = 1 + m.K * nk + m.P
			for p in range(m.P):
				st = off + (p * m.K + k) * nk
				g[st:st + nk] = s[p] * basis
		return g

	def slope_table(self, names, s=None, method='ellipsoid', alpha=0.05):
		"""f_k(0.5) - f_k(-0.5): does the data identify the *direction* in which k moves SDF weights?"""
		if method == 'bootstrap':
			self._require_boot()
		rows = []
		for k, name in enumerate(names):
			g = self._shape_functional(k, 0.5, s) - self._shape_functional(k, -0.5, s)
			if method == 'bootstrap':
				ref = g @ self.theta
				lo, hi = self._boot_interval(self.boot_thetas @ g, ref, alpha / len(names))
			else:
				ref, lo, hi = self.linear_range(g)
			rows.append({'feature': name, 'slope': ref, 'lo': lo, 'hi': hi, 'sign_identified': lo > 0 or hi < 0})
		return pd.DataFrame(rows)

	def weight_ambiguity(self, batch):
		"""Stock-month SDF ambiguity.

		Returns A (T x N, half-width of the weight range divided by the cross-sectional std of reference
		weights; nan if unobserved), sign_ambiguous (T x N: the range contains zero) and the monthly index.
		"""
		T, N = batch['R'].shape
		A = np.full((T, N), np.nan)
		sign_amb = np.zeros((T, N), dtype=bool)
		with torch.no_grad():
			for t in range(T):
				m = batch['mask'][t].numpy()
				ph = self.model.features(batch, t).double().numpy()
				w = ph @ self.theta
				half = np.sqrt(self.r2 * ((ph @ self.Ainv) * ph).sum(1))
				A[t, m] = half / max(w.std(), 1e-12)
				sign_amb[t, m] = np.abs(w) < half
		return A, sign_amb, np.nanmean(A, axis=1)

	def sharpe_range(self, batch, n=2000, seed=0):
		"""Range of the (raw) SDF-portfolio Sharpe ratio across sampled members on the boundary."""
		Psi = self.model._cached_design(batch).double().numpy()
		th = self.sample(n, seed)
		F = th @ Psi.T
		sr = F.mean(1) / F.std(1)
		ref = (Psi @ self.theta)
		return ref.mean() / ref.std(), sr.min(), sr.max()


# ------------------------------------------------------------------------------------------------
# economic tests of SDF ambiguity
# ------------------------------------------------------------------------------------------------
def ambiguity_sorts(panel, A, M, n_groups=5, min_stocks=10):
	"""Monthly sorts on stock-level ambiguity: EW returns, alphas w.r.t. the SDF (market control)."""
	from .audit import alpha_moments, market_control, nw_tstats
	T = panel.T
	X = np.zeros((T, n_groups))
	for t in range(T):
		m = panel.mask[t] & np.isfinite(A[t])
		if m.sum() < n_groups * min_stocks:
			continue
		a, r = A[t, m], panel.R[t, m]
		q = np.quantile(a, np.linspace(0, 1, n_groups + 1))
		g = np.clip(np.searchsorted(q, a, side='right') - 1, 0, n_groups - 1)
		for k in range(n_groups):
			X[t, k] = r[g == k].mean() if (g == k).any() else 0.0
	X = np.column_stack([X, X[:, -1] - X[:, 0]])
	G = alpha_moments(X, M, market_control(panel))
	t_alpha, _ = nw_tstats(G)
	t_mean, _ = nw_tstats(X)
	labels = ['Q%d' % (k + 1) for k in range(n_groups)] + ['H-L']
	return pd.DataFrame({'mean': X.mean(0), 't_mean': t_mean, 'alpha': G.mean(0), 't_alpha': t_alpha}, index=labels)


def fama_macbeth(y, X, mask, names):
	"""Cross-sectional regressions of y (T x N) on X (T x N x J) each month; NW t-stats of the averages."""
	from .audit import nw_tstats
	T = y.shape[0]
	coefs = []
	for t in range(T):
		m = mask[t] & np.isfinite(y[t]) & np.all(np.isfinite(X[t]), axis=1)
		if m.sum() < X.shape[2] + 10:
			continue
		Xt = np.column_stack([np.ones(m.sum()), X[t, m]])
		c, *_ = np.linalg.lstsq(Xt, y[t, m], rcond=None)
		coefs.append(c)
	C = np.array(coefs)
	t, _ = nw_tstats(C)
	return pd.DataFrame({'coef': C.mean(0), 't': t}, index=['const'] + list(names))


def _standardize_cs(x, mask):
	out = np.full(x.shape, np.nan)
	for t in range(x.shape[0]):
		m = mask[t] & np.isfinite(x[t])
		if m.sum() > 1:
			out[t, m] = (x[t, m] - x[t, m].mean()) / max(x[t, m].std(), 1e-12)
	return out


def ambiguity_regressions(panel, A, controls=None, outcome=None, extra_controls=None, squares=True):
	"""Fama-MacBeth of returns (default) or of panel.extra[outcome] on standardized ambiguity.

	controls: characteristic names (default: all). extra_controls: {name: T x N array}, e.g. the reference
	SDF weight and its absolute value -- ambiguity is largest where weights are extreme, so without them
	ambiguity can proxy for the model's own pricing signal. squares=True adds squared characteristics:
	ambiguity is high for extreme characteristics, so with linear controls only it picks up nonlinear
	characteristic premia (synthetic data with no planted effect: t = 3.6 linear, -0.2 with squares).
	The ambiguity and extra regressors are standardized each month.
	"""
	controls = panel.char_names if controls is None else controls
	cols, names = [_standardize_cs(A, panel.mask)], ['ambiguity']
	for name, x in (extra_controls or {}).items():
		cols.append(_standardize_cs(x, panel.mask))
		names.append(name)
	Zc = panel.Z[:, :, [panel.char_idx(c) for c in controls]]
	blocks, cnames = [np.stack(cols, axis=2), Zc], names + list(controls)
	if squares:
		blocks.append(Zc ** 2)
		cnames += [c + '^2' for c in controls]
	X = np.concatenate(blocks, axis=2)
	y = panel.R if outcome is None else panel.extra[outcome]
	return fama_macbeth(y, X, panel.mask, cnames)


def ambiguity_index_regressions(panel, index):
	"""Correlation of the monthly ambiguity index with each macro series."""
	rows = [{'macro': name, 'corr': np.corrcoef(index, panel.macro[:, j])[0, 1]} for j, name in enumerate(panel.macro_names)]
	return pd.DataFrame(rows).sort_values('corr', key=np.abs, ascending=False)
