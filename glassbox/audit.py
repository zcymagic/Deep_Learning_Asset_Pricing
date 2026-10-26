"""Glass-box auditor: find, test and classify where a candidate SDF fails to price.

A test asset is a readable rule: up to two characteristic conditions (which stocks) and at most one
macro-state condition (when the portfolio is held). Its managed portfolio return is the equal-weighted
excess return of the selected stocks, x_t (0 when the rule is inactive). A correct SDF has E[M x] = 0.

Pricing errors are measured as the time-series alpha of x on the SDF portfolio F = 1 - M, the
equal-weighted market (a control the SDF should price; it absorbs factor exposure that a time-varying
SDF portfolio leaves unspanned under a constant-beta regression) and any portfolios already added in
stepwise search. If M prices F, E[M x] = 0 <=> alpha = 0, but the alpha
removes the factor noise in M x and is far more powerful; out of sample it asks whether x expands the
Sharpe ratio of the SDF portfolio (Barillas-Shanken logic).

Workflow (valid inference through sample splitting):
	discover(panel_A, M_A)          beam search over the rule class on the discovery sample
	confirm(panel_B, M_B, rules)    out-of-sample t-stats, Holm and bootstrap max-t p-values
	global_test(panel, M)           one fixed-class test of "the SDF prices every rule" (max-t bootstrap)
	classify / failure_map          limits-to-arbitrage (mispricing-type) vs state-dependent (risk-type)
	augment_sdf                     add a discovered portfolio to the SDF and re-test (missing factor?)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing import factor_sdf_coefficients

# Trading-friction characteristics in the CPZ data (subset of their "Trading Frictions" category,
# excluding market betas). Rules built only from these are the limits-to-arbitrage region.
DEFAULT_LTA = ('LME', 'Spread', 'IdioVol', 'Resid_Var', 'LTurnover', 'SUV', 'Variance', 'Rel2High', 'AT')


@dataclass(frozen=True)
class Condition:
	feature: str
	kind: str		# 'char' or 'state'
	op: str			# '>' or '<'
	threshold: float

	def __str__(self):
		return '%s %s %.2f' % (self.feature, self.op, self.threshold)


@dataclass(frozen=True)
class Rule:
	conds: tuple

	@property
	def char_conds(self):
		return [c for c in self.conds if c.kind == 'char']

	@property
	def state_conds(self):
		return [c for c in self.conds if c.kind == 'state']

	def __str__(self):
		return ' AND '.join(str(c) for c in self.conds) if self.conds else 'ALL STOCKS'


# ------------------------------------------------------------------------------------------------
# statistics helpers
# ------------------------------------------------------------------------------------------------
def nw_tstats(G, lags=None):
	"""Newey-West t-statistics of the column means of G (T x J). Returns (t, se)."""
	T = G.shape[0]
	if lags is None:
		lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
	E = G - G.mean(axis=0)
	s = (E * E).sum(axis=0) / T
	for l in range(1, lags + 1):
		s += 2 * (1 - l / (lags + 1)) * (E[l:] * E[:-l]).sum(axis=0) / T
	se = np.sqrt(np.maximum(s, 1e-20) / T)
	return G.mean(axis=0) / se, se


def stationary_bootstrap_indices(T, n_boot, mean_block, rng):
	"""Politis-Romano stationary bootstrap time indices, n_boot x T."""
	idx = np.empty((n_boot, T), dtype=int)
	p = 1.0 / mean_block
	for b in range(n_boot):
		i = rng.integers(T)
		for t in range(T):
			idx[b, t] = i
			i = rng.integers(T) if rng.random() < p else (i + 1) % T
	return idx


def max_t_pvalues(G, signs=None, n_boot=1000, mean_block=6, seed=0):
	"""Single-step bootstrap max-t p-values (White reality check / Romano-Wolf style).

	signs: +1/-1 per column for one-sided tests in a pre-specified direction; None = two-sided.
	"""
	rng = np.random.default_rng(seed)
	t, se = nw_tstats(G)
	stat = np.abs(t) if signs is None else signs * t
	Gc = G - G.mean(axis=0)
	idx = stationary_bootstrap_indices(G.shape[0], n_boot, mean_block, rng)
	maxes = np.empty(n_boot)
	for b in range(n_boot):
		tb = Gc[idx[b]].mean(axis=0) / se
		maxes[b] = (np.abs(tb) if signs is None else signs * tb).max()
	p = np.array([(maxes >= s).mean() for s in stat])
	return p, maxes


def holm(p):
	p = np.asarray(p)
	order = np.argsort(p)
	adj = np.empty_like(p, dtype=float)
	running = 0.0
	m = len(p)
	for rank, j in enumerate(order):
		running = max(running, min(1.0, (m - rank) * p[j]))
		adj[j] = running
	return adj


def alpha_moments(X, M, extra=None):
	"""Influence-function series of the alphas from regressing each column of X on [1, 1 - M, extra].

	alpha_hat = sum_t a_t x_t with a = Fm (Fm'Fm)^-1 e_0, so g_t = alpha_hat + T a_t e_t has mean alpha_hat
	and its Newey-West variance is the HAC sandwich variance of alpha_hat, including the estimation error
	of the slopes (important once high-Sharpe portfolios are added in stepwise search). Column means are
	the alphas, so nw_tstats(alpha_moments(...)) gives alpha t-statistics.
	"""
	T = len(M)
	cols = [np.ones(T), 1.0 - np.asarray(M, dtype=float)]
	if extra is not None:
		cols.append(np.asarray(extra).reshape(T, -1))
	Fm = np.column_stack(cols)
	XtX_inv = np.linalg.pinv(Fm.T @ Fm)
	coef = XtX_inv @ Fm.T @ X
	E = X - Fm @ coef
	a = Fm @ XtX_inv[:, 0]
	return coef[0][None, :] + T * a[:, None] * E


# ------------------------------------------------------------------------------------------------
# rule portfolios
# ------------------------------------------------------------------------------------------------
def _cond_mask_char(Zt, char_names, cond):
	z = Zt[:, char_names.index(cond.feature)]
	return z > cond.threshold if cond.op == '>' else z < cond.threshold


def _cond_mask_state(panel, cond):
	s = panel.macro[:, panel.macro_names.index(cond.feature)]
	return s > cond.threshold if cond.op == '>' else s < cond.threshold


def rule_returns(panel, rules, min_stocks=20):
	"""T x J equal-weighted managed-portfolio returns for arbitrary rules (0 when inactive)."""
	X = np.zeros((panel.T, len(rules)))
	active = np.zeros((panel.T, len(rules)), dtype=bool)
	n_stocks = np.zeros((panel.T, len(rules)))
	state_on = np.ones((panel.T, len(rules)), dtype=bool)
	for j, rule in enumerate(rules):
		for c in rule.state_conds:
			state_on[:, j] &= _cond_mask_state(panel, c)
	for t in range(panel.T):
		m = panel.mask[t]
		Zt, Rt = panel.Z[t, m], panel.R[t, m]
		for j, rule in enumerate(rules):
			if not state_on[t, j]:
				continue
			sel = np.ones(len(Rt), dtype=bool)
			for c in rule.char_conds:
				sel &= _cond_mask_char(Zt, panel.char_names, c)
			n = sel.sum()
			n_stocks[t, j] = n
			if n >= min_stocks:
				X[t, j] = Rt[sel].mean()
				active[t, j] = True
	return X, active, n_stocks


def market_control(panel):
	"""Equal-weighted market excess return, T x 1."""
	n = np.maximum(panel.mask.sum(axis=1), 1)
	return ((panel.R * panel.mask).sum(axis=1) / n)[:, None]


def _controls(panel, controls, found=None):
	cols = []
	if controls == 'market':
		cols.append(market_control(panel))
	elif controls is not None:
		cols.append(np.asarray(controls).reshape(panel.T, -1))
	if found is not None and len(found):
		cols.append(np.column_stack(found))
	return np.column_stack(cols) if cols else None


def rule_moments(panel, M, rules, min_stocks=20, controls='market', found=None):
	"""Pricing-error series (alpha + residual) of each rule portfolio under SDF M."""
	X, active, n_stocks = rule_returns(panel, rules, min_stocks)
	return alpha_moments(X, M, _controls(panel, controls, found)), active, n_stocks


# ------------------------------------------------------------------------------------------------
# discovery
# ------------------------------------------------------------------------------------------------
def candidate_conditions(panel, state_vars, char_thresholds=(-0.3, -0.1, 0.1, 0.3),
		state_quantiles=(0.2, 0.5, 0.8), chars=None):
	chars = panel.char_names if chars is None else chars
	char_conds = []
	for name in chars:
		for q in char_thresholds:
			char_conds.append(Condition(name, 'char', '<' if q < 0 else '>', q))
	state_conds = []
	for name in state_vars:
		s = panel.macro[:, panel.macro_names.index(name)]
		for q in state_quantiles:
			v = float(np.quantile(s, q))
			if q <= 0.5:
				state_conds.append(Condition(name, 'state', '<', v))
			if q >= 0.5:
				state_conds.append(Condition(name, 'state', '>', v))
	return char_conds, state_conds


def _stock_rule_returns(panel, char_conds, base_pairs, min_stocks):
	"""Returns for: ALL, each single char condition, and each (base, extension) pair. T x J."""
	Jc = len(char_conds)
	k_idx = np.array([panel.char_names.index(c.feature) for c in char_conds])
	thr = np.array([c.threshold for c in char_conds])
	gt = np.array([c.op == '>' for c in char_conds])
	nb = len(base_pairs)
	X = np.zeros((panel.T, 1 + Jc + nb * Jc))
	for t in range(panel.T):
		m = panel.mask[t]
		Zt, Rt = panel.Z[t, m], panel.R[t, m]
		zc = Zt[:, k_idx]
		C = np.where(gt, zc > thr, zc < thr).astype(np.float64)	# n x Jc
		X[t, 0] = Rt.mean() if len(Rt) >= min_stocks else 0.0
		num, den = Rt @ C, C.sum(axis=0)
		X[t, 1:1 + Jc] = np.where(den >= min_stocks, num / np.maximum(den, 1), 0.0)
		if nb:
			Bm = C[:, base_pairs]
			num2 = (Bm * Rt[:, None]).T @ C
			den2 = Bm.T @ C
			X[t, 1 + Jc:] = np.where(den2 >= min_stocks, num2 / np.maximum(den2, 1), 0.0).ravel()
	return X


def _rank_candidates(panel, M, char_conds, state_masks, beam, min_stocks, complexity_penalty, n_keep, extra=None):
	"""Score every rule in the class (beam-pruned at level 2) under SDF M. Returns rows, X, stock_rules."""
	Jc = len(char_conds)
	# level 1: single characteristic conditions (and ALL), crossed with each state condition
	X1 = _stock_rule_returns(panel, char_conds, [], min_stocks)
	best1 = np.zeros(X1.shape[1])
	for sm in state_masks:
		t, _ = nw_tstats(alpha_moments(X1 * sm[:, None], M, extra))
		best1 = np.maximum(best1, np.abs(t))
	base = [j - 1 for j in np.argsort(-best1) if j > 0][:beam]

	# level 2: extend the beam with a second characteristic condition on a different feature
	X = _stock_rule_returns(panel, char_conds, base, min_stocks)
	stock_rules = [()] + [(j,) for j in range(Jc)]
	for b in base:
		for j in range(Jc):
			stock_rules.append((b, j))
	seen, uniq = set(), []
	for i, r in enumerate(stock_rules):
		if len(r) == 2 and char_conds[r[0]].feature == char_conds[r[1]].feature:
			continue
		key = tuple(sorted(r))
		if key not in seen:
			seen.add(key)
			uniq.append(i)
	X = X[:, uniq]
	stock_rules = [stock_rules[i] for i in uniq]

	n_conds = np.array([len(r) for r in stock_rules])
	rows = []
	for si, sm in enumerate(state_masks):
		Xs = X * sm[:, None]
		t, _ = nw_tstats(alpha_moments(Xs, M, extra))
		score = np.abs(t) - complexity_penalty * np.maximum(n_conds + (si > 0) - 1, 0)
		score[Xs.std(axis=0) == 0] = -np.inf
		for j in np.argsort(-score)[:n_keep]:
			rows.append((score[j], t[j], j, si))
	rows.sort(key=lambda r: -r[0])
	return rows, X, stock_rules


def discover(panel, M, state_vars, beam=30, top=15, min_stocks=20, complexity_penalty=0.5,
		stepwise=True, stop_t='auto', max_corr=0.7, controls='market',
		char_thresholds=(-0.3, -0.1, 0.1, 0.3), state_quantiles=(0.2, 0.5, 0.8), chars=None):
	"""Search for the readable test assets with the largest pricing errors under SDF M.

	Rules are ranked by |t| - complexity_penalty * (number of conditions - 1), so an extra condition
	must raise the t-statistic to enter (parsimony; keeps rules readable and hold-out power high).

	stepwise=True (default): forward selection. After each rule is found its portfolio is added to the
	pricing factors and the search is repeated, so every later rule is a failure *not explained by the
	earlier ones*; stops when the best |t| falls below stop_t. stop_t='auto' is the Bonferroni critical
	value for the size of the searched class (5%, two-sided), the level the maximum noise |t| reaches.
	stepwise=False: one pass, keeping rules whose moment series correlate less than max_corr.
	"""
	char_conds, state_conds = candidate_conditions(panel, state_vars, char_thresholds, state_quantiles, chars)
	state_masks = [np.ones(panel.T, dtype=bool)] + [_cond_mask_state(panel, c) for c in state_conds]

	def make_rule(j, si, stock_rules):
		conds = tuple(char_conds[k] for k in stock_rules[j])
		return Rule(conds + ((state_conds[si - 1],) if si > 0 else ()))

	if stop_t == 'auto':
		from scipy.stats import norm
		n_class = (1 + len(char_conds) * (1 + beam)) * len(state_masks)
		stop_t = float(norm.ppf(1 - 0.025 / n_class))

	rules, t_step = [], []
	if stepwise:
		X_found = []
		for _ in range(top):
			extra = _controls(panel, controls, X_found)
			rows, _, stock_rules = _rank_candidates(panel, M, char_conds, state_masks, beam, min_stocks,
				complexity_penalty, 1, extra)
			_, tval, j, si = rows[0]
			rule = make_rule(j, si, stock_rules)
			if abs(tval) < stop_t or rule in rules:
				break
			rules.append(rule)
			t_step.append(tval)
			X_found.append(rule_returns(panel, [rule], min_stocks)[0][:, 0])
	else:
		extra = _controls(panel, controls)
		rows, X, stock_rules = _rank_candidates(panel, M, char_conds, state_masks, beam, min_stocks,
			complexity_penalty, top * 20, extra)
		cols = []
		for _, tval, j, si in rows:
			g = alpha_moments(X[:, [j]] * state_masks[si][:, None], M, extra)[:, 0]
			if any(abs(np.corrcoef(g, c)[0, 1]) > max_corr for c in cols):
				continue
			rules.append(make_rule(j, si, stock_rules))
			t_step.append(tval)
			cols.append(g)
			if len(rules) >= top:
				break

	Gs, active, n_stocks = rule_moments(panel, M, rules, min_stocks, controls)
	t, _ = nw_tstats(Gs) if rules else (np.zeros(0), None)
	return pd.DataFrame({
		'rule': rules,
		'description': [str(r) for r in rules],
		't_discovery': t,
		't_stepwise': t_step,
		'alpha': Gs.mean(axis=0),
		'alpha_when_active': Gs.mean(axis=0) / np.maximum(active.mean(axis=0), 1e-12),
		'share_active': active.mean(axis=0),
		'avg_stocks': n_stocks.sum(axis=0) / np.maximum(active.sum(axis=0), 1),
	})


# ------------------------------------------------------------------------------------------------
# confirmation and global test
# ------------------------------------------------------------------------------------------------
def state_dependence(panel, M, rules, state_vars, min_stocks=20, controls='market'):
	"""Does each rule's pricing error vary with the macro state? Regress the alpha series on [1, s_p].

	Returns (best state variable, its slope t-stat) per rule, choosing the largest |t| over state_vars.
	A characteristic-only rule whose alpha loads on a state is a state-dependent (risk-type) failure even
	though the search did not need the state condition to find it.
	"""
	G, _, _ = rule_moments(panel, M, rules, min_stocks, controls)
	best_var, best_t = [], []
	for j in range(G.shape[1]):
		bv, bt = None, 0.0
		for v in state_vars:
			s = panel.macro[:, panel.macro_names.index(v)]
			X = np.column_stack([np.ones(panel.T), s])
			coef, *_ = np.linalg.lstsq(X, G[:, j], rcond=None)
			e = G[:, j] - X @ coef
			XtX_inv = np.linalg.inv(X.T @ X)
			infl = panel.T * (X @ XtX_inv[:, 1]) * e + coef[1]
			t = nw_tstats(infl[:, None])[0][0]
			if abs(t) > abs(bt):
				bv, bt = v, t
		best_var.append(bv)
		best_t.append(bt)
	return best_var, np.array(best_t)


def confirm(panel, M, discovered, n_boot=1000, mean_block=6, seed=0, min_stocks=20, controls='market',
		state_vars=None):
	"""Re-test discovered rules on a hold-out sample, one-sided in the discovered direction.

	Also reports t_holdout_stepwise: the hold-out alpha of rule k after controlling for the SDF portfolio
	and rules 1..k-1 (is it a failure beyond the earlier ones out of sample too?), and, with state_vars,
	the hold-out state-dependence test of each rule's alpha.
	"""
	from scipy.stats import norm
	rules = list(discovered['rule'])
	G, active, _ = rule_moments(panel, M, rules, min_stocks, controls)
	t, _ = nw_tstats(G)
	signs = np.sign(discovered['t_discovery'].values)
	if len(rules) > 0:
		XB = rule_returns(panel, rules, min_stocks)[0]
		t_sw = [t[0]]
		for k in range(1, len(rules)):
			t_sw.append(nw_tstats(alpha_moments(XB[:, [k]], M, _controls(panel, controls, [XB[:, :k]])))[0][0])
		t_sw = np.array(t_sw)
	p_raw = 1 - norm.cdf(signs * t)
	p_boot, _ = max_t_pvalues(G, signs=signs, n_boot=n_boot, mean_block=mean_block, seed=seed)
	out = discovered.copy()
	out['t_holdout'] = t
	out['alpha_holdout'] = G.mean(axis=0)
	out['p_holdout'] = p_raw
	out['p_holm'] = holm(p_raw)
	out['p_maxt'] = p_boot
	out['share_active_holdout'] = active.mean(axis=0)
	if len(rules) > 0:
		out['t_holdout_stepwise'] = t_sw
		out['p_holdout_stepwise'] = 1 - norm.cdf(np.sign(discovered['t_stepwise'].values) * t_sw)
	if state_vars:
		out['state_dep_var'], out['t_state_dep'] = state_dependence(panel, M, rules, state_vars, min_stocks, controls)
		out.attrs['n_state_vars'] = len(state_vars)
	return out


def fixed_rule_class(panel, state_vars, char_thresholds=(-0.3, 0.3), state_quantiles=(0.2, 0.8), chars=None):
	"""Data-independent rule class for the global test: single char conditions x (no state / one state)."""
	char_conds, state_conds = candidate_conditions(panel, state_vars, char_thresholds, state_quantiles, chars)
	rules = [Rule((c,)) for c in char_conds]
	rules += [Rule((s,)) for s in state_conds]
	rules += [Rule((c, s)) for c in char_conds for s in state_conds]
	return rules


def global_test(panel, M, rules, n_boot=1000, mean_block=6, seed=0, min_stocks=20, controls='market'):
	"""Test H0: E[M x_j] = 0 for every rule j (two-sided max-|t| bootstrap)."""
	G, _, _ = rule_moments(panel, M, rules, min_stocks, controls)
	t, _ = nw_tstats(G)
	keep = G.std(axis=0) > 0
	p, maxes = max_t_pvalues(G[:, keep], n_boot=n_boot, mean_block=mean_block, seed=seed)
	return {'max_abs_t': float(np.abs(t[keep]).max()), 'p_value': float(p[np.argmax(np.abs(t[keep]))]),
		'n_rules': int(keep.sum()), 'rms_alpha': float(np.sqrt((G.mean(axis=0) ** 2).mean()))}


# ------------------------------------------------------------------------------------------------
# economic classification
# ------------------------------------------------------------------------------------------------
def classify(rule, lta_chars=DEFAULT_LTA, state_dependent=False):
	"""Map a rule to the risk-vs-mispricing taxonomy.

	limits-to-arbitrage : only trading-friction characteristics, no state dependence -> mispricing-type
	state-dependent     : state condition or state-dependent alpha, no friction char -> risk-type
	characteristic      : non-friction characteristics, no state dependence          -> missing characteristic premium
	mixed               : everything else
	state_dependent=True marks a rule whose hold-out alpha significantly loads on a macro state.
	"""
	chars = [c.feature for c in rule.char_conds]
	has_state = len(rule.state_conds) > 0 or state_dependent
	n_lta = sum(f in lta_chars for f in chars)
	if not has_state and chars and n_lta == len(chars):
		return 'limits-to-arbitrage'
	if has_state and n_lta == 0:
		return 'state-dependent'
	if not has_state and n_lta == 0 and chars:
		return 'characteristic'
	return 'mixed'


def failure_map(confirmed, alpha=0.05, p_col='p_maxt', lta_chars=DEFAULT_LTA):
	"""Share of significant out-of-sample pricing failures (by squared t) in each class.

	If confirm() ran the state-dependence test, a rule counts as state-dependent when its slope t passes
	a Bonferroni bound over the state variables tried (two-sided, level alpha).
	"""
	from scipy.stats import norm
	df = confirmed.copy()
	if 't_state_dep' in df:
		n_states = max(1, int(df.attrs.get('n_state_vars', 1)))
		crit = norm.ppf(1 - alpha / (2 * n_states))
		dep = (df['t_state_dep'].abs() > crit).values
	else:
		dep = np.zeros(len(df), dtype=bool)
	df['class'] = [classify(r, lta_chars, d) for r, d in zip(df['rule'], dep)]
	sig = df[df[p_col] < alpha]
	classes = ['limits-to-arbitrage', 'state-dependent', 'characteristic', 'mixed']
	tot = (sig['t_holdout'] ** 2).sum()
	share = {c: float((sig.loc[sig['class'] == c, 't_holdout'] ** 2).sum() / tot) if tot > 0 else 0.0 for c in classes}
	count = {c: int((sig['class'] == c).sum()) for c in classes}
	return df, pd.DataFrame({'n_significant': count, 'share_of_t2': share})


def augment_sdf(M_A, x_A, M_B, x_B):
	"""Add managed portfolios x to an SDF of the CPZ form M = 1 - F (F = 1 - M), estimated on sample A.

	Returns the augmented SDF on sample B: M_aug = 1 - b1 F - b2' x with b = E[ff']^-1 E[f] from A.
	"""
	fA = np.column_stack([1.0 - M_A, x_A])
	fB = np.column_stack([1.0 - M_B, x_B])
	b = factor_sdf_coefficients(fA)
	return 1.0 - fB @ b
