"""The complexity ladder and the price of interpretability (Paper 1, part A).

Fit a nested sequence of SDFs on the same data, against the same kind of adversary:
	linear -> linear + state -> additive -> additive + state -> + screened pairs -> neural (CPZ-style)
and measure what each step buys out of sample: Sharpe ratio, EV / XS-R2, and whether the most complex
SDF portfolio earns alpha the simpler one cannot span (spanning test). The headline number is the
squared-Sharpe gap SR(full)^2 - SR(rung)^2: the price of interpretability.
"""
import itertools

import numpy as np
import pandas as pd
import torch

from .adversary import NeuralAdversary
from .audit import alpha_moments, market_control, nw_tstats
from .pricing import spanning_test, sharpe
from .sdf import build_sdf
from .train import TrainConfig, evaluate, fit_gan

DEFAULT_RUNGS = [
	{'name': 'linear', 'type': 'linear', 'state_varying': False},
	{'name': 'linear_state', 'type': 'linear', 'state_varying': True},
	{'name': 'additive', 'type': 'additive', 'state_varying': False},
	{'name': 'additive_state', 'type': 'additive', 'state_varying': True},
	{'name': 'additive_state_pairs', 'type': 'additive', 'state_varying': True, 'pairs': 'screen'},
	{'name': 'neural', 'type': 'neural'},
]


def screen_pairs(panel, M, top=5, chars=None, controls='market'):
	"""Interactions the current SDF leaves unpriced.

	For every pair (j, k) form the managed portfolio x_t = mean_i z_tij z_tik R_ti and rank pairs by the
	|t| of its alpha relative to SDF M. Returns [(j, k, t)], top first.
	"""
	idx = range(panel.K) if chars is None else [panel.char_idx(c) for c in chars]
	pairs = list(itertools.combinations(idx, 2))
	X = np.zeros((panel.T, len(pairs)))
	for t in range(panel.T):
		m = panel.mask[t]
		Zt, Rt = panel.Z[t, m], panel.R[t, m]
		for q, (j, k) in enumerate(pairs):
			X[t, q] = np.mean(Zt[:, j] * Zt[:, k] * Rt)
	ctrl = market_control(panel) if controls == 'market' else None
	tstat, _ = nw_tstats(alpha_moments(X, M, ctrl))
	order = np.argsort(-np.abs(tstat))[:top]
	return [(pairs[q][0], pairs[q][1], float(tstat[q])) for q in order]


def _ensemble_eval(models, batch):
	"""Average raw weights over seeds (as CPZ do for their 9 trials), then evaluate."""
	from .pricing import normalized_weights, sharpe as _sr, explained_variation
	for m in models:
		m.eval()	# reloaded models start in train mode (dropout on)
	with torch.no_grad():
		w = torch.stack([m(batch) for m in models]).mean(0)
		M = torch.stack([m.sdf(batch) for m in models]).mean(0)
	wn = normalized_weights(w, batch['mask'])
	F = (wn * batch['R']).sum(1).numpy()
	ev, xs = explained_variation(w.numpy(), batch['R'].numpy(), batch['mask'].numpy())
	return {'sharpe': _sr(F), 'F': F, 'M': M.numpy(), 'ev': ev, 'xs_r2': xs, 'sharpe_raw': _sr(1 - M.numpy())}


def run_ladder(train, valid, test, state_vars, rungs=DEFAULT_RUNGS, cfg=TrainConfig(), n_seeds=1,
		n_pairs=5, adversary_moments=8, verbose=True, prefitted=None):
	"""Fit every rung; return (results DataFrame, {name: [models]}, {name: test evaluation}).

	prefitted: {rung name: {'states': [state_dict per seed], 'pairs': [(j, k)] or None}} reloads rungs
	fitted earlier instead of refitting them (e.g. add the slow neural rung to a finished run).
	"""
	prefitted = prefitted or {}
	b_tr, b_va, b_te = [p.to_torch(state_vars) for p in (train, valid, test)]
	K, P, Md = train.K, len(state_vars), len(train.macro_names)
	fitted, evals, rows = {}, {}, []
	pairs_used = None
	for rung in rungs:
		spec = dict(rung)
		pre = prefitted.get(rung['name'])
		if pre is not None and spec.get('pairs') == 'screen':
			spec['pairs'] = [tuple(p) for p in pre['pairs']]
		if spec.get('pairs') == 'screen':
			base = fitted.get('additive_state') or fitted.get('additive')
			if base is None:
				raise ValueError('pair screening needs an additive rung before it')
			M_base = _ensemble_eval(base, b_tr)['M']
			screened = screen_pairs(train, M_base, top=n_pairs)
			pairs_used = [(j, k) for j, k, _ in screened]
			spec['pairs'] = pairs_used
		models = []
		for state in (pre['states'] if pre is not None else []):
			model = build_sdf(spec, K, P, Md)
			model.load_state_dict(state)
			models.append(model)
		for s in range(0 if pre is not None else n_seeds):
			torch.manual_seed(s)
			model = build_sdf(spec, K, P, Md)
			adv = NeuralAdversary(K, Md, adversary_moments)
			c = TrainConfig(**{**cfg.__dict__, 'seed': s})
			fit_gan(model, adv, b_tr, b_va, c)
			models.append(model)
		fitted[rung['name']] = models
		ev = [_ensemble_eval(models, b) for b in (b_tr, b_va, b_te)]
		evals[rung['name']] = ev[2]
		row = {'model': rung['name'], 'n_params': sum(p.numel() for p in models[0].parameters()),
			'SR_train': ev[0]['sharpe'], 'SR_valid': ev[1]['sharpe'], 'SR_test': ev[2]['sharpe'],
			'EV_test': ev[2]['ev'], 'XS_R2_test': ev[2]['xs_r2']}
		if spec.get('pairs'):
			row['pairs'] = ', '.join('%s x %s' % (train.char_names[j], train.char_names[k]) for j, k in spec['pairs'])
		rows.append(row)
		if verbose:
			print('[ladder] %-22s SR train/valid/test %.3f / %.3f / %.3f' % (rung['name'], row['SR_train'], row['SR_valid'], row['SR_test']))
	df = pd.DataFrame(rows)
	return price_of_interpretability(df, evals), fitted, evals


def price_of_interpretability(df, evals, reference=None):
	"""Add spanning-test columns: does the reference (most complex) SDF portfolio have alpha on each rung?"""
	reference = df['model'].iloc[-1] if reference is None else reference
	F_ref = evals[reference]['F']
	alphas, ts, r2s = [], [], []
	for name in df['model']:
		a, t, r2 = spanning_test(F_ref, evals[name]['F'])
		alphas.append(a); ts.append(t); r2s.append(r2)
	df = df.copy()
	df['SR2_gap_vs_' + reference] = sharpe(F_ref) ** 2 - df['SR_test'] ** 2
	df['alpha_of_' + reference] = alphas
	df['t_alpha_of_' + reference] = ts
	df['R2_span_' + reference] = r2s
	return df
