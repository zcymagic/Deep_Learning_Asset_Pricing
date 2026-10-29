"""Robustness of Papers 1 and 2 to the audit's rule thresholds and the glass-box SDF's knots.

	python scripts/run_knot_threshold_robustness.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/knot_threshold_cpz

Part A, rule thresholds: the fitted SDFs (glass-box rungs reloaded from --models, factor models, external
SDFs) are re-audited with alternative characteristic cut-points, state-variable quantiles and minimum
portfolio sizes; the fixed rule class of the global test changes with each grid.
Part B, knots: the additive SDF is refit with each knot count (Paper 1: Sharpe ratios and audit) and
the Paper 2 closed-form analysis is repeated (shrinkage chosen on validation, identification at
eps = 1% and by bootstrap).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glassbox import audit
from glassbox.adversary import NeuralAdversary
from glassbox.data import concat_time, load_config, load_splits
from glassbox.ladder import _ensemble_eval, run_ladder
from glassbox.rashomon import RashomonEllipsoid, select_shrinkage
from glassbox.report import md_table
from glassbox.sdf import build_sdf
from glassbox.train import TrainConfig, evaluate, fit_gan
from run_paper1 import external_sdf, factor_model_sdf

THRESHOLD_GRIDS = {
	'baseline': {'char': (-0.3, -0.1, 0.1, 0.3), 'state': (0.2, 0.5, 0.8), 'fixed_char': (-0.3, 0.3), 'fixed_state': (0.2, 0.8), 'min_stocks': None},
	'coarse': {'char': (-0.3, 0.3), 'state': (0.2, 0.8), 'fixed_char': (-0.3, 0.3), 'fixed_state': (0.2, 0.8), 'min_stocks': None},
	'fine': {'char': (-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4), 'state': (0.1, 0.25, 0.5, 0.75, 0.9),
		'fixed_char': (-0.4, -0.2, 0.2, 0.4), 'fixed_state': (0.1, 0.9), 'min_stocks': None},
	'extreme': {'char': (-0.4, -0.2, 0.2, 0.4), 'state': (0.1, 0.5, 0.9), 'fixed_char': (-0.4, 0.4), 'fixed_state': (0.1, 0.9), 'min_stocks': None},
	'min_stocks_50': {'char': (-0.3, -0.1, 0.1, 0.3), 'state': (0.2, 0.5, 0.8), 'fixed_char': (-0.3, 0.3), 'fixed_state': (0.2, 0.8), 'min_stocks': 50},
}
RUNG_SPECS = {
	'linear': {'type': 'linear', 'state_varying': False},
	'additive': {'type': 'additive', 'state_varying': False},
	'additive_state': {'type': 'additive', 'state_varying': True},
}


def run_audit(name, M_tr, M_ho, train, holdout, cfg_a, grid, state_vars, lta):
	min_stocks = grid['min_stocks'] or cfg_a['min_stocks']
	disc = audit.discover(train, M_tr, state_vars, beam=cfg_a['beam'], top=cfg_a['top'], min_stocks=min_stocks,
		char_thresholds=grid['char'], state_quantiles=grid['state'])
	fixed = audit.fixed_rule_class(train, state_vars, grid['fixed_char'], grid['fixed_state'])
	glob = audit.global_test(holdout, M_ho, fixed, n_boot=cfg_a['n_boot'], min_stocks=min_stocks)
	row = {'model': name, 'max_abs_t': glob['max_abs_t'], 'global_p': glob['p_value'], 'n_rules': len(disc),
		'n_confirmed': 0, 'top_rule': '', 't_holdout': np.nan, 'p_maxt': np.nan, 'class': '', 'reversal_rule_confirmed': False}
	if len(disc):
		conf = audit.confirm(holdout, M_ho, disc, n_boot=cfg_a['n_boot'], min_stocks=min_stocks, state_vars=state_vars)
		table, _ = audit.failure_map(conf, lta_chars=lta)
		sig = table[table['p_maxt'] < 0.05]
		row['n_confirmed'] = len(sig)
		pick = sig if len(sig) else table
		top = pick.iloc[int(np.argmax(np.abs(pick['t_holdout'].values)))]
		row.update({'top_rule': top['description'], 't_holdout': top['t_holdout'], 'p_maxt': top['p_maxt'], 'class': top['class'],
			'reversal_rule_confirmed': bool(any('ST_REV' in d for d in sig['description']))})
	return row


def load_rung(models_dir, rung, train, state_vars):
	models = []
	for st in torch.load(os.path.join(models_dir, 'model_%s.pt' % rung)):
		m = build_sdf(RUNG_SPECS[rung], train.K, len(state_vars), len(train.macro_names))
		m.load_state_dict(st)
		models.append(m)
	return models


def identification(cfg, train, valid, test, n_knots, n_boot, value_group):
	"""Paper 2 closed-form analysis of the additive SDF with n_knots (one GAN fit for the instruments)."""
	rc = cfg.get('rashomon', {})
	state_vars = cfg['state_vars']
	spec = {'type': 'additive', 'state_varying': False, 'lam_group': 0.0, 'lam_smooth': rc.get('lam_smooth', 1e-3), 'n_knots': n_knots}
	b_tr, b_va, b_te = [p.to_torch(state_vars) for p in (train, valid, test)]
	torch.manual_seed(0)
	model = build_sdf(spec, train.K, len(state_vars), len(train.macro_names))
	adv = NeuralAdversary(train.K, len(train.macro_names), 8)
	fit_gan(model, adv, b_tr, b_va, TrainConfig(**cfg.get('train', {})))
	adv.eval()
	with torch.no_grad():
		h = adv(b_tr)
	shrink, _ = select_shrinkage(model, b_tr, h, b_va, grid=rc.get('shrink_grid', (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)))
	E = RashomonEllipsoid(model, b_tr, h, eps=0.01, n_boot=n_boot, shrink=shrink)
	names = train.char_names
	groups = {'value': [c for c in value_group if c in names]}
	out = {'n_knots': n_knots, 'n_params': E.theta.size, 'shrink': shrink, 'test_sharpe': evaluate(model, b_te)['sharpe']}
	for method in ('ellipsoid', 'bootstrap'):
		imp = E.importance_table(names, groups=groups, method=method).set_index('feature')
		slo = E.slope_table(names, method=method).set_index('feature')
		tag = 'eps1' if method == 'ellipsoid' else 'boot'
		ident = [f for f in imp.index if imp.loc[f, 'identified']]
		signs = [f for f in slo.index if slo.loc[f, 'sign_identified']]
		out['%s_identified' % tag] = ', '.join(ident)
		out['%s_sign_identified' % tag] = ', '.join(signs)
		out['%s_n_identified_chars' % tag] = sum(1 for f in ident if not f.startswith('group:'))
		out['%s_ST_REV_identified' % tag] = 'ST_REV' in ident
		out['%s_ST_REV_sign' % tag] = 'ST_REV' in signs
		out['%s_value_group_identified' % tag] = 'group:value' in ident
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--models', required=True, help='run_paper1 output with model_<rung>.pt (baseline 9 knots)')
	ap.add_argument('--out', required=True)
	ap.add_argument('--knots', default='5,9,13')
	ap.add_argument('--n_seeds', type=int, default=None)
	ap.add_argument('--n_boot', type=int, default=100)
	ap.add_argument('--skip', default='', help='comma list: thresholds, knots')
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	holdout = concat_time([valid, test])
	state_vars = cfg['state_vars']
	audit_states = cfg.get('audit_state_vars', state_vars)
	lta = tuple(cfg.get('lta_chars', audit.DEFAULT_LTA))
	n_seeds = args.n_seeds or cfg.get('ladder', {}).get('n_seeds', 1)
	skip = set(filter(None, args.skip.split(',')))
	os.makedirs(args.out, exist_ok=True)
	report = ['# Robustness to rule thresholds and knots: %s\n' % os.path.basename(args.config)]

	# ---- Part A: rule thresholds ------------------------------------------------------------------
	if 'thresholds' not in skip:
		sdfs = {}
		for rung in RUNG_SPECS:
			path = os.path.join(args.models, 'model_%s.pt' % rung)
			if os.path.exists(path):
				ms = load_rung(args.models, rung, train, state_vars)
				ev = [_ensemble_eval(ms, p.to_torch(state_vars)) for p in (train, valid, test)]
				sdfs[rung] = (ev[0]['M'], np.concatenate([ev[1]['M'], ev[2]['M']]))
		for name, spec in cfg.get('external_sdfs', {}).items():
			if os.path.exists(spec['path']):
				Ms = external_sdf(spec, [train.T, valid.T, test.T])
				sdfs[name] = (Ms[0], np.concatenate(Ms[1:]))
		for name in ('capm', 'ff3', 'ff5_umd'):
			spec = cfg.get('factor_files', {}).get(name)
			if spec and os.path.exists(spec['path']):
				Ms = factor_model_sdf(spec, [train, valid, test])
				sdfs[name] = (Ms[0], np.concatenate(Ms[1:]))
		rows = []
		for gname, grid in THRESHOLD_GRIDS.items():
			for name, (M_tr, M_ho) in sdfs.items():
				row = run_audit(name, M_tr, M_ho, train, holdout, cfg['audit'], grid, audit_states, lta)
				row['grid'] = gname
				rows.append(row)
				print('[thresholds] %-14s %-18s max|t| %.2f p=%.3f confirmed %d top %s (t=%.2f)' % (gname, name,
					row['max_abs_t'], row['global_p'], row['n_confirmed'], row['top_rule'], row['t_holdout']))
			pd.DataFrame(rows).to_csv(os.path.join(args.out, 'thresholds.csv'), index=False)
		th = pd.DataFrame(rows)
		cols = ['grid', 'model', 'max_abs_t', 'global_p', 'n_rules', 'n_confirmed', 'top_rule', 't_holdout', 'p_maxt', 'class', 'reversal_rule_confirmed']
		report.append('## Part A: rule thresholds (discovery: train; confirmation: valid + test)\n\n' + md_table(th[cols], 3) + '\n')

	# ---- Part B: knots ------------------------------------------------------------------------------
	if 'knots' not in skip:
		tcfg = TrainConfig(**cfg.get('train', {}))
		ladder_rows, audit_rows, ident_rows = [], [], []
		fixed_rules = audit.fixed_rule_class(train, audit_states)
		for k in [int(x) for x in args.knots.split(',')]:
			name = 'additive_k%d' % k
			rung = {'name': name, 'type': 'additive', 'state_varying': False, 'n_knots': k}
			prefitted = {}
			base = os.path.join(args.models, 'model_additive.pt')
			if k == 9 and os.path.exists(base):
				prefitted[name] = {'states': torch.load(base), 'pairs': None}
			ldf, fitted, _ = run_ladder(train, valid, test, state_vars, [rung], tcfg, n_seeds=n_seeds, n_pairs=0, prefitted=prefitted)
			ladder_rows.append(ldf[['model', 'n_params', 'SR_train', 'SR_valid', 'SR_test', 'EV_test', 'XS_R2_test']])
			ev = [_ensemble_eval(fitted[name], p.to_torch(state_vars)) for p in (train, valid, test)]
			row = run_audit(name, ev[0]['M'], np.concatenate([ev[1]['M'], ev[2]['M']]), train, holdout, cfg['audit'],
				THRESHOLD_GRIDS['baseline'], audit_states, lta)
			audit_rows.append(row)
			print('[knots] %s SR test %.3f; audit max|t| %.2f p=%.3f top %s (t=%.2f)' % (name, ldf['SR_test'].iloc[0],
				row['max_abs_t'], row['global_p'], row['top_rule'], row['t_holdout']))
			ident = identification(cfg, train, valid, test, k, args.n_boot, cfg.get('value_group', []))
			ident_rows.append(ident)
			print('[knots] %s Paper 2: shrink %g, eps1 identified [%s], boot identified [%s]' % (name, ident['shrink'],
				ident['eps1_identified'], ident['boot_identified']))
			pd.concat(ladder_rows).to_csv(os.path.join(args.out, 'knots_ladder.csv'), index=False)
			pd.DataFrame(audit_rows).to_csv(os.path.join(args.out, 'knots_audit.csv'), index=False)
			pd.DataFrame(ident_rows).to_csv(os.path.join(args.out, 'knots_identification.csv'), index=False)
		report.append('## Part B: knots of the additive SDF\n\n### Paper 1 ladder\n\n' + md_table(pd.concat(ladder_rows)) + '\n')
		ad = pd.DataFrame(audit_rows)
		report.append('### Paper 1 audit (baseline thresholds)\n\n' + md_table(ad[['model', 'max_abs_t', 'global_p', 'n_rules',
			'n_confirmed', 'top_rule', 't_holdout', 'p_maxt', 'class']], 3) + '\n')
		report.append('### Paper 2 identification\n\n' + md_table(pd.DataFrame(ident_rows), 3) + '\n')

	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('\n'.join(report))
	print('wrote', args.out)


if __name__ == '__main__':
	main()
