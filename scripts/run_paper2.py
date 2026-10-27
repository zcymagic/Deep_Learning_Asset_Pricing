"""Paper 2: the Rashomon set of SDFs and SDF ambiguity.

	python scripts/run_paper2.py --config config_glassbox/synthetic.json --out results/paper2_synthetic

Steps
	1. fit the glass-box SDF (rung from config) against a neural adversary; freeze its instruments
	2. closed-form refit and the Rashomon ellipsoid; bootstrap refits for marginal inference
	3. identification tables for each eps and for the bootstrap: importance (with value group) and slope
	   signs                                                                -> identification_*.csv
	4. shape functions with exact Rashomon bands                            -> shape_bands.png
	5. stock-level SDF ambiguity on every split, the monthly ambiguity index, and hold-out tests:
	   sorts, Fama-MacBeth on returns (and on panel extras), index vs macro -> ambiguity_*.csv
	6. Sharpe range across the Rashomon set (test)                         -> sharpe_range.csv
	7. REPORT.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.adversary import NeuralAdversary
from glassbox.data import concat_time, load_config, load_splits
from glassbox.interpret import plot_shapes
from glassbox.rashomon import (RashomonEllipsoid, ambiguity_index_regressions, ambiguity_regressions,
	ambiguity_sorts, select_shrinkage)
from glassbox.report import md_table
from glassbox.sdf import build_sdf
from glassbox.train import TrainConfig, evaluate, fit_gan

RUNG_SPECS = {
	'linear_state': {'type': 'linear', 'state_varying': True},
	'additive': {'type': 'additive', 'state_varying': False},
	'additive_state': {'type': 'additive', 'state_varying': True},
}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--rung', default=None, help='override rashomon.rung from the config')
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	rc = cfg.get('rashomon', {})
	train, valid, test = load_splits(cfg)
	state_vars = cfg['state_vars']
	os.makedirs(args.out, exist_ok=True)
	names = train.char_names
	groups = {'value': [c for c in cfg.get('value_group', []) if c in names]} if cfg.get('value_group') else None

	# 1. fit
	rung = args.rung or rc.get('rung', 'additive_state')
	spec = dict(RUNG_SPECS[rung])
	spec.update({'lam_group': 0.0, 'lam_smooth': rc.get('lam_smooth', 1e-3), 'n_knots': rc.get('n_knots', 9)})
	K, P, Md = train.K, len(state_vars), len(train.macro_names)
	b_tr, b_va, b_te = [p.to_torch(state_vars) for p in (train, valid, test)]
	torch.manual_seed(0)
	model = build_sdf(spec, K, P, Md)
	adv = NeuralAdversary(K, Md, 8)
	fit_gan(model, adv, b_tr, b_va, TrainConfig(**cfg.get('train', {})))
	adv.eval()
	with torch.no_grad():
		h = adv(b_tr)

	# 2. ellipsoid (closed-form refit) + bootstrap
	shrink, shrink_df = select_shrinkage(model, b_tr, h, b_va, grid=rc.get('shrink_grid', (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)))
	shrink_df.to_csv(os.path.join(args.out, 'shrinkage_selection.csv'), index=False)
	print('[rashomon] shrinkage selected on validation: %g' % shrink)
	E = RashomonEllipsoid(model, b_tr, h, eps=rc.get('eps_grid', [0.01])[0], n_boot=rc.get('n_boot', 200), shrink=shrink)
	perf = pd.DataFrame([{**{'split': s}, **{k: v for k, v in evaluate(model, b).items() if k in ('sharpe', 'sharpe_raw', 'ev', 'xs_r2')}}
		for s, b in (('train', b_tr), ('valid', b_va), ('test', b_te))])
	perf.to_csv(os.path.join(args.out, 'reference_performance.csv'), index=False)

	# 3. identification tables
	tables = {}
	for eps in rc.get('eps_grid', [0.01]):
		E.set_eps(eps)
		imp = E.importance_table(names, groups=groups)
		slo = E.slope_table(names, s=np.zeros(P) if P else None)
		tab = imp.merge(slo, on='feature', how='left')
		tab.to_csv(os.path.join(args.out, 'identification_eps%s.csv' % eps), index=False)
		tables['eps=%s' % eps] = tab
	imp = E.importance_table(names, groups=groups, method='bootstrap')
	slo = E.slope_table(names, s=np.zeros(P) if P else None, method='bootstrap')
	tab = imp.merge(slo, on='feature', how='left')
	tab.to_csv(os.path.join(args.out, 'identification_bootstrap.csv'), index=False)
	tables['bootstrap (marginal, Bonferroni)'] = tab

	# 4. shape bands at the middle eps
	grid = rc.get('eps_grid', [0.01])
	E.set_eps(grid[len(grid) // 2])
	plot_shapes(model, names, P=P, path=os.path.join(args.out, 'shape_bands.png'), bands=E)

	# 5. ambiguity
	amb = {}
	for split, panel, b in (('train', train, b_tr), ('valid', valid, b_va), ('test', test, b_te)):
		A, sign_amb, index = E.weight_ambiguity(b)
		np.savez_compressed(os.path.join(args.out, 'ambiguity_%s.npz' % split), A=A, sign_ambiguous=sign_amb, index=index)
		amb[split] = (A, sign_amb, index)
	holdout = concat_time([valid, test])
	N = holdout.N
	pad = lambda x, fill: np.concatenate([x, np.full((x.shape[0], N - x.shape[1]), fill)], axis=1) if x.shape[1] < N else x
	A_ho = np.concatenate([pad(amb['valid'][0], np.nan), pad(amb['test'][0], np.nan)])
	with torch.no_grad():
		M_ho = np.concatenate([model.sdf(b_va).numpy(), model.sdf(b_te).numpy()])
		w_ho = np.concatenate([pad(model(b_va).numpy(), np.nan), pad(model(b_te).numpy(), np.nan)])
	ctrl = {'sdf_weight': w_ho, 'abs_sdf_weight': np.abs(w_ho)}
	sorts = ambiguity_sorts(holdout, A_ho, M_ho)
	sorts.to_csv(os.path.join(args.out, 'ambiguity_sorts_holdout.csv'))
	fm = ambiguity_regressions(holdout, A_ho, extra_controls=ctrl)
	fm.to_csv(os.path.join(args.out, 'ambiguity_fama_macbeth_holdout.csv'))
	extra_fm = {}
	for key in holdout.extra:
		extra_fm[key] = ambiguity_regressions(holdout, A_ho, outcome=key, extra_controls=ctrl)
		extra_fm[key].to_csv(os.path.join(args.out, 'ambiguity_fm_%s.csv' % key))
	index_all = np.concatenate([amb['train'][2], amb['valid'][2], amb['test'][2]])
	full = concat_time([train, valid, test])
	idx_macro = ambiguity_index_regressions(full, index_all)
	idx_macro.to_csv(os.path.join(args.out, 'ambiguity_index_vs_macro.csv'), index=False)
	pd.DataFrame({'date': full.dates, 'ambiguity_index': index_all}).to_csv(os.path.join(args.out, 'ambiguity_index.csv'), index=False)
	sign_share = {s: float(amb[s][1][p.mask].mean()) for s, p in (('train', train), ('valid', valid), ('test', test))}

	# 6. Sharpe range on test
	ref, lo, hi = E.sharpe_range(b_te)
	sr = pd.DataFrame([{'eps': E.eps, 'sharpe_raw_ref': ref, 'min': lo, 'max': hi}])
	sr.to_csv(os.path.join(args.out, 'sharpe_range.csv'), index=False)

	# 7. report
	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Paper 2 results: %s\n\n' % os.path.basename(args.config))
		f.write('Glass-box SDF: `%s`, %d parameters, L2 shrinkage %g (chosen on validation Sharpe), J* = %.4g, bootstrap-implied simultaneous eps (95%%) = %.3f\n\n'
			% (rung, E.theta.size, E.shrink, E.J_star, np.quantile(E.boot_dist, 0.95) / E.J_star))
		f.write('## Shrinkage selection (closed-form SDF)\n\n' + md_table(shrink_df) + '\n\n')
		f.write('## Reference performance\n\n' + md_table(perf) + '\n\n')
		for label, tab in tables.items():
			f.write('## Identification: %s\n\n' % label + md_table(tab, 3) + '\n\n')
		f.write('## SDF ambiguity\n\nShare of stock-months whose SDF weight sign is not identified (eps=%s): %s\n\n' % (E.eps, sign_share))
		f.write('### Hold-out sorts on ambiguity\n\n' + md_table(sorts.reset_index().rename(columns={'index': 'portfolio'})) + '\n\n')
		f.write('### Hold-out Fama-MacBeth: returns on standardized ambiguity, reference SDF weight, |weight|, characteristics and their squares\n\n' + md_table(fm.reset_index().rename(columns={'index': 'variable'})) + '\n\n')
		for key, t in extra_fm.items():
			f.write('### Hold-out Fama-MacBeth: %s on ambiguity\n\n' % key + md_table(t.reset_index().rename(columns={'index': 'variable'})) + '\n\n')
		f.write('### Monthly ambiguity index vs macro series\n\n' + md_table(idx_macro) + '\n\n')
		f.write('## Sharpe ratio across the Rashomon set (test, raw SDF portfolio)\n\n' + md_table(sr) + '\n')
	print('wrote', args.out)


if __name__ == '__main__':
	main()
