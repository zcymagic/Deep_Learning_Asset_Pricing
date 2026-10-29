"""Robustness of the Paper 1 audit to the discovery window and to the friction (limits-to-arbitrage) list.

	python scripts/run_window_friction_robustness.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --out results/window_friction_cpz

The SDFs themselves stay fixed (glass-box rungs reloaded from --models, factor-model coefficients
estimated on the training split); only the audit changes.
Part A, discovery window: the split of 1967-2016 into a discovery and a confirmation sample moves.
Part B, friction list: confirmed failures from the baseline window are re-classified with alternative
lists of limits-to-arbitrage characteristics.
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
from glassbox.data import concat_time, load_config, load_splits
from glassbox.ladder import _ensemble_eval
from glassbox.report import md_table
from run_knot_threshold_robustness import RUNG_SPECS, load_rung
from run_paper1 import external_sdf, factor_model_sdf

WINDOWS = {
	'baseline': ((196701, 198612), (198701, 201612)),
	'late': ((197701, 198612), (198701, 201612)),
	'extended': ((196701, 199112), (199201, 201612)),
	'within_holdout': ((198701, 200112), (200201, 201612)),
}
BASE_LTA = ['LME', 'Spread', 'IdioVol', 'Resid_Var', 'LTurnover', 'SUV', 'Variance', 'Rel2High', 'AT']
FRICTION_LISTS = {
	'baseline': BASE_LTA,
	'without_Rel2High': [c for c in BASE_LTA if c != 'Rel2High'],
	'narrow_liquidity': ['LME', 'Spread', 'LTurnover'],
	'reversal_as_liquidity': BASE_LTA + ['ST_REV', 'r2_1'],
}


def window(panel, M, lo, hi):
	ym = panel.dates.astype(int) // 100
	idx = np.where((ym >= lo) & (ym <= hi))[0]
	return panel.subset_time(idx), M[idx]


def summarize(name, table, glob):
	row = {'model': name, 'max_abs_t': glob['max_abs_t'], 'global_p': glob['p_value'], 'n_rules': len(table),
		'n_confirmed': 0, 'top_rule': '', 't_holdout': np.nan, 'p_maxt': np.nan, 'reversal_rule_confirmed': False}
	if len(table):
		sig = table[table['p_maxt'] < 0.05]
		row['n_confirmed'] = len(sig)
		pick = sig if len(sig) else table
		top = pick.iloc[int(np.argmax(np.abs(pick['t_holdout'].values)))]
		row.update({'top_rule': top['description'], 't_holdout': top['t_holdout'], 'p_maxt': top['p_maxt'],
			'reversal_rule_confirmed': bool(any('ST_REV' in d for d in sig['description']))})
	return row


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--models', required=True)
	ap.add_argument('--out', required=True)
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	full = concat_time([train, valid, test])
	state_vars = cfg['state_vars']
	audit_states = cfg.get('audit_state_vars', state_vars)
	ca = cfg['audit']
	os.makedirs(args.out, exist_ok=True)

	# SDF series over 1967-2016 (fixed: estimated on the training split)
	sdfs = {}
	for rung in RUNG_SPECS:
		if os.path.exists(os.path.join(args.models, 'model_%s.pt' % rung)):
			ms = load_rung(args.models, rung, train, state_vars)
			sdfs[rung] = np.concatenate([_ensemble_eval(ms, p.to_torch(state_vars))['M'] for p in (train, valid, test)])
	for name, spec in cfg.get('external_sdfs', {}).items():
		if os.path.exists(spec['path']):
			sdfs[name] = np.concatenate(external_sdf(spec, [train.T, valid.T, test.T]))
	for name in ('capm', 'ff3', 'ff5_umd'):
		spec = cfg.get('factor_files', {}).get(name)
		if spec and os.path.exists(spec['path']):
			sdfs[name] = np.concatenate(factor_model_sdf(spec, [train, valid, test]))

	# ---- Part A: discovery windows -----------------------------------------------------------------
	rows, baseline_tables = [], {}
	for wname, ((dlo, dhi), (clo, chi)) in WINDOWS.items():
		for name, M in sdfs.items():
			disc_p, disc_M = window(full, M, dlo, dhi)
			conf_p, conf_M = window(full, M, clo, chi)
			disc = audit.discover(disc_p, disc_M, audit_states, beam=ca['beam'], top=ca['top'], min_stocks=ca['min_stocks'])
			fixed = audit.fixed_rule_class(disc_p, audit_states)
			glob = audit.global_test(conf_p, conf_M, fixed, n_boot=ca['n_boot'], min_stocks=ca['min_stocks'])
			table = audit.confirm(conf_p, conf_M, disc, n_boot=ca['n_boot'], min_stocks=ca['min_stocks'],
				state_vars=audit_states) if len(disc) else disc
			if wname == 'baseline':
				baseline_tables[name] = table
			row = summarize(name, table, glob)
			row.update({'window': wname, 'discovery': '%d-%d' % (dlo // 100, dhi // 100), 'confirmation': '%d-%d' % (clo // 100, chi // 100)})
			rows.append(row)
			print('[window] %-15s %-18s max|t| %.2f p=%.3f confirmed %d top %s (t=%.2f)' % (wname, name, row['max_abs_t'],
				row['global_p'], row['n_confirmed'], row['top_rule'], row['t_holdout']))
		pd.DataFrame(rows).to_csv(os.path.join(args.out, 'windows.csv'), index=False)
	win = pd.DataFrame(rows)

	# ---- Part B: friction lists (re-classify the baseline window's confirmed failures) ---------------
	frows, shares = [], []
	for lname, lta in FRICTION_LISTS.items():
		for name, table in baseline_tables.items():
			if not len(table):
				continue
			df, fmap = audit.failure_map(table, lta_chars=tuple(lta))
			sig = df[df['p_maxt'] < 0.05]
			for _, r in sig.iterrows():
				frows.append({'friction_list': lname, 'model': name, 'rule': r['description'], 't_holdout': r['t_holdout'],
					'state_dep_var': r.get('state_dep_var', ''), 't_state_dep': r.get('t_state_dep', np.nan), 'class': r['class']})
			srow = {'friction_list': lname, 'model': name}
			srow.update({'share_' + k: v for k, v in fmap['share_of_t2'].items()})
			shares.append(srow)
	fr = pd.DataFrame(frows)
	sh = pd.DataFrame(shares)
	fr.to_csv(os.path.join(args.out, 'friction_rules.csv'), index=False)
	sh.to_csv(os.path.join(args.out, 'friction_shares.csv'), index=False)

	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Robustness to the discovery window and the friction list: %s\n\n' % os.path.basename(args.config))
		cols = ['window', 'discovery', 'confirmation', 'model', 'max_abs_t', 'global_p', 'n_rules', 'n_confirmed', 'top_rule',
			't_holdout', 'p_maxt', 'reversal_rule_confirmed']
		f.write('## Part A: discovery windows\n\n' + md_table(win[cols], 3) + '\n\n')
		f.write('## Part B: friction lists (baseline window)\n\n### Confirmed failures and their class\n\n' + md_table(fr, 3) + '\n\n')
		f.write('### Share of squared hold-out t by class\n\n' + md_table(sh, 3) + '\n')
	print('wrote', args.out)


if __name__ == '__main__':
	main()
