"""Robustness of the Paper 1 results to the choice of macro state variables.

	python scripts/run_state_robustness.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural \
		--out results/state_robustness_cpz

State variables enter in two places, and each alternative set is used for both:
	1. the SDF: the state-varying rungs (linear_state, additive_state) are refit with the set
	2. the audit: rules may condition on the set, and the state-dependence test uses it
The state-free SDFs (additive glass-box reloaded from --models, CPZ GAN, CAPM, FF5 + UMD) are re-audited
under each set, which isolates the effect of the audit's state variables on the classification.
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
from glassbox.ladder import _ensemble_eval, run_ladder
from glassbox.report import md_table
from glassbox.sdf import build_sdf
from glassbox.train import TrainConfig
from run_paper1 import audit_one, external_sdf, factor_model_sdf

STATE_SETS = {
	'baseline': ['BAAFFM', 'T10YFFM', 'VXOCLSx'],
	'welch_goyal': ['dfy', 'tms', 'svar'],
	'real_activity': ['UNRATE', 'INDPRO', 'CPIAUCSL'],
	'valuation': ['dp', 'tbl', 'ntis'],
	'vol_only': ['VXOCLSx'],
}
STATE_RUNGS = [
	{'name': 'linear_state', 'type': 'linear', 'state_varying': True},
	{'name': 'additive_state', 'type': 'additive', 'state_varying': True},
]


def top_rule(table):
	"""The confirmed rule with the largest |hold-out t| (or the first discovered one if none confirmed)."""
	if not len(table):
		return {'top_rule': '', 't_holdout': np.nan, 'p_maxt': np.nan, 'class': '', 'state_dep_var': '', 't_state_dep': np.nan}
	sig = table[table['p_maxt'] < 0.05]
	row = (sig if len(sig) else table).iloc[np.argmax(np.abs((sig if len(sig) else table)['t_holdout'].values))]
	return {'top_rule': row['description'], 't_holdout': row['t_holdout'], 'p_maxt': row['p_maxt'],
		'class': row.get('class', ''), 'state_dep_var': row.get('state_dep_var', ''), 't_state_dep': row.get('t_state_dep', np.nan)}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', default='config_glassbox/cpz.json')
	ap.add_argument('--models', default='results/paper1_cpz_neural', help='run_paper1 output with model_additive.pt')
	ap.add_argument('--out', default='results/state_robustness_cpz')
	ap.add_argument('--sets', default=','.join(STATE_SETS), help='comma-separated subset of ' + ','.join(STATE_SETS))
	ap.add_argument('--n_seeds', type=int, default=None, help='override ladder.n_seeds')
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	holdout = concat_time([valid, test])
	lta = tuple(cfg.get('lta_chars', audit.DEFAULT_LTA))
	tcfg = TrainConfig(**cfg.get('train', {}))
	n_seeds = args.n_seeds or cfg.get('ladder', {}).get('n_seeds', 1)
	os.makedirs(args.out, exist_ok=True)
	sets = {k: STATE_SETS[k] for k in args.sets.split(',')}
	for vars_ in sets.values():
		for v in vars_:
			if v not in train.macro_names:
				raise SystemExit('%s not in the macro file' % v)

	# state-free SDFs: their M does not depend on the state set
	fixed = {}
	models = []
	for st in torch.load(os.path.join(args.models, 'model_additive.pt')):
		m = build_sdf({'type': 'additive', 'state_varying': False}, train.K, 0, len(train.macro_names))
		m.load_state_dict(st)
		models.append(m)
	ev = [_ensemble_eval(models, p.to_torch([])) for p in (train, valid, test)]
	fixed['additive'] = (ev[0]['M'], np.concatenate([ev[1]['M'], ev[2]['M']]))
	for name, spec in cfg.get('external_sdfs', {}).items():
		if os.path.exists(spec['path']):
			Ms = external_sdf(spec, [train.T, valid.T, test.T])
			fixed[name] = (Ms[0], np.concatenate(Ms[1:]))
	for name in ('capm', 'ff5_umd'):
		spec = cfg.get('factor_files', {}).get(name)
		if spec and os.path.exists(spec['path']):
			Ms = factor_model_sdf(spec, [train, valid, test])
			fixed[name] = (Ms[0], np.concatenate(Ms[1:]))

	ladder_rows, audit_rows = [], []
	for set_name, state_vars in sets.items():
		print('=== state set %s: %s' % (set_name, state_vars))
		ladder_df, fitted, _ = run_ladder(train, valid, test, state_vars, STATE_RUNGS, tcfg, n_seeds=n_seeds, n_pairs=0)
		ladder_df.insert(0, 'state_set', set_name)
		ladder_rows.append(ladder_df[['state_set', 'model', 'n_params', 'SR_train', 'SR_valid', 'SR_test', 'EV_test', 'XS_R2_test']])
		sdfs = dict(fixed)
		b = [p.to_torch(state_vars) for p in (train, valid, test)]
		for name, ms in fitted.items():
			e = [_ensemble_eval(ms, x) for x in b]
			sdfs[name] = (e[0]['M'], np.concatenate([e[1]['M'], e[2]['M']]))
		fixed_rules = audit.fixed_rule_class(train, state_vars)
		for name, (M_tr, M_ho) in sdfs.items():
			table, summary = audit_one('%s [%s]' % (name, set_name), M_tr, M_ho, train, holdout, cfg['audit'],
				state_vars, lta, fixed_rules)
			table.drop(columns=['rule']).to_csv(os.path.join(args.out, 'audit_%s_%s.csv' % (set_name, name)), index=False)
			audit_rows.append({'state_set': set_name, 'model': name, 'holdout_max_abs_t': summary['holdout_max_abs_t'],
				'holdout_global_p': summary['holdout_global_p'], 'n_rules': summary['n_rules_discovered'], **top_rule(table)})
		pd.concat(ladder_rows).to_csv(os.path.join(args.out, 'ladder.csv'), index=False)
		pd.DataFrame(audit_rows).to_csv(os.path.join(args.out, 'audit.csv'), index=False)

	ladder = pd.concat(ladder_rows)
	audits = pd.DataFrame(audit_rows)
	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Robustness to the macro state variables\n\n')
		f.write('State sets: ' + '; '.join('`%s` = %s' % (k, ', '.join(v)) for k, v in sets.items()) + '\n\n')
		f.write('## State-varying SDFs refit with each set\n\n' + md_table(ladder) + '\n\n')
		f.write('## Audit under each set (discovery: train; confirmation: valid + test)\n\n' + md_table(audits, 3) + '\n')
	print('wrote', args.out)


if __name__ == '__main__':
	main()
