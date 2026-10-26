"""Paper 1: glass-box SDF + glass-box auditor.

	python scripts/run_paper1.py --config config_glassbox/synthetic.json --out results/paper1_synthetic

Steps
	1. complexity ladder (linear -> additive -> state -> pairs -> neural): out-of-sample Sharpe, EV, XS-R2,
	   spanning tests = the price of interpretability                      -> ladder.csv
	2. exact read-outs of the glass-box SDF: shape functions, importance    -> shapes.csv, importance.csv, *.png
	3. audit every SDF (ladder rungs + external SDFs such as the CPZ GAN ensemble):
	   discover readable failures on the training period, confirm them on valid + test, classify them
	   (limits-to-arbitrage / state-dependent / characteristic / mixed), global max-t test
	                                                                       -> audit_<model>.csv, audit_summary.csv
	4. REPORT.md with all tables.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import audit
from glassbox.data import concat_time, load_config, load_splits
from glassbox.interpret import contribution_importance, plot_pair, plot_shapes, shape_table
from glassbox.ladder import DEFAULT_RUNGS, _ensemble_eval, run_ladder
from glassbox.report import md_table
from glassbox.train import TrainConfig


def external_sdf(spec, T_parts):
	x = np.load(spec['path']).reshape(-1)
	if len(x) != sum(T_parts):
		raise ValueError('%s has %d periods, expected %d' % (spec['path'], len(x), sum(T_parts)))
	M = x if spec.get('kind', 'M') == 'M' else 1.0 - x
	c = np.cumsum([0] + list(T_parts))
	return [M[c[i]:c[i + 1]] for i in range(len(T_parts))]


def factor_model_sdf(spec, panels):
	"""M = 1 - b'F for traded factors in a CSV, b estimated on the training split only.

	spec: {"path": csv, "columns": [...], "date_col": "date"}; rows are matched to the panel dates
	(the npz 'date' field) when date_col is given, otherwise taken in order (train, valid, test).
	"""
	from glassbox.pricing import factor_sdf_coefficients
	df = pd.read_csv(spec['path'])
	cols = spec.get('columns') or [c for c in df.columns if c != spec.get('date_col')]
	if spec.get('date_col'):
		df = df.set_index(df[spec['date_col']].astype(int))
		parts = [df.loc[[int(d) for d in p.dates], cols].values for p in panels]
	else:
		F = df[cols].values
		c = np.cumsum([0] + [p.T for p in panels])
		parts = [F[c[i]:c[i + 1]] for i in range(len(panels))]
	b = factor_sdf_coefficients(parts[0])
	return [1.0 - F @ b for F in parts]


def audit_one(name, M_tr, M_ho, train, holdout, cfg_a, state_vars, lta, fixed_rules):
	t0 = time.time()
	disc = audit.discover(train, M_tr, state_vars, beam=cfg_a['beam'], top=cfg_a['top'], min_stocks=cfg_a['min_stocks'])
	if len(disc):
		conf = audit.confirm(holdout, M_ho, disc, n_boot=cfg_a['n_boot'], min_stocks=cfg_a['min_stocks'], state_vars=state_vars)
		table, fmap = audit.failure_map(conf, lta_chars=lta)
	else:
		table, fmap = disc, None
	glob = audit.global_test(holdout, M_ho, fixed_rules, n_boot=cfg_a['n_boot'], min_stocks=cfg_a['min_stocks'])
	summary = {'model': name, 'holdout_max_abs_t': glob['max_abs_t'], 'holdout_global_p': glob['p_value'],
		'holdout_rms_alpha': glob['rms_alpha'], 'n_rules_discovered': len(disc)}
	if fmap is not None:
		for cls, r in fmap.iterrows():
			summary['n_sig_' + cls] = r['n_significant']
			summary['share_' + cls] = r['share_of_t2']
	print('[audit] %-24s global max|t| %.2f (p=%.3f), %d rules, %.0fs' % (name, glob['max_abs_t'], glob['p_value'], len(disc), time.time() - t0))
	return table, summary


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--list_macro', action='store_true')
	ap.add_argument('--skip_neural', action='store_true')
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	if args.list_macro:
		print('\n'.join(train.macro_names))
		return
	for v in cfg['state_vars'] + cfg.get('audit_state_vars', []):
		if v not in train.macro_names:
			raise SystemExit('state variable %r not in the macro file; run with --list_macro' % v)
	os.makedirs(args.out, exist_ok=True)
	state_vars = cfg['state_vars']
	audit_states = cfg.get('audit_state_vars', state_vars)
	lta = tuple(cfg.get('lta_chars', audit.DEFAULT_LTA))
	tcfg = TrainConfig(**cfg.get('train', {}))

	# 1. complexity ladder
	rungs = [dict(r) for r in DEFAULT_RUNGS if not (args.skip_neural and r['type'] == 'neural')]
	for r in rungs:
		if r['type'] == 'neural':
			r['hidden'] = cfg.get('ladder', {}).get('neural_hidden', [64, 64])
	ladder_df, fitted, _ = run_ladder(train, valid, test, state_vars, rungs, tcfg,
		n_seeds=cfg.get('ladder', {}).get('n_seeds', 1), n_pairs=cfg.get('ladder', {}).get('n_pairs', 5))
	ladder_df.to_csv(os.path.join(args.out, 'ladder.csv'), index=False)

	# 2. exact read-outs of the richest glass-box rung
	gb_name = 'additive_state_pairs' if 'additive_state_pairs' in fitted else 'additive_state'
	gb = fitted[gb_name][0]
	b_tr = train.to_torch(state_vars)
	shape_table(gb, train.char_names, state_vars).to_csv(os.path.join(args.out, 'shapes.csv'), index=False)
	contribution_importance(gb, b_tr, train.char_names).to_csv(os.path.join(args.out, 'importance.csv'), index=False)
	for p, sname in enumerate(state_vars):
		plot_shapes(gb, train.char_names, state_name=sname, state_index=p, P=len(state_vars),
			path=os.path.join(args.out, 'shapes_%s.png' % sname))
	for q in range(len(getattr(gb, 'pairs', []))):
		plot_pair(gb, q, train.char_names, path=os.path.join(args.out, 'pair_%d.png' % q))
	for name, models in fitted.items():
		torch.save([m.state_dict() for m in models], os.path.join(args.out, 'model_%s.pt' % name))

	# 3. audit: discovery on train, confirmation on valid + test
	holdout = concat_time([valid, test])
	fixed_rules = audit.fixed_rule_class(train, audit_states)
	sdfs = {}
	b_va, b_te = valid.to_torch(state_vars), test.to_torch(state_vars)
	for name, models in fitted.items():
		ev = [_ensemble_eval(models, b) for b in (b_tr, b_va, b_te)]
		sdfs[name] = (ev[0]['M'], np.concatenate([ev[1]['M'], ev[2]['M']]))
	for name, spec in cfg.get('external_sdfs', {}).items():
		if os.path.exists(spec['path']):
			Ms = external_sdf(spec, [train.T, valid.T, test.T])
			sdfs[name] = (Ms[0], np.concatenate(Ms[1:]))
		else:
			print('[audit] skipping %s: %s not found' % (name, spec['path']))
	for name, spec in cfg.get('factor_files', {}).items():
		if os.path.exists(spec['path']):
			Ms = factor_model_sdf(spec, [train, valid, test])
			sdfs[name] = (Ms[0], np.concatenate(Ms[1:]))
		else:
			print('[audit] skipping %s: %s not found' % (name, spec['path']))
	summaries = []
	for name, (M_tr, M_ho) in sdfs.items():
		table, summary = audit_one(name, M_tr, M_ho, train, holdout, cfg['audit'], audit_states, lta, fixed_rules)
		table.drop(columns=['rule']).to_csv(os.path.join(args.out, 'audit_%s.csv' % name), index=False)
		summaries.append(summary)
	summary_df = pd.DataFrame(summaries)
	summary_df.to_csv(os.path.join(args.out, 'audit_summary.csv'), index=False)

	# 4. report
	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Paper 1 results: %s\n\n' % os.path.basename(args.config))
		f.write('## Complexity ladder (price of interpretability)\n\n' + md_table(ladder_df) + '\n\n')
		f.write('## Glass-box SDF (%s): weight-variance shares\n\n' % gb_name)
		f.write(md_table(pd.read_csv(os.path.join(args.out, 'importance.csv'))) + '\n\n')
		f.write('## Audit summary (discovery: train; confirmation: valid + test)\n\n' + md_table(summary_df) + '\n\n')
		for name in sdfs:
			t = pd.read_csv(os.path.join(args.out, 'audit_%s.csv' % name))
			cols = [c for c in ['description', 't_discovery', 't_stepwise', 't_holdout', 't_holdout_stepwise', 'p_maxt',
				'state_dep_var', 't_state_dep', 'class'] if c in t]
			f.write('### Failures of `%s`\n\n' % name + (md_table(t[cols], 3) if len(t) else '_none found_') + '\n\n')
	with open(os.path.join(args.out, 'config_used.json'), 'w') as f:
		json.dump(cfg, f, indent=2)
	print('wrote', args.out)


if __name__ == '__main__':
	main()
