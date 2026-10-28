"""Combined glass-box + CPZ GAN SDF and its audit.

	python scripts/run_combined.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural \
		--out results/combined_cpz

The combined SDF is M = 1 - b' [F_glass, F_gan], the tangency of the two normalized SDF portfolios
(b = E[FF']^-1 E[F], pricing.factor_sdf_coefficients). Where b is estimated matters because the GAN was
fit on the training period (its train Sharpe ratio is in-sample), so two variants are reported:
	combo_train  b from the training split; audited like every other SDF (confirm on valid + test)
	combo_valid  b from the validation split, where both models are out of sample; confirmation on test
	             only, so the weights never see the confirmation sample
Each variant is audited next to the two component SDFs on the same confirmation sample.
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
from glassbox.pricing import factor_sdf_coefficients, sharpe, spanning_test
from glassbox.report import md_table
from glassbox.sdf import build_sdf
from run_paper1 import audit_one


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', default='config_glassbox/cpz.json')
	ap.add_argument('--models', default='results/paper1_cpz_neural', help='output dir of run_paper1 with model_<rung>.pt')
	ap.add_argument('--rung', default='additive')
	ap.add_argument('--gan', default='sample_checkpoints/sdf_normalized_ensemble.npy')
	ap.add_argument('--out', default='results/combined_cpz')
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	panels = {'train': train, 'valid': valid, 'test': test}
	state_vars = cfg['state_vars']
	audit_states = cfg.get('audit_state_vars', state_vars)
	lta = tuple(cfg.get('lta_chars', audit.DEFAULT_LTA))
	os.makedirs(args.out, exist_ok=True)

	# component SDF portfolios (normalized weights, as in the ladder)
	spec = {'type': 'additive', 'state_varying': args.rung != 'additive'} if args.rung.startswith('additive') else {'type': args.rung}
	if args.rung == 'additive_state_pairs':
		raise SystemExit('pair rungs need their screened pairs; use additive or additive_state')
	models = []
	for st in torch.load(os.path.join(args.models, 'model_%s.pt' % args.rung)):
		m = build_sdf(spec, train.K, len(state_vars), len(train.macro_names))
		m.load_state_dict(st)
		models.append(m)
	F_gb = {s: _ensemble_eval(models, p.to_torch(state_vars))['F'] for s, p in panels.items()}
	gan = np.load(args.gan).reshape(-1)
	c = np.cumsum([0, train.T, valid.T, test.T])
	F_gan = {s: gan[c[i]:c[i + 1]] for i, s in enumerate(panels)}
	sign = np.sign(F_gan['train'].mean())		# the saved series can be the negative portfolio
	F_gan = {s: sign * x for s, x in F_gan.items()}
	F = {s: np.column_stack([F_gb[s], F_gan[s]]) for s in panels}

	# combinations
	sdfs, rows, wrows = {}, [], []
	for name, est in (('combo_train', 'train'), ('combo_valid', 'valid')):
		b = factor_sdf_coefficients(F[est])
		w = b / np.abs(b).sum()
		wrows.append({'variant': name, 'estimated_on': est, 'b_glass': b[0], 'b_gan': b[1], 'weight_glass': w[0], 'weight_gan': w[1]})
		sdfs[name] = {s: 1.0 - F[s] @ b for s in panels}
		comb = {s: F[s] @ b for s in panels}
		for s in panels:
			rows.append({'portfolio': name, 'split': s, 'sharpe': sharpe(comb[s])})
	for s in panels:
		rows.append({'portfolio': 'glass_box_' + args.rung, 'split': s, 'sharpe': sharpe(F_gb[s])})
		rows.append({'portfolio': 'cpz_gan', 'split': s, 'sharpe': sharpe(F_gan[s])})
	perf = pd.DataFrame(rows).pivot(index='portfolio', columns='split', values='sharpe')[['train', 'valid', 'test']].reset_index()
	weights = pd.DataFrame(wrows)
	# components as SDFs M = 1 - b F_i, each scaled on the training split (alphas are scale-free)
	for name, Fi in (('glass_box_' + args.rung, F_gb), ('cpz_gan', F_gan)):
		bi = factor_sdf_coefficients(Fi['train'][:, None])
		sdfs[name] = {s: 1.0 - Fi[s] * bi[0] for s in panels}

	# spanning on test: does each component have alpha on the combination and vice versa?
	span = []
	for name in ('combo_train', 'combo_valid'):
		comb_te = 1.0 - sdfs[name]['test']
		for comp, Fc in (('glass_box', F_gb['test']), ('cpz_gan', F_gan['test'])):
			a, t, r2 = spanning_test(Fc, comb_te)
			span.append({'combination': name, 'component': comp, 'alpha_component_on_combo': a, 't': t, 'R2': r2})
	span = pd.DataFrame(span)

	# audits: discovery on train; confirmation on valid + test (A) and on test only (B)
	fixed_rules = audit.fixed_rule_class(train, audit_states)
	holdouts = {'valid+test': (concat_time([valid, test]), ('valid', 'test')), 'test': (test, ('test',))}
	plan = [('valid+test', ['combo_train', 'glass_box_' + args.rung, 'cpz_gan']),
		('test', ['combo_valid', 'combo_train', 'glass_box_' + args.rung, 'cpz_gan'])]
	summaries, tables = [], {}
	for ho_name, names in plan:
		holdout, parts = holdouts[ho_name]
		for name in names:
			M_ho = np.concatenate([sdfs[name][s] for s in parts])
			table, summary = audit_one('%s [%s]' % (name, ho_name), sdfs[name]['train'], M_ho, train, holdout,
				cfg['audit'], audit_states, lta, fixed_rules)
			summary['confirmation'] = ho_name
			summaries.append(summary)
			tables['%s [%s]' % (name, ho_name)] = table
			table.drop(columns=['rule']).to_csv(os.path.join(args.out, 'audit_%s_%s.csv' % (name, ho_name.replace('+', '_'))), index=False)
	summary_df = pd.DataFrame(summaries)

	perf.to_csv(os.path.join(args.out, 'performance.csv'), index=False)
	weights.to_csv(os.path.join(args.out, 'weights.csv'), index=False)
	span.to_csv(os.path.join(args.out, 'spanning_test.csv'), index=False)
	summary_df.to_csv(os.path.join(args.out, 'audit_summary.csv'), index=False)
	keep = ['model', 'confirmation', 'holdout_max_abs_t', 'holdout_global_p', 'holdout_rms_alpha', 'n_rules_discovered']
	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Combined glass-box (%s) + CPZ GAN SDF\n\n' % args.rung)
		f.write('## Tangency weights\n\n' + md_table(weights) + '\n\n')
		f.write('## Sharpe ratios (monthly)\n\n' + md_table(perf) + '\n\n')
		f.write('## Test-period spanning: component alpha on the combination\n\n' + md_table(span) + '\n\n')
		f.write('## Audit (discovery: train)\n\n' + md_table(summary_df[[k for k in keep if k in summary_df]]) + '\n\n')
		for name, t in tables.items():
			cols = [x for x in ['description', 't_discovery', 't_holdout', 't_holdout_stepwise', 'p_maxt',
				'state_dep_var', 't_state_dep', 'class'] if x in t]
			f.write('### %s\n\n' % name + (md_table(t[cols], 3) if len(t) else '_no rule passed discovery_') + '\n\n')
	print('wrote', args.out)


if __name__ == '__main__':
	main()
