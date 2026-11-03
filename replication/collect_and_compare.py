"""Collect the replication outputs into the layout of docs/ and compare them with the committed tables.

	python replication/collect_and_compare.py [--results results] [--out replication_output]

Every committed table under docs/ that a script produces is listed in MAPPING with the result file it
comes from. Each available result is copied to <out>/<docs path>; CSV files are then compared with the
committed copy, numerically for numeric columns and exactly otherwise:
	match     same shape and labels, numeric values equal to 1e-6 (relative)
	close     same shape and labels, numeric values within 5% (relative) or 1e-4 (absolute)
	differs   anything else (details printed)
	missing   the result file was not produced (stage not run)
Results that depend on stochastic network training can differ in the last digits across machines,
BLAS libraries and PyTorch versions; data builds, factor-model audits and closed-form steps should match.
A summary is written to <out>/comparison.csv.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

R = '{results}'
D = '{datasets}'


def _cpz():
	m = {}
	p1 = R + '/paper1_cpz_neural/'
	for k in ['additive', 'additive_state_pairs', 'capm', 'cpz_gan_ensemble', 'ff3', 'ff5_umd']:
		m['cpz_results/paper1_audit_%s.csv' % k] = p1 + 'audit_%s.csv' % k
	m.update({
		'cpz_results/paper1_audit_summary.csv': p1 + 'audit_summary.csv',
		'cpz_results/paper1_importance_additive_state_pairs.csv': p1 + 'importance.csv',
		'cpz_results/paper1_ladder.csv': p1 + 'ladder.csv',
		'cpz_results/paper1_shapes_BAAFFM.png': p1 + 'shapes_BAAFFM.png',
		'cpz_results/paper1_spanning_vs_cpz_gan.csv': R + '/spanning_cpz_gan/spanning_test.csv',
		'cpz_results/combined_audit_summary.csv': R + '/combined_cpz/audit_summary.csv',
		'cpz_results/combined_performance.csv': R + '/combined_cpz/performance.csv',
		'cpz_results/combined_spanning_test.csv': R + '/combined_cpz/spanning_test.csv',
		'cpz_results/combined_weights.csv': R + '/combined_cpz/weights.csv',
		'cpz_results/state_robustness_audit.csv': R + '/state_robustness_cpz/audit.csv',
		'cpz_results/state_robustness_ladder.csv': R + '/state_robustness_cpz/ladder.csv',
	})
	for rung in ['additive', 'additive_state']:
		p2 = R + '/paper2_cpz_%s/' % rung
		for f in ['identification_bootstrap', 'identification_eps0.002', 'identification_eps0.01', 'identification_eps0.05',
				'shrinkage_selection']:
			m['cpz_results/paper2_%s_%s.csv' % (rung, f)] = p2 + f + '.csv'
		m['cpz_results/paper2_%s_shape_bands.png' % rung] = p2 + 'shape_bands.png'
	return m


def _osap():
	m = {}
	for tag, d in (('bridge', 'paper1_osap'), ('post', 'paper1_osap_post_only')):
		for k in ['additive', 'capm', 'ff3', 'ff5_umd', 'linear_state']:
			m['osap_results/paper1_audit_%s_%s.csv' % (tag, k)] = R + '/%s/audit_%s.csv' % (d, k)
		m['osap_results/paper1_audit_summary_%s.csv' % tag] = R + '/%s/audit_summary.csv' % d
		m['osap_results/paper1_ladder_%s.csv' % tag] = R + '/%s/ladder.csv' % d
	m['osap_results/paper1_importance_additive_state_pairs.csv'] = R + '/paper1_osap/importance.csv'
	for f in ['identification_bootstrap', 'identification_eps0.002', 'identification_eps0.01', 'identification_eps0.05',
			'reference_performance', 'shrinkage_selection']:
		m['osap_results/paper2_additive_%s.csv' % f] = R + '/paper2_osap_additive/%s.csv' % f
	for f in ['rule_alpha.csv', 'rule_alpha_periods.csv', 'rule_alpha.png']:
		m['osap_results/' + f] = R + '/rule_alpha/' + f
	m['osap_results/char_consistency.csv'] = R + '/char_consistency/char_consistency.csv'
	m['osap_results/char_crossvalidation_jkp.csv'] = R + '/char_crossvalidation/char_crossvalidation_jkp.csv'
	m['osap_results/char_crossvalidation_jkp_full.csv'] = R + '/char_crossvalidation/char_crossvalidation_jkp_full.csv'
	m['osap_results/build_meta.json'] = D + '/osap/build_meta.json'
	return m


def _robustness():
	m = {}
	for p, d in (('cpz', 'cpz'), ('osap', 'osap')):
		for f in ['knots_audit', 'knots_identification', 'knots_ladder']:
			m['robustness/%s_%s.csv' % (p, f)] = R + '/knot_threshold_%s/%s.csv' % (d, f)
		m['robustness/%s_thresholds.csv' % p] = R + '/knot_threshold_%s/thresholds.csv' % d
		for f in ['windows', 'friction_rules', 'friction_shares']:
			m['robustness/%s_%s.csv' % (p, f)] = R + '/window_friction_%s/%s.csv' % (d, f)
	return m


def _jkp():
	m = {}
	runs = {'jkp153': 'paper1_jkp', 'jkp153_post': 'paper1_jkp_post', 'themes': 'paper1_jkp_themes',
		'themes_post': 'paper1_jkp_themes_post'}
	for tag, d in runs.items():
		m['jkp_results/ladder_%s.csv' % tag] = R + '/%s/ladder.csv' % d
		m['jkp_results/audit_summary_%s.csv' % tag] = R + '/%s/audit_summary.csv' % d
		for k in ['capm', 'ff3', 'ff5_umd', 'linear', 'additive']:
			m['jkp_results/audit_%s_%s.csv' % (k, tag)] = R + '/%s/audit_%s.csv' % (d, k)
	m['jkp_results/ladder_jkp153_unclipped_states.csv'] = R + '/paper1_jkp_noclip/ladder.csv'
	m['jkp_results/audit_summary_themes_on_jkp153_rules.csv'] = R + '/cross_audit_themes_on_jkp153/audit_summary.csv'
	for tag, d in (('jkp153', 'paper2_jkp_additive'), ('themes', 'paper2_jkp_themes_additive')):
		for f in ['identification_eps0.01', 'identification_eps0.05', 'identification_bootstrap',
				'ambiguity_fama_macbeth_holdout', 'ambiguity_sorts_holdout', 'shrinkage_selection', 'sharpe_range']:
			m['jkp_results/paper2_%s_%s.csv' % (tag, f)] = R + '/%s/%s.csv' % (d, f)
	m['jkp_results/themes_shape_bands.png'] = R + '/paper2_jkp_themes_additive/shape_bands.png'
	m['jkp_results/build_meta_jkp153.json'] = D + '/jkp_panel/build_meta.json'
	return m


def _synthetic():
	return {
		'synthetic_validation/paper1_REPORT.md': R + '/paper1_synthetic/REPORT.md',
		'synthetic_validation/paper2_REPORT.md': R + '/paper2_synthetic/REPORT.md',
		'synthetic_validation/paper2_shape_bands.png': R + '/paper2_synthetic/shape_bands.png',
	}


MAPPING = {**_synthetic(), **_cpz(), **_osap(), **_robustness(), **_jkp()}
# columns of the committed table named differently by the script that now produces it
RENAME = {'cpz_results/paper1_spanning_vs_cpz_gan.csv': {'SR2_gap_vs_external': 'SR2_gap_vs_cpz',
	'alpha_external_on_rung': 'alpha_cpz_on_rung', 'alpha_rung_on_external': 'alpha_rung_on_cpz'}}


def compare_csv(new_path, old_path):
	a, b = pd.read_csv(new_path), pd.read_csv(old_path)
	if a.shape != b.shape or list(a.columns) != list(b.columns):
		return 'differs', 'shape/columns %s %s vs committed %s %s' % (a.shape, list(a.columns)[:6], b.shape, list(b.columns)[:6])
	worst_rel, worst_abs, worst_rel_c = 0.0, 0.0, ''
	for c in a.columns:
		if a[c].dtype == bool or b[c].dtype == bool:			# flags: compare as text
			x = y = pd.Series(np.nan, index=a.index)
		else:
			x, y = pd.to_numeric(a[c], errors='coerce'), pd.to_numeric(b[c], errors='coerce')
		num = x.notna() & y.notna()
		if (x.notna() != y.notna()).any() or not num.any():
			sa = a[c].astype(object).where(a[c].notna(), '').astype(str).to_numpy(dtype=object)
			sb = b[c].astype(object).where(b[c].notna(), '').astype(str).to_numpy(dtype=object)
			if (sa != sb).any():
				i = int(np.argmax(sa != sb))
				return 'differs', 'column %s row %d: %r vs committed %r' % (c, i, sa[i], sb[i])
			continue
		d = (x[num] - y[num]).abs()
		if d.max() > worst_abs:
			worst_abs, worst_rel_c = d.max(), c
		worst_rel = max(worst_rel, float(np.where(d > 1e-12, d / np.maximum(y[num].abs(), 1e-12), 0).max()))
	if worst_rel <= 1e-6 or worst_abs <= 1e-9:
		return 'match', ''
	if worst_rel <= 0.05 or worst_abs <= 1e-4:
		return 'close', 'max rel diff %.2g (max abs %.2g in %s)' % (worst_rel, worst_abs, worst_rel_c)
	return 'differs', 'max rel diff %.2g (max abs %.2g in %s)' % (worst_rel, worst_abs, worst_rel_c)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--results', default='results')
	ap.add_argument('--datasets', default='datasets')
	ap.add_argument('--docs', default='docs')
	ap.add_argument('--out', default='replication_output')
	args = ap.parse_args()
	rows = []
	for doc, src in MAPPING.items():
		src = src.format(results=args.results, datasets=args.datasets)
		committed = os.path.join(args.docs, doc)
		if not os.path.exists(src):
			rows.append({'table': doc, 'status': 'missing', 'detail': src})
			continue
		dst = os.path.join(args.out, doc)
		os.makedirs(os.path.dirname(dst), exist_ok=True)
		if doc in RENAME:
			pd.read_csv(src).rename(columns=RENAME[doc]).to_csv(dst, index=False)
		else:
			shutil.copy(src, dst)
		if doc.endswith('.csv') and os.path.exists(committed):
			status, detail = compare_csv(dst, committed)
		else:
			status, detail = 'copied', ''
		rows.append({'table': doc, 'status': status, 'detail': detail})
	df = pd.DataFrame(rows)
	os.makedirs(args.out, exist_ok=True)
	df.to_csv(os.path.join(args.out, 'comparison.csv'), index=False)
	pd.set_option('display.width', 200)
	pd.set_option('display.max_colwidth', 110)
	print(df[df['status'] != 'missing'].to_string(index=False))
	print('\n' + ', '.join('%s: %d' % kv for kv in df['status'].value_counts().items()))
	print('written to', os.path.join(args.out, 'comparison.csv'))


if __name__ == '__main__':
	main()
