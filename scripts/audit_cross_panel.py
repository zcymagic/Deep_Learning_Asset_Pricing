"""Audit SDFs fitted on one panel against the rule space of another panel with the same stock-months.

	python scripts/audit_cross_panel.py --models results/paper1_jkp_themes \
		--model_config config_glassbox/jkp_themes.json --audit_config config_glassbox/jkp.json \
		--out results/cross_audit_themes_on_jkp153

Used to test the 13-theme glass-box SDFs against rules built from JKP's 153 underlying characteristics:
the theme panel is derived from the 153-characteristic panel, so returns, stocks and months coincide,
and an SDF is a monthly series M_t that can be audited on either. Discovery on the training split,
confirmation and the global test on validation + test, as in run_paper1.py.
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
from run_knot_threshold_robustness import load_rung
from run_paper1 import audit_one


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--models', required=True)
	ap.add_argument('--model_config', required=True)
	ap.add_argument('--audit_config', required=True)
	ap.add_argument('--rungs', nargs='+', default=['linear', 'additive', 'additive_state'])
	ap.add_argument('--out', required=True)
	args = ap.parse_args()
	torch.set_num_threads(max(1, os.cpu_count() or 1))
	os.makedirs(args.out, exist_ok=True)

	# 1. SDF series from the models' own panel
	mcfg = load_config(args.model_config)
	m_splits = load_splits(mcfg)
	sv = mcfg['state_vars']
	sdfs, dates = {}, np.concatenate([p.dates for p in m_splits])
	for rung in args.rungs:
		if os.path.exists(os.path.join(args.models, 'model_%s.pt' % rung)):
			ms = load_rung(args.models, rung, m_splits[0], sv)
			sdfs[rung] = [_ensemble_eval(ms, p.to_torch(sv))['M'] for p in m_splits]
	del m_splits

	# 2. audit on the other panel
	acfg = load_config(args.audit_config)
	train, valid, test = load_splits(acfg)
	a_dates = np.concatenate([p.dates for p in (train, valid, test)])
	if not np.array_equal(dates, a_dates):
		raise ValueError('the two panels do not cover the same months')
	holdout = concat_time([valid, test])
	audit_states = acfg.get('audit_state_vars', acfg['state_vars'])
	lta = tuple(acfg.get('lta_chars', audit.DEFAULT_LTA))
	fixed_rules = audit.fixed_rule_class(train, audit_states)
	rows = []
	with open(os.path.join(args.out, 'REPORT.md'), 'w') as f:
		f.write('# Cross-panel audit: SDFs from %s audited on %s\n\n' % (args.models, os.path.basename(args.audit_config)))
		tables = {}
		for rung, (M_tr, M_va, M_te) in sdfs.items():
			table, summary = audit_one(rung, M_tr, np.concatenate([M_va, M_te]), train, holdout, acfg['audit'],
				audit_states, lta, fixed_rules)
			table.to_csv(os.path.join(args.out, 'audit_%s.csv' % rung), index=False)
			rows.append(summary)
			tables[rung] = table
		summ = pd.DataFrame(rows)
		summ.to_csv(os.path.join(args.out, 'audit_summary.csv'), index=False)
		f.write('## Summary (discovery: train; confirmation: valid + test)\n\n' + md_table(summ) + '\n\n')
		for rung, table in tables.items():
			cols = [c for c in ['description', 't_discovery', 't_holdout', 'p_maxt', 'class'] if c in table]
			f.write('## %s\n\n' % rung + (md_table(table[cols], 3) if len(table) else 'no rules discovered') + '\n\n')
	print('wrote', args.out)


if __name__ == '__main__':
	main()
