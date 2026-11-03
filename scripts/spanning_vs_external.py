"""Spanning tests of an external SDF (the published CPZ GAN ensemble) against each glass-box rung.

	python scripts/spanning_vs_external.py --config config_glassbox/cpz.json \
		--models results/paper1_cpz_neural --external cpz_gan_ensemble --out results/spanning_cpz_gan

On the test split, F = 1 - M is each SDF's portfolio return. For every rung saved in --models:
	SR2_gap         SR(external)^2 - SR(rung)^2
	alpha_ext_on_rung, t, R2   regression of F_external on F_rung (alpha, NW t, R2)
	alpha_rung_on_ext, t_rev   regression of F_rung on F_external
	corr            correlation of the two portfolio returns
Writes spanning_test.csv (docs/cpz_results/paper1_spanning_vs_cpz_gan.csv).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glassbox.data import load_config, load_splits
from glassbox.ladder import _ensemble_eval
from glassbox.pricing import spanning_test
from run_paper1 import external_sdf

RUNGS = ['linear', 'linear_state', 'additive', 'additive_state', 'additive_state_pairs', 'neural']


def load_models(models_dir, rung, train, state_vars, cfg):
	"""Rebuild a saved rung (all seeds), including screened pairs and the neural rung."""
	from glassbox.ladder import DEFAULT_RUNGS
	from glassbox.sdf import build_sdf
	spec = dict(next(r for r in DEFAULT_RUNGS if r['name'] == rung))
	if spec.get('pairs') == 'screen':
		lad = pd.read_csv(os.path.join(models_dir, 'ladder.csv'))
		txt = lad.loc[lad['model'] == rung, 'pairs'].iloc[0]
		spec['pairs'] = [tuple(train.char_idx(c.strip()) for c in p.split(' x ')) for p in txt.split(', ')]
	if spec['type'] == 'neural':
		spec['hidden'] = cfg.get('ladder', {}).get('neural_hidden', [64, 64])
	models = []
	for state in torch.load(os.path.join(models_dir, 'model_%s.pt' % rung)):
		m = build_sdf(spec, train.K, len(state_vars), len(train.macro_names))
		m.load_state_dict(state)
		models.append(m)
	return models


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--models', required=True)
	ap.add_argument('--external', default='cpz_gan_ensemble', help='key in the config\'s external_sdfs')
	ap.add_argument('--out', required=True)
	args = ap.parse_args()
	cfg = load_config(args.config)
	train, valid, test = load_splits(cfg)
	sv = cfg['state_vars']
	M_ext = external_sdf(cfg['external_sdfs'][args.external], [train.T, valid.T, test.T])[2]
	F_ext = 1.0 - M_ext
	sr_ext = F_ext.mean() / F_ext.std()
	b_te = test.to_torch(sv)
	rows = []
	for rung in RUNGS:
		if not os.path.exists(os.path.join(args.models, 'model_%s.pt' % rung)):
			continue
		ev = _ensemble_eval(load_models(args.models, rung, train, sv, cfg), b_te)
		F = np.asarray(ev['F'], dtype=float)
		a, t, r2 = spanning_test(F_ext, F)
		a_rev, t_rev, _ = spanning_test(F, F_ext)
		rows.append({'model': rung, 'SR_test': ev['sharpe'], 'SR2_gap_vs_external': sr_ext ** 2 - ev['sharpe'] ** 2,
			'alpha_external_on_rung': a, 't': t, 'R2': r2, 'alpha_rung_on_external': a_rev, 't_rev': t_rev,
			'corr': np.corrcoef(F, F_ext)[0, 1]})
	os.makedirs(args.out, exist_ok=True)
	df = pd.DataFrame(rows)
	df.to_csv(os.path.join(args.out, 'spanning_test.csv'), index=False)
	print('external SDF %s: test SR %.3f' % (args.external, sr_ext))
	print(df.round(3).to_string(index=False))


if __name__ == '__main__':
	main()
