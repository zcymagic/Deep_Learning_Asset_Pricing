"""Collapse the 153-characteristic JKP panel to JKP's 13 themes.

	python scripts/build_jkp_theme_panel.py --src datasets/jkp_panel --out datasets/jkp_themes

Each month, a stock's theme score is the mean of direction x rank over the theme's characteristics
(glassbox.jkp_themes: THEMES, DIRECTION), counting only characteristics observed for that stock (rank != 0).
Signing by JKP's direction makes every member point the same way (high = the side the original study
found to earn higher returns), as in JKP's theme portfolios. Scores are re-ranked to [-0.5, 0.5] among
stocks with at least one observed member; a stock with none gets 0 (the median). Returns, dates, permnos
and macro files are copied unchanged.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.jkp_themes import DIRECTION, THEMES

UNK = -99.99


def theme_scores(data, names):
	T, N, _ = data.shape
	out = np.full((T, N, 1 + len(THEMES)), UNK, dtype=np.float32)
	out[:, :, 0] = data[:, :, 0]
	for t in range(T):
		live = data[t, :, 0] != UNK
		Z = data[t, live, 1:]
		for j, (theme, chars) in enumerate(THEMES.items()):
			idx = [names.index(c) for c in chars]
			sgn = np.array([DIRECTION[c] for c in chars], dtype=np.float32)
			X = Z[:, idx] * sgn
			obs = Z[:, idx] != 0
			n = obs.sum(1)
			score = np.zeros(live.sum(), dtype=np.float32)
			has = n > 0
			score[has] = X[has].sum(1) / n[has]
			if has.sum() > 1:
				score[has] = (rankdata(score[has]) - 1) / (has.sum() - 1) - 0.5
			out[t, live, 1 + j] = score
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--src', default='datasets/jkp_panel')
	ap.add_argument('--out', default='datasets/jkp_themes')
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)
	meta = {'source': args.src, 'themes': {k: v for k, v in THEMES.items()}, 'splits': {}}
	for split in ('train', 'valid', 'test', 'post'):
		path = os.path.join(args.src, 'Char_%s.npz' % split)
		if not os.path.exists(path):
			continue
		d = np.load(path)
		names = [str(v) for v in d['variable'][1:]]
		data = theme_scores(d['data'], names)
		np.savez_compressed(os.path.join(args.out, 'Char_%s.npz' % split), data=data, date=d['date'],
			variable=np.array(['RET'] + list(THEMES)), permno=d['permno'])
		shutil.copy(os.path.join(args.src, 'macro_%s.npz' % split), os.path.join(args.out, 'macro_%s.npz' % split))
		meta['splits'][split] = {'months': int(data.shape[0]), 'stocks_total': int(data.shape[1])}
		print(split, data.shape, flush=True)
		del d, data
	with open(os.path.join(args.out, 'build_meta.json'), 'w') as f:
		json.dump(meta, f, indent=2)


if __name__ == '__main__':
	main()
