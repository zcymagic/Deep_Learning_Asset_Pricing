"""Consistency of the rebuilt CRSP + OSAP characteristics with the CPZ characteristics, 1967-2016.

	python scripts/compare_cpz_osap_chars.py --out docs/osap_results

The CPZ files carry no stock identifiers, so stocks are matched within each month on the excess return:
a CPZ stock-month and an OSAP stock-month match when their returns agree to 1e-5 and the return is unique
within the month in both panels. Both panels hold ranks in [-0.5, 0.5], but over different universes, so
ranks are compared through the month-by-month Spearman correlation on the matched stocks (invariant to the
universe if the underlying orderings agree). Cells the OSAP panel imputed with the cross-sectional median
(rank exactly 0) are left out of each characteristic's comparison.

Writes char_consistency.csv: per characteristic, the OSAP source, matched stock-months, the mean monthly
Spearman correlation (and its 10th/90th percentile over months), and the same mean for 1967-1986 and
1987-2016 separately.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UNK = -99.99
SPLITS = ['train', 'valid', 'test']


def match_on_return(C, O, tol):
	"""Row indices (ic, io) of stocks whose returns (column 0) agree to tol and are unique in both months."""
	key_c = np.round(C[:, 0].astype(np.float64) / tol).astype(np.int64)
	key_o = np.round(O[:, 0].astype(np.float64) / tol).astype(np.int64)
	uc, cc = np.unique(key_c, return_counts=True)
	uo, co = np.unique(key_o, return_counts=True)
	good = np.intersect1d(uc[cc == 1], uo[co == 1])
	ic = np.where(np.isin(key_c, good))[0]
	io = np.where(np.isin(key_o, good))[0]
	return ic[np.argsort(key_c[ic])], io[np.argsort(key_o[io])]


def load(path):
	d = np.load(path, allow_pickle=True)
	return d['data'], d['date'].astype(int), [str(v) for v in d['variable']]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cpz', default='datasets/char')
	ap.add_argument('--osap', default='datasets/osap')
	ap.add_argument('--out', default='docs/osap_results')
	ap.add_argument('--tol', type=float, default=1e-5)
	args = ap.parse_args()
	mapping = json.load(open(os.path.join(args.osap, 'build_meta.json')))['mapping']

	rows = {}			# char -> list of (yyyymm, rho, n)
	n_cpz = n_match = 0
	for s in SPLITS:
		dc, tc, vc = load(os.path.join(args.cpz, 'Char_%s.npz' % s))
		do, to, vo = load(os.path.join(args.osap, 'Char_%s.npz' % s))
		common = [v for v in vc[1:] if v in vo[1:]]
		ic = [vc.index(v) for v in common]
		io = [vo.index(v) for v in common]
		pos_o = {t // 100 if t > 10 ** 6 else t: k for k, t in enumerate(to)}
		for k, t in enumerate(tc):
			ym = t // 100 if t > 10 ** 6 else t
			if ym not in pos_o:
				continue
			C = dc[k][dc[k][:, 0] != UNK]
			O = do[pos_o[ym]]
			O = O[np.isfinite(O[:, 0]) & (O[:, 0] != UNK)]
			n_cpz += len(C)
			mc, mo = match_on_return(C, O, args.tol)
			if len(mc) < 30:
				continue
			Cm, Om = C[mc], O[mo]
			n_match += len(mc)
			for name, a, b in zip(common, ic, io):
				x, y = Cm[:, a].astype(np.float64), Om[:, b].astype(np.float64)
				ok = np.isfinite(y) & (y != 0.0)
				if ok.sum() < 30:
					continue
				rho = spearmanr(x[ok], y[ok]).correlation
				rows.setdefault(name, []).append((ym, rho, int(ok.sum())))

	out = []
	for name, r in rows.items():
		r = np.array(r)
		ym, rho, n = r[:, 0], r[:, 1], r[:, 2]
		early = ym <= 198612
		out.append({'characteristic': name, 'osap_source': mapping.get(name, ''), 'months': len(r),
			'matched_stock_months': int(n.sum()), 'mean_spearman': rho.mean(), 'p10': np.percentile(rho, 10),
			'p90': np.percentile(rho, 90), 'mean_1967_1986': rho[early].mean() if early.any() else np.nan,
			'mean_1987_2016': rho[~early].mean() if (~early).any() else np.nan})
	df = pd.DataFrame(out).sort_values('mean_spearman', ascending=False)
	os.makedirs(args.out, exist_ok=True)
	df.to_csv(os.path.join(args.out, 'char_consistency.csv'), index=False)
	print('CPZ stock-months %d, matched on return %d (%.1f%%)' % (n_cpz, n_match, 100 * n_match / max(n_cpz, 1)))
	print(df.round(3).to_string(index=False))


if __name__ == '__main__':
	main()
