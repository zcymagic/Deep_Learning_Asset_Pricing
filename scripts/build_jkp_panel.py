"""Build a glass-box SDF panel from the Jensen, Kelly & Pedersen (2023) characteristics.

	tar -xjOf kelly.tar.bz2 kelly.csv | python scripts/build_jkp_panel.py --out datasets/jkp_panel

Input: JKP's US stock-month file. One row per stock-month: date, id (CRSP permno), the month's excess
return (ret_exc) and characteristics lagged one month (suffix _l, known at the end of the previous
month), already ranked cross-sectionally to [-1, 1] with missing values set to 0. Same timing as CPZ.

Characteristics: the 153 of JKP's paper (JKP153, grouped into their 13 themes in glassbox/jkp_themes.py).
Stock-months need at least --min_observed of them observed (non-zero). Each month every characteristic
is re-ranked among the kept stocks with an observed value to [-0.5, 0.5] (CPZ scaling); missing values
are set to 0, the cross-sectional median (Gu, Kelly & Xiu 2020).

Splits match the OSAP panel; JKP's file starts in 1970, so the training split is 1970-1986:
	train 1970-1986, valid 1987-1991, test 1992-2016, post 2017-
Dates are taken from the factor files (CRSP last trading day of the month) so that factor SDFs line up.
Macro series are the Welch-Goyal series of the OSAP panel (datasets/osap/macro_*.npz), matched by month.
Output: Char_{split}.npz (data T x N x (1 + 153), date, variable, permno) and macro_{split}.npz, as in
datasets/osap, plus build_meta.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.jkp_themes import JKP153

UNK = -99.99
SPLITS = {'train': (197001, 198612), 'valid': (198701, 199112), 'test': (199201, 201612), 'post': (201701, 202412)}


def read_jkp(stream, start, chunksize):
	cols = ['date', 'id', 'ret_exc'] + [c + '_l' for c in JKP153]
	parts, n = [], 0
	for chunk in pd.read_csv(stream, usecols=cols, chunksize=chunksize,
			dtype={c + '_l': np.float32 for c in JKP153}):
		n += len(chunk)
		chunk['yyyymm'] = (chunk['date'] // 100).astype(np.int64)
		chunk = chunk[(chunk['yyyymm'] >= start) & chunk['ret_exc'].notna()]
		parts.append(chunk.drop(columns='date'))
		print('read %d rows' % n, file=sys.stderr, flush=True)
	df = pd.concat(parts, ignore_index=True)
	return df.rename(columns={c + '_l': c for c in JKP153})


def rerank(df):
	"""Re-rank each characteristic within month among observed (non-zero, finite) values to [-0.5, 0.5]."""
	X = df[JKP153].values.astype(np.float32)
	X[~np.isfinite(X)] = 0.0
	order = np.argsort(df['yyyymm'].values, kind='stable')
	ym = df['yyyymm'].values[order]
	bounds = np.flatnonzero(np.diff(ym)) + 1
	for rows in np.split(order, bounds):
		B = X[rows]
		for k in range(B.shape[1]):
			obs = B[:, k] != 0
			n = obs.sum()
			if n > 1:
				B[obs, k] = (rankdata(B[obs, k]) - 1) / (n - 1) - 0.5
		X[rows] = B
	df[JKP153] = X
	return df


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', default='datasets/jkp_panel')
	ap.add_argument('--factors', nargs='+', default=['datasets/factors/ff5_umd_osap.csv', 'datasets/factors/ff5_umd_osap_post.csv'])
	ap.add_argument('--macro_dir', default='datasets/osap')
	ap.add_argument('--min_observed', type=float, default=2 / 3)
	ap.add_argument('--chunksize', type=int, default=200000)
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)

	df = read_jkp(sys.stdin, SPLITS['train'][0], args.chunksize)
	n_raw = len(df)
	obs_share = (df[JKP153].fillna(0).values != 0).mean(1)
	df = df[obs_share >= args.min_observed].reset_index(drop=True)
	df = df.drop_duplicates(['id', 'yyyymm'])
	print('kept %d of %d stock-months with >= %.2f observed' % (len(df), n_raw, args.min_observed), file=sys.stderr)
	df = rerank(df)

	fac = pd.concat([pd.read_csv(f) for f in args.factors]).drop_duplicates('date')
	date_of = dict(zip(fac['date'].astype(int) // 100, fac['date'].astype(int)))
	mac_parts = [np.load(os.path.join(args.macro_dir, 'macro_%s.npz' % s), allow_pickle=True) for s in ('train', 'valid', 'test', 'post')]
	macro_vars = [str(v) for v in mac_parts[0]['variable']]
	mac = pd.DataFrame(np.concatenate([m['data'] for m in mac_parts]), columns=macro_vars,
		index=np.concatenate([m['date'] for m in mac_parts]).astype(int) // 100)

	meta = {'source': 'Jensen, Kelly & Pedersen (2023) US characteristics', 'characteristics': JKP153,
		'min_observed': args.min_observed, 'stock_months_raw': int(n_raw), 'stock_months_kept': int(len(df)), 'splits': {}}
	last = int(df['yyyymm'].max())
	for name, (lo, hi) in SPLITS.items():
		hi = min(hi, last)
		sub = df[(df['yyyymm'] >= lo) & (df['yyyymm'] <= hi)]
		months = sorted(m for m in sub['yyyymm'].unique() if m in date_of)
		if not months:
			continue
		sub = sub[sub['yyyymm'].isin(months)]
		permnos = np.sort(sub['id'].unique())
		ti = pd.Index(months).get_indexer(sub['yyyymm'])
		ii = pd.Index(permnos).get_indexer(sub['id'])
		data = np.full((len(months), len(permnos), 1 + len(JKP153)), UNK, dtype=np.float32)
		data[ti, ii, 0] = sub['ret_exc'].values
		data[ti, ii, 1:] = sub[JKP153].values
		dates = np.array([date_of[m] for m in months])
		np.savez_compressed(os.path.join(args.out, 'Char_%s.npz' % name), data=data, date=dates,
			variable=np.array(['RET'] + JKP153), permno=permnos)
		np.savez(os.path.join(args.out, 'macro_%s.npz' % name), data=mac.reindex(months).values, date=dates,
			variable=np.array(macro_vars))
		n = (data[:, :, 0] != UNK).sum(1)
		meta['splits'][name] = {'first': months[0], 'last': months[-1], 'months': len(months),
			'stocks_total': int(len(permnos)), 'stocks_per_month_median': int(np.median(n))}
		print(name, meta['splits'][name], file=sys.stderr, flush=True)
	with open(os.path.join(args.out, 'build_meta.json'), 'w') as f:
		json.dump(meta, f, indent=2)


if __name__ == '__main__':
	main()
