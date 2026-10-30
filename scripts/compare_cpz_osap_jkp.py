"""Cross-validate the CPZ and the rebuilt CRSP + OSAP characteristics against Jensen, Kelly & Pedersen (2023).

	python scripts/compare_cpz_osap_jkp.py --jkp datasets/jkp/jkp_sample.npz --out docs/osap_results

JKP is an independent third construction (their "Is there a replication crisis in finance?" data), with CRSP
permnos. When CPZ and OSAP disagree on a characteristic, JKP breaks the tie: the source that agrees with
JKP's version of the same characteristic is likely the one built to the standard definition.

Stocks: CPZ and OSAP stock-months are matched on the excess return (compare_cpz_osap_chars.match_on_return);
OSAP supplies the permno, which joins JKP on (permno, month). JKP rows are lagged like CPZ rows (return of
month t, characteristics known at the end of t-1); the join is checked by comparing JKP's excess return with
OSAP's. Only the months in the JKP sample (June and December) are used.

Each comparison is the month-by-month Spearman correlation on the three-way matched stocks, so differences
in universe and rank scaling do not matter. Cells set to the cross-sectional median by imputation (rank
exactly 0, in OSAP or JKP) are left out.

For every CPZ characteristic the table gives
	jkp           the JKP characteristic with the same definition (DESIGNATED), or the best match if none
	cpz_jkp       rho(CPZ, JKP)       osap_jkp  rho(OSAP, JKP)       cpz_osap  rho(CPZ, OSAP)
	cpz_best      the JKP characteristic closest to CPZ's, and its rho; osap_best likewise for OSAP
	verdict       which source agrees with JKP (difference of at least 0.10), otherwise 'agree' or 'all differ'
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_cpz_osap_chars import UNK, SPLITS, load, match_on_return

# JKP counterpart by definition (None: no clean counterpart; the best-matching JKP characteristic is used)
DESIGNATED = {
	'ST_REV': 'ret_1_0', 'r2_1': None, 'r12_2': 'ret_12_1', 'r12_7': 'ret_12_7', 'r36_13': 'ret_36_12',
	'LT_Rev': 'ret_60_12', 'LME': 'market_equity', 'AT': 'assets', 'Beta': 'beta_60m', 'MktBeta': None,
	'IdioVol': 'ivol_ff3_21d', 'Resid_Var': None, 'Variance': 'rvol_21d', 'Spread': 'bidaskhl_21d',
	'Rel2High': 'prc_highprc_252d', 'LTurnover': 'turnover_126d', 'SUV': None,
	'A2ME': 'at_me', 'BEME': 'be_me', 'C': 'cash_at', 'CF': 'fcf_be', 'CF2P': 'ocf_me', 'D2P': 'div12m_me',
	'E2P': 'ni_me', 'Lev': None, 'S2P': 'sale_me',
	'ATO': None, 'CTO': None, 'D2A': None, 'OP': 'ope_be', 'PM': None, 'PROF': None, 'RNA': None,
	'ROA': 'ni_at', 'ROE': 'ni_be',
	'Investment': 'at_gr1', 'NOA': 'noa_at', 'DPI2A': 'ppeinv_gr1a', 'NI': 'chcsho_12m', 'AC': None,
	'OA': 'oaccruals_at', 'OL': None,
}


def nanrank(A):
	"""Column-wise average ranks, NaN kept as NaN."""
	return pd.DataFrame(A).rank().values


def rank_corr(rx, RY):
	"""Correlation of the rank vector rx with each column of RY over the pairs where both are observed.

	Ranks are taken once over each full column (nanrank), not re-taken within each pair set, which is a
	close approximation to the pairwise Spearman correlation and much faster.
	"""
	M = np.isfinite(RY) & np.isfinite(rx)[:, None]
	n = M.sum(0)
	X = np.where(M, rx[:, None], 0.0)
	Y = np.where(M, RY, 0.0)
	mx, my = X.sum(0) / np.maximum(n, 1), Y.sum(0) / np.maximum(n, 1)
	cov = (X * Y).sum(0) / np.maximum(n, 1) - mx * my
	vx = (X * X).sum(0) / np.maximum(n, 1) - mx ** 2
	vy = (Y * Y).sum(0) / np.maximum(n, 1) - my ** 2
	with np.errstate(invalid='ignore', divide='ignore'):
		r = cov / np.sqrt(vx * vy)
	r[n < 30] = np.nan
	return r


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cpz', default='datasets/char')
	ap.add_argument('--osap', default='datasets/osap')
	ap.add_argument('--jkp', default='datasets/jkp/jkp_sample.npz')
	ap.add_argument('--out', default='docs/osap_results')
	ap.add_argument('--tol', type=float, default=1e-5)
	args = ap.parse_args()

	j = np.load(args.jkp, allow_pickle=True)
	jv = [str(v) for v in j['variable']]
	jkey = pd.Series(np.arange(len(j['permno'])), index=pd.MultiIndex.from_arrays([j['permno'], j['yyyymm']]))
	jkey = jkey[~jkey.index.duplicated()]
	J_all = j['data']
	months = set(np.unique(j['yyyymm']).tolist())

	acc = {}			# char -> {'cpz': [months x n_jkp], 'osap': [...], 'cpz_osap': [...]}
	n_three, ret_gap = 0, []
	for s in SPLITS:
		dc, tc, vc = load(os.path.join(args.cpz, 'Char_%s.npz' % s))
		do, to, vo = load(os.path.join(args.osap, 'Char_%s.npz' % s))
		permno = np.load(os.path.join(args.osap, 'Char_%s.npz' % s))['permno']
		pos_o = {t // 100: k for k, t in enumerate(to)}
		common = [v for v in vc[1:] if v in vo[1:]]
		for k, t in enumerate(tc):
			ym = t // 100
			if ym not in months or ym not in pos_o:
				continue
			kc = np.where(dc[k][:, 0] != UNK)[0]
			ko = np.where(np.isfinite(do[pos_o[ym]][:, 0]) & (do[pos_o[ym]][:, 0] != UNK))[0]
			C, O = dc[k][kc], do[pos_o[ym]][ko]
			mc, mo = match_on_return(C, O, args.tol)
			idx = jkey.reindex(pd.MultiIndex.from_arrays([permno[ko[mo]], np.full(len(mo), ym)])).values
			has = np.isfinite(idx)
			if has.sum() < 30:
				continue
			Cm, Om = C[mc][has].astype(np.float64), O[mo][has].astype(np.float64)
			Jm = J_all[idx[has].astype(np.int64)].astype(np.float64)
			ret_gap.append(np.abs(j['ret_exc'][idx[has].astype(np.int64)] - Om[:, 0]))
			n_three += int(has.sum())
			Om[Om == 0] = np.nan
			Jm[Jm == 0] = np.nan
			RC, RO, RJ = nanrank(Cm), nanrank(Om), nanrank(Jm)
			for name in common:
				x = RC[:, vc.index(name)]
				y = RO[:, vo.index(name)]
				d = acc.setdefault(name, {'cpz': [], 'osap': [], 'cpz_osap': []})
				d['cpz'].append(rank_corr(x, RJ))
				d['osap'].append(rank_corr(y, RJ))
				d['cpz_osap'].append(rank_corr(x, y[:, None])[0])
		print('split %s done, three-way matched stock-months so far %d' % (s, n_three), flush=True)

	gap = np.concatenate(ret_gap)
	print('JKP vs OSAP excess return: share within 1e-4 = %.3f, within 1e-3 = %.3f'
		% ((gap < 1e-4).mean(), (gap < 1e-3).mean()))

	rows = []
	for name, d in acc.items():
		rc = np.nanmean(np.array(d['cpz']), axis=0)
		ro = np.nanmean(np.array(d['osap']), axis=0)
		bc, bo = int(np.nanargmax(np.abs(rc))), int(np.nanargmax(np.abs(ro)))
		des = DESIGNATED.get(name)
		jj = jv.index(des) if des in jv else bc
		a, b = rc[jj], ro[jj]
		if max(abs(a), abs(b)) < 0.6:
			verdict = 'all differ'
		elif abs(a) - abs(b) >= 0.10:
			verdict = 'CPZ agrees with JKP'
		elif abs(b) - abs(a) >= 0.10:
			verdict = 'OSAP agrees with JKP'
		else:
			verdict = 'agree'
		rows.append({'characteristic': name, 'jkp': jv[jj], 'designated': des is not None, 'cpz_jkp': a,
			'osap_jkp': b, 'cpz_osap': np.nanmean(d['cpz_osap']), 'cpz_best': jv[bc], 'cpz_best_rho': rc[bc],
			'osap_best': jv[bo], 'osap_best_rho': ro[bo], 'verdict': verdict})
	df = pd.DataFrame(rows)
	df['order'] = [list(DESIGNATED).index(c) if c in DESIGNATED else 99 for c in df['characteristic']]
	df = df.sort_values('order').drop(columns='order')
	os.makedirs(args.out, exist_ok=True)
	df.to_csv(os.path.join(args.out, 'char_crossvalidation_jkp.csv'), index=False)
	print('three-way matched stock-months %d' % n_three)
	print(df.round(2).to_string(index=False))


if __name__ == '__main__':
	main()
