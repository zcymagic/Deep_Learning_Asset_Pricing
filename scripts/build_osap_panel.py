"""Build a CPZ-style panel (1967-2024) from CRSP monthly returns + Open Source Asset Pricing signals.

	python scripts/build_osap_panel.py --crsp crsp_download/crsp_msf_1961_2024.csv.gz \
		--osap osap_download/signals --goyal goyal_classic_download/Data2025.xlsx --out datasets/osap

Output (same npz format as datasets/char and datasets/macro, so every glassbox script runs unchanged):
	Char_{train,valid,test,post}.npz   train 1967-86, valid 1987-91, test 1992-2016, post 2017-2024
	macro_{train,valid,test,post}.npz  Welch-Goyal monthly series with full 1967-2024 coverage
	build_meta.json                    characteristic mapping, stocks per month, dropped macro series

Conventions, matched to the CPZ files (verified on datasets/char): row t holds the excess return of
month t and characteristics known at the end of month t-1; ST_REV is the month t-1 return and r2_1 the
month t-2 return; characteristics are rank-normalized to [-0.5, 0.5] each month; by default only stocks
with every characteristic observed are kept (CPZ have no median-imputed values). --impute median instead
keeps stocks with at least --min_observed of the characteristics and sets the missing ranks to 0, the
cross-sectional median (Gu, Kelly and Xiu 2020). If the CRSP export has vol and cfacshr, LTurnover, SUV
and NI are computed from CRSP instead of OSAP (VolMkt, VolumeTrend, ShareIss1Y). OSAP signals dated yyyymm are
known at the end of that month (OSAP convention), so they enter row yyyymm + 1. Macro series also enter
with a one-month lag. Delisting returns: combined with RET; when missing for a performance delisting
(codes 500, 520-584) a -30% return is used (Shumway 1997).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_ff_factors import FILES, fetch, monthly_table

UNK = -99.99

# CPZ characteristic -> source. 'crsp:*' are computed from CRSP returns / prices; the rest are OSAP files.
MAPPING = {
	# past returns (exact CPZ definitions from CRSP)
	'r2_1': 'crsp:ret_lag2', 'r12_2': 'crsp:cum_12_2', 'r12_7': 'crsp:cum_12_7', 'r36_13': 'crsp:cum_36_13',
	'ST_REV': 'crsp:ret_lag1', 'LT_Rev': 'crsp:cum_60_13',
	# trading frictions
	'LME': 'crsp:log_me', 'AT': 'osap:AM*ME', 'Beta': 'Beta', 'MktBeta': 'BetaFP', 'IdioVol': 'IdioVol3F',
	'Resid_Var': 'IdioVolAHT', 'Variance': 'RealizedVol', 'Spread': 'BidAskSpread', 'Rel2High': 'High52',
	'LTurnover': 'VolMkt', 'SUV': 'VolumeTrend',
	# value
	'A2ME': 'AM', 'BEME': 'BM', 'C': 'Cash', 'CF': 'CF', 'CF2P': 'cfp', 'D2P': 'DivYieldAnn', 'E2P': 'EP',
	'Lev': 'Leverage', 'S2P': 'SP',
	# profitability
	'ATO': 'AssetTurnover', 'CTO': 'CapTurnover', 'D2A': 'depr', 'OP': 'OperProf', 'PM': 'PM', 'PROF': 'GP',
	'RNA': 'RetNOA', 'ROA': 'roaq', 'ROE': 'RoE',
	# investment
	'Investment': 'AssetGrowth', 'NOA': 'NOA', 'DPI2A': 'InvestPPEInv', 'NI': 'ShareIss1Y',
	# intangibles
	'AC': 'Accruals', 'OA': 'PctAcc', 'OL': 'OPLeverage',
}
NO_COUNTERPART = ['Q', 'SGA2S', 'FC2Y', 'PCM']		# CPZ characteristics without an OSAP / CRSP counterpart
SPLITS = {'train': (196701, 198612), 'valid': (198701, 199112), 'test': (199201, 201612), 'post': (201701, 202412)}


def month_index(yyyymm):
	return (np.asarray(yyyymm) // 100) * 12 + (np.asarray(yyyymm) % 100) - 1


def index_to_yyyymm(m):
	return (m // 12) * 100 + m % 12 + 1


def load_crsp(path):
	c = pd.read_csv(path)
	c.columns = c.columns.str.lower()
	for k in ('ret', 'dlret'):
		c[k] = pd.to_numeric(c[k], errors='coerce')	# CRSP letter codes -> missing
	ret, dl = c['ret'], c['dlret']
	perf = c['dlstcd'].between(520, 584) | (c['dlstcd'] == 500)
	dl = dl.where(dl.notna() | ~perf, -0.30)
	c['ret_adj'] = np.where(ret.notna() & dl.notna(), (1 + ret) * (1 + dl) - 1, ret.fillna(dl))
	c['me'] = c['prc'].abs() * c['shrout']
	c.loc[c['me'] <= 0, 'me'] = np.nan
	c['m'] = month_index(c['yyyymm'])
	keep = ['permno', 'yyyymm', 'm', 'date', 'ret_adj', 'me']
	if 'vol' in c and 'cfacshr' in c:
		c['turn'] = c['vol'] * 100 / (c['shrout'] * 1000)	# msf: vol in 100s of shares, shrout in 1000s
		c.loc[c['turn'] <= 0, 'turn'] = np.nan
		c['adj_shares'] = c['shrout'] * c['cfacshr']
		c.loc[c['adj_shares'] <= 0, 'adj_shares'] = np.nan
		keep += ['turn', 'adj_shares']
	return c[keep]


def crsp_characteristics(c):
	"""Past-return and size characteristics for every (permno, return month t), from wide matrices."""
	permnos = np.sort(c['permno'].unique())
	months = np.arange(c['m'].min(), c['m'].max() + 1)
	col = {p: i for i, p in enumerate(permnos)}
	ii = c['permno'].map(col).values
	tt = (c['m'] - months[0]).values
	R = np.full((len(months), len(permnos)), np.nan)
	ME = np.full_like(R, np.nan)
	R[tt, ii] = c['ret_adj'].values
	ME[tt, ii] = c['me'].values
	logR = np.log1p(R)

	def cum(lo, hi):
		"""Cumulative return over months t-hi .. t-lo (all months required)."""
		out = np.full_like(R, np.nan)
		width = hi - lo + 1
		cs = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.nan_to_num(logR), axis=0)])
		cnt = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.isfinite(logR), axis=0)])
		for t in range(hi, len(months)):
			a, b = t - hi, t - lo + 1
			s, n = cs[b] - cs[a], cnt[b] - cnt[a]
			out[t] = np.where(n == width, np.expm1(s), np.nan)
		return out

	def lag(X, k):
		out = np.full_like(X, np.nan)
		out[k:] = X[:-k]
		return out

	feats = {
		'crsp:ret_lag1': lag(R, 1), 'crsp:ret_lag2': lag(R, 2), 'crsp:cum_12_2': cum(2, 12), 'crsp:cum_12_7': cum(7, 12),
		'crsp:cum_36_13': cum(13, 36), 'crsp:cum_60_13': cum(13, 60), 'crsp:log_me': np.log(lag(ME, 1)),
	}
	if 'turn' in c:
		TURN = np.full_like(R, np.nan)
		SH = np.full_like(R, np.nan)
		TURN[tt, ii] = c['turn'].values
		SH[tt, ii] = c['adj_shares'].values
		lt = np.log(TURN)
		feats['crsp:log_turnover'] = lag(lt, 1)
		# monthly volume surprise: log turnover in t-1 standardized by its mean / std over t-13 .. t-2
		mu = np.full_like(R, np.nan)
		sd = np.full_like(R, np.nan)
		for t in range(13, len(months)):
			w = lt[t - 13:t - 1]
			ok = np.isfinite(w).sum(0) >= 6
			mu[t] = np.where(ok, np.nanmean(np.where(np.isfinite(w), w, np.nan), 0), np.nan)
			sd[t] = np.where(ok, np.nanstd(np.where(np.isfinite(w), w, np.nan), 0), np.nan)
		feats['crsp:volume_surprise'] = (lag(lt, 1) - mu) / np.where(sd > 0, sd, np.nan)
		# net share issuance over the year to t-1, split adjusted
		feats['crsp:share_issuance'] = np.log(lag(SH, 1)) - np.log(lag(SH, 13))
	me_lag1 = lag(ME, 1)
	long = c[['permno', 'm', 'yyyymm', 'date', 'ret_adj']].copy()
	for k, X in feats.items():
		long[k] = X[tt, ii]
	long['me_lag1'] = me_lag1[tt, ii]
	return long


def load_osap(osap_dir, name):
	d = pd.read_csv(os.path.join(osap_dir, name + '.csv'))
	v = d.columns[2]
	d = d.rename(columns={v: name})
	d[name] = d[name].replace([np.inf, -np.inf], np.nan)
	d['m'] = month_index(d['yyyymm']) + 1		# known at end of yyyymm -> enters the next month's row
	return d[['permno', 'm', name]]


def rank_normalize(df, cols, impute=False):
	g = df.groupby('m')
	for c in cols:
		r = g[c].rank(method='average')
		n = g[c].transform('count')
		df[c] = ((r - 1) / (n - 1).clip(lower=1) - 0.5).astype(np.float32)
		if impute:
			df[c] = df[c].fillna(0.0)
	return df


def goyal_macro(path):
	g = pd.read_excel(path, sheet_name='Monthly')
	g = g.rename(columns={'yyyymm': 'yyyymm'})
	g['m'] = month_index(g['yyyymm'].astype(int)) + 1	# one-month lag
	return g.drop(columns=['yyyymm'])


def save_split(name, panel, chars, macro_df, macro_vars, out_dir, rf):
	lo, hi = SPLITS[name]
	sub = panel[(panel['yyyymm'] >= lo) & (panel['yyyymm'] <= hi)]
	months = np.arange(month_index(lo), month_index(hi) + 1)
	permnos = np.sort(sub['permno'].unique())
	col = {p: i for i, p in enumerate(permnos)}
	data = np.full((len(months), len(permnos), 1 + len(chars)), UNK, dtype=np.float32)
	ti = (sub['m'] - months[0]).values
	ii = sub['permno'].map(col).values
	data[ti, ii, 0] = (sub['ret_adj'] - sub['m'].map(rf)).values
	data[ti, ii, 1:] = sub[chars].values
	dates = sub.groupby('m')['date'].max().reindex(months)
	dates = np.where(dates.notna(), dates, index_to_yyyymm(months) * 100 + 28).astype(int)	# empty months: placeholder day
	np.savez(os.path.join(out_dir, 'Char_%s.npz' % name), data=data, date=dates,
		variable=np.array(['RET'] + chars), permno=permnos)
	mac = macro_df.set_index('m').reindex(months)[macro_vars].values.astype(np.float64)
	np.savez(os.path.join(out_dir, 'macro_%s.npz' % name), data=mac, date=dates, variable=np.array(macro_vars))
	n = (data[:, :, 0] != UNK).sum(1)
	return {'months': len(months), 'stocks_total': len(permnos), 'stocks_per_month_min': int(n.min()),
		'stocks_per_month_median': int(np.median(n)), 'stocks_per_month_max': int(n.max())}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--crsp', default='crsp_download/crsp_msf_1961_2024.csv.gz')
	ap.add_argument('--osap', default='osap_download/signals')
	ap.add_argument('--goyal', default='goyal_classic_download/Data2025.xlsx')
	ap.add_argument('--out', default='datasets/osap')
	ap.add_argument('--allow_missing_signals', action='store_true',
		help='drop characteristics whose OSAP file is not present instead of failing')
	ap.add_argument('--impute', choices=['none', 'median'], default='none')
	ap.add_argument('--min_observed', type=float, default=0.67,
		help='with --impute median: minimum share of characteristics observed for a stock-month')
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)

	mapping = dict(MAPPING)
	header = pd.read_csv(args.crsp, nrows=0).columns.str.lower()
	if 'vol' in header and 'cfacshr' in header:
		mapping.update({'LTurnover': 'crsp:log_turnover', 'SUV': 'crsp:volume_surprise', 'NI': 'crsp:share_issuance'})
	osap_names = sorted({v for v in mapping.values() if not v.startswith('crsp:') and v != 'osap:AM*ME'} | {'AM'})
	missing = [n for n in osap_names if not os.path.exists(os.path.join(args.osap, n + '.csv'))
		or os.path.getsize(os.path.join(args.osap, n + '.csv')) < 100000]
	if missing and not args.allow_missing_signals:
		raise SystemExit('missing OSAP files: %s (or pass --allow_missing_signals)' % missing)
	dropped = [k for k, v in mapping.items() if v in missing]
	for k in dropped:
		del mapping[k]
	if 'AM' in missing:
		mapping.pop('AT', None)
		dropped.append('AT')

	print('[build] CRSP')
	c = load_crsp(args.crsp)
	panel = crsp_characteristics(c)
	panel = panel[panel['ret_adj'].notna() & panel['me_lag1'].notna()]
	for name in osap_names:
		if name in missing:
			continue
		print('[build] OSAP', name)
		panel = panel.merge(load_osap(args.osap, name), on=['permno', 'm'], how='left')
	if 'AT' in mapping:
		with np.errstate(divide='ignore', invalid='ignore'):
			panel['osap:AM*ME'] = np.log(panel['AM'] * panel['me_lag1'])
	chars = list(mapping)
	for k, v in mapping.items():
		panel[k] = panel[v].replace([np.inf, -np.inf], np.nan)
	panel = panel[(panel['yyyymm'] >= SPLITS['train'][0]) & (panel['yyyymm'] <= SPLITS['post'][1])]
	n_before = panel.groupby('m').size()
	if args.impute == 'none':
		panel = panel.dropna(subset=chars).copy()		# complete cases, as in CPZ
	else:
		share = panel[chars].notna().mean(axis=1)
		panel = panel[(share >= args.min_observed) & panel[['ST_REV', 'LME']].notna().all(axis=1)].copy()
	panel = rank_normalize(panel, chars, impute=args.impute == 'median')

	ff = monthly_table(fetch(FILES['ff5']))
	rf = pd.Series(ff['RF'].values, index=month_index(ff.index.values))

	macro = goyal_macro(args.goyal)
	months = np.arange(month_index(SPLITS['train'][0]), month_index(SPLITS['post'][1]) + 1)
	cov = macro.set_index('m').reindex(months)
	levels = {'price', 'd12', 'e12'}		# non-stationary index / dividend / earnings levels
	macro_vars = [v for v in cov.columns if v not in levels and pd.api.types.is_numeric_dtype(cov[v]) and cov[v].notna().all()]
	dropped_macro = [v for v in cov.columns if v not in macro_vars]

	meta = {'mapping': mapping, 'impute': args.impute, 'min_observed': args.min_observed, 'used_characteristics': chars, 'dropped_characteristics': dropped + NO_COUNTERPART,
		'missing_osap_files': missing, 'macro_variables': macro_vars, 'dropped_macro_incomplete_coverage': dropped_macro,
		'complete_case_share': float(len(panel) / n_before.sum()), 'splits': {}}
	for name in SPLITS:
		meta['splits'][name] = save_split(name, panel, chars, macro, macro_vars, args.out, rf)
		print('[build] %s %s' % (name, meta['splits'][name]))
	with open(os.path.join(args.out, 'build_meta.json'), 'w') as f:
		json.dump(meta, f, indent=2)
	print('[build] %d characteristics, %d macro series -> %s' % (len(chars), len(macro_vars), args.out))


if __name__ == '__main__':
	main()
