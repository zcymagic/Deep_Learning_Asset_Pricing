"""Alpha of the short-term-reversal failure rules over time (CRSP + OSAP panel, 1967-2024).

	python scripts/plot_rule_alpha.py --out docs/osap_results

For each rule, the equal-weighted managed-portfolio return x_t (audit.rule_returns) is compared with
FF5 + UMD in two ways:
	rolling     60-month OLS alpha of x on the factors, with +/- 2 standard-error bands
	cumulative  sum of the pricing error x_t - b'F_t with b estimated on 1967-1986 only and then held
	            fixed (no look-ahead); its slope at any date is the alpha at that date
Writes rule_alpha.png and rule_alpha.csv (monthly series) plus a subperiod table rule_alpha_periods.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox import audit
from glassbox.audit import Condition, Rule
from glassbox.data import concat_time, load_config, load_npz_panel
from glassbox.pricing import newey_west_se

RULES = {
	'Recent winners with weak momentum': Rule((Condition('ST_REV', 'char', '>', 0.3), Condition('r12_2', 'char', '<', -0.3))),
	'Recent winners far below 52-week high': Rule((Condition('ST_REV', 'char', '>', 0.3), Condition('Rel2High', 'char', '<', -0.1))),
}
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']
PERIODS = [('1967-1986 (train)', 196701, 198612), ('1987-2016 (valid + test)', 198701, 201612), ('2017-2024 (post)', 201701, 202412)]
COLORS = ['#2a78d6', '#eb6834']		# validated categorical slots 1-2 (light surface)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load_full_panel(cfg, post_cfg):
	parts = []
	mu = sd = None
	for key, c in (('individual_feature_file', cfg), ('individual_feature_file_valid', cfg),
			('individual_feature_file_test', cfg), ('individual_feature_file_test', post_cfg)):
		p, mu, sd = load_npz_panel(c[key], None, None, mu, sd)
		parts.append(p)
	return concat_time(parts)


def load_factors(paths, dates):
	f = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates('date').set_index('date')
	return f.loc[dates.astype(int), FACTORS].values


def rolling_alpha(x, F, window=60):
	T = len(x)
	a, se = np.full(T, np.nan), np.full(T, np.nan)
	for t in range(window - 1, T):
		y = x[t - window + 1:t + 1]
		X = np.column_stack([np.ones(window), F[t - window + 1:t + 1]])
		coef, *_ = np.linalg.lstsq(X, y, rcond=None)
		e = y - X @ coef
		s2 = e @ e / (window - X.shape[1])
		a[t] = coef[0]
		se[t] = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
	return a, se


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', default='config_glassbox/osap.json')
	ap.add_argument('--post_config', default='config_glassbox/osap_post.json')
	ap.add_argument('--out', default='docs/osap_results')
	ap.add_argument('--window', type=int, default=60)
	args = ap.parse_args()
	cfg, post_cfg = load_config(args.config), load_config(args.post_config)
	panel = load_full_panel(cfg, post_cfg)
	ff_paths = sorted({s['path'] for c in (cfg, post_cfg) for s in c['factor_files'].values()})
	F = load_factors(ff_paths, panel.dates)
	ym = panel.dates.astype(int) // 100
	when = pd.to_datetime(ym.astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
	X, active, n_stocks = audit.rule_returns(panel, list(RULES.values()), min_stocks=30)

	train = (ym >= 196701) & (ym <= 198612)
	series, periods = {'date': ym}, []
	for j, name in enumerate(RULES):
		x = X[:, j]
		a, se = rolling_alpha(x, F, args.window)
		Xtr = np.column_stack([np.ones(train.sum()), F[train]])
		b = np.linalg.lstsq(Xtr, x[train], rcond=None)[0][1:]
		pe = x - F @ b							# pricing error with betas fixed on 1967-86
		series[name + ' | rolling alpha'] = a
		series[name + ' | rolling se'] = se
		series[name + ' | pricing error'] = pe
		for label, lo, hi in PERIODS:
			m = (ym >= lo) & (ym <= hi)
			periods.append({'rule': name, 'period': label, 'alpha_pct_per_month': 100 * pe[m].mean(),
				't_nw': pe[m].mean() / newey_west_se(pe[m]), 'months': int(m.sum()), 'avg_stocks': float(n_stocks[m, j].mean())})
	pd.DataFrame(series).to_csv(os.path.join(args.out, 'rule_alpha.csv'), index=False)
	per = pd.DataFrame(periods)
	per.to_csv(os.path.join(args.out, 'rule_alpha_periods.csv'), index=False)
	print(per.round(3).to_string(index=False))

	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
		'ytick.color': MUTED, 'text.color': INK, 'axes.titlecolor': INK})
	fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, facecolor=SURFACE,
		gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.18})
	for ax in axes:
		ax.set_facecolor(SURFACE)
		ax.grid(axis='y', color=GRID, lw=0.8)
		ax.axhline(0, color=MUTED, lw=0.8)
		for spine in ('top', 'right'):
			ax.spines[spine].set_visible(False)
		ax.axvspan(pd.Timestamp('1987-01-01'), pd.Timestamp('2016-12-31'), color='#f1f0ec', lw=0, zorder=0)
		ax.axvline(pd.Timestamp('2017-01-01'), color=MUTED, lw=0.8, ls=(0, (3, 3)))
	for j, name in enumerate(RULES):
		a = 100 * series[name + ' | rolling alpha']
		se = 100 * series[name + ' | rolling se']
		axes[0].fill_between(when, a - 2 * se, a + 2 * se, color=COLORS[j], alpha=0.14, lw=0)
		axes[0].plot(when, a, color=COLORS[j], lw=2, label=name)
		cum = 100 * np.cumsum(series[name + ' | pricing error'])
		axes[1].plot(when, cum, color=COLORS[j], lw=2, label=name)
		axes[1].annotate(name, (when[-1], cum[-1]), xytext=(6, 0),
			textcoords='offset points', va='center', fontsize=8, color=INK)
	axes[0].set_ylabel('Rolling %d-month alpha\n(%% per month, ±2 SE)' % args.window)
	axes[1].set_ylabel('Cumulative pricing error (%)\nbetas fixed on 1967–1986')
	axes[0].legend(loc='lower left', frameon=False, fontsize=8)
	ymax = axes[0].get_ylim()[1]
	axes[0].text(pd.Timestamp('1987-06-01'), ymax, 'CPZ validation + test (1987–2016)', va='top', fontsize=8, color=MUTED)
	axes[0].text(pd.Timestamp('2017-06-01'), ymax, '2017–2024', va='top', fontsize=8, color=MUTED)
	fig.suptitle('Short-term reversal failure rules: alpha relative to FF5 + UMD, 1967–2024', x=0.06, ha='left',
		fontsize=11, color=INK)
	fig.text(0.06, 0.005, 'Equal-weighted portfolios of stocks in each rule (CRSP + OSAP panel, ranks in [-0.5, 0.5]); '
		'shaded band = 1987–2016, dashed line = start of the post-2016 sample.', fontsize=7, color=MUTED)
	fig.subplots_adjust(left=0.1, right=0.78, top=0.92, bottom=0.07)
	path = os.path.join(args.out, 'rule_alpha.png')
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	print('wrote', path)


if __name__ == '__main__':
	main()
