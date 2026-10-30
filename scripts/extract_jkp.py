"""Extract a monthly subsample of the Jensen-Kelly-Pedersen (2023) characteristics from their CSV.

	tar -xjOf kelly.tar.bz2 kelly.csv | python scripts/extract_jkp.py --out datasets/jkp/jkp_sample.npz

The CSV holds one row per stock-month: date (yyyymmdd), id (CRSP permno), the month's return and excess
return, and 405 characteristics lagged one month (suffix _l), i.e. known at the end of the previous month,
the same convention as a CPZ row. The full file does not fit in memory, so it is streamed in chunks and only
the months in --months (default June and December) of --start..--end are kept, as float32.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--start', type=int, default=196701)
	ap.add_argument('--end', type=int, default=201612)
	ap.add_argument('--months', default='6,12', help='calendar months to keep')
	ap.add_argument('--chunksize', type=int, default=200000)
	args = ap.parse_args()
	keep_months = {int(m) for m in args.months.split(',')}

	ids, yms, rets, blocks, names = [], [], [], [], None
	n_read = 0
	for chunk in pd.read_csv(sys.stdin, chunksize=args.chunksize, low_memory=False):
		n_read += len(chunk)
		ym = chunk['date'].values // 100
		sel = (ym >= args.start) & (ym <= args.end) & np.isin(ym % 100, list(keep_months))
		if names is None:
			names = [c for c in chunk.columns if c.endswith('_l')]
		if sel.any():
			c = chunk.loc[sel]
			ids.append(c['id'].values.astype(np.int64))
			yms.append(ym[sel].astype(np.int64))
			rets.append(c['ret_exc'].values.astype(np.float64))
			blocks.append(c[names].values.astype(np.float32))
		print('read %d rows, kept %d' % (n_read, sum(len(b) for b in blocks)), file=sys.stderr, flush=True)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	np.savez(args.out, permno=np.concatenate(ids), yyyymm=np.concatenate(yms), ret_exc=np.concatenate(rets),
		data=np.concatenate(blocks), variable=np.array([n[:-2] for n in names]))
	print('wrote', args.out, file=sys.stderr)


if __name__ == '__main__':
	main()
