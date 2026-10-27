"""Download Fama-French 5 factors (2x3) and momentum from the Kenneth French data library.

	python scripts/download_ff_factors.py --config config_glassbox/cpz.json

Writes datasets/factors/ff5_umd.csv with a `date` column equal to the panel's npz dates (matched by
month), factor excess returns in decimals (the library reports percent), and RF. Factors are matched
by YYYYMM, so the panel's month-end trading dates need no calendar logic.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.data import load_config

BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
FILES = {'ff5': 'F-F_Research_Data_5_Factors_2x3_CSV.zip', 'mom': 'F-F_Momentum_Factor_CSV.zip'}


def monthly_table(raw):
	"""Parse the monthly block of a French CSV: rows whose first field is a 6-digit YYYYMM."""
	lines = raw.decode('latin-1').splitlines()
	header = None
	rows = []
	for line in lines:
		fields = [f.strip() for f in line.split(',')]
		if header is None:
			if len(fields) > 1 and fields[0] == '' and all(fields[1:]):
				header = fields[1:]
			continue
		if fields[0].isdigit() and len(fields[0]) == 6:
			rows.append([int(fields[0])] + [float(x) for x in fields[1:]])
		elif rows:
			break	# end of the monthly block (annual factors follow)
	df = pd.DataFrame(rows, columns=['yyyymm'] + header).set_index('yyyymm')
	return df / 100.0


def fetch(name):
	with urllib.request.urlopen(BASE + name, timeout=60) as r:
		z = zipfile.ZipFile(io.BytesIO(r.read()))
	return z.read(z.namelist()[0])


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', default='config_glassbox/cpz.json')
	ap.add_argument('--out', default='datasets/factors/ff5_umd.csv')
	args = ap.parse_args()
	cfg = load_config(args.config)
	ff5 = monthly_table(fetch(FILES['ff5']))
	mom = monthly_table(fetch(FILES['mom']))
	mom.columns = ['UMD']
	fac = ff5.join(mom, how='inner')

	dates = np.concatenate([np.load(cfg[k], allow_pickle=True)['date'] for k in
		('individual_feature_file', 'individual_feature_file_valid', 'individual_feature_file_test')]).astype(int)
	yyyymm = dates // 100
	missing = sorted(set(yyyymm) - set(fac.index))
	if missing:
		raise SystemExit('factor data missing for months %s ... (%d)' % (missing[:5], len(missing)))
	out = fac.loc[yyyymm].reset_index(drop=True)
	out.insert(0, 'date', dates)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	out.to_csv(args.out, index=False)
	print('wrote %s: %d months %d-%d, columns %s' % (args.out, len(out), dates[0], dates[-1], list(out.columns[1:])))


if __name__ == '__main__':
	main()
