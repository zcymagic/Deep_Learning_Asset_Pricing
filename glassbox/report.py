"""Minimal markdown tables (avoids a tabulate dependency)."""
import numpy as np


def md_table(df, digits=4):
	def fmt(v):
		if isinstance(v, (float, np.floating)):
			return '' if np.isnan(v) else ('%.*f' % (digits, v))
		return str(v)
	cols = list(df.columns)
	lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '---|' * len(cols)]
	for _, row in df.iterrows():
		lines.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
	return '\n'.join(lines)
