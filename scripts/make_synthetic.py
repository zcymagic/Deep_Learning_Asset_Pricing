"""Write the synthetic validation panel in the repo's npz format (see glassbox/synthetic.py)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.data import save_npz_panel
from glassbox.synthetic import SyntheticSpec, make_synthetic_panel, split_panel


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', default='datasets/synthetic')
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--n_train', type=int, default=300)
	ap.add_argument('--n_valid', type=int, default=100)
	ap.add_argument('--null', action='store_true', help='no mispricing and no missing factor')
	args = ap.parse_args()
	spec = SyntheticSpec(seed=args.seed)
	if args.null:
		spec.mispricing, spec.missing_factor = 0.0, 0.0
	panel, truth = make_synthetic_panel(spec)
	os.makedirs(args.out, exist_ok=True)
	for name, part in zip(['train', 'valid', 'test'], split_panel(panel, args.n_train, args.n_valid)):
		save_npz_panel(part, os.path.join(args.out, 'Char_%s.npz' % name), os.path.join(args.out, 'macro_%s.npz' % name))
	np.save(os.path.join(args.out, 'M_oracle.npy'), truth['M_oracle'])
	with open(os.path.join(args.out, 'spec.json'), 'w') as f:
		json.dump(spec.__dict__, f, indent=2)
	print('wrote', args.out, 'T=%d N=%d K=%d' % (panel.T, panel.N, panel.K))


if __name__ == '__main__':
	main()
