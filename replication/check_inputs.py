"""Check that the input data for each replication stage is in place.

	python replication/check_inputs.py [stage ...]

Prints every required input with OK / MISSING and exits with status 1 if any requested stage lacks one.
Stages and their inputs are listed in INPUTS; see replication/README.md for where to obtain each file.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

CPZ = ['datasets/char/Char_%s.npz' % s for s in ('train', 'valid', 'test')] + \
	['datasets/macro/macro_%s.npz' % s for s in ('train', 'valid', 'test')]
CPZ_GAN = ['sample_checkpoints/sdf_normalized_ensemble.npy']
OSAP_PANEL = ['datasets/osap/Char_%s.npz' % s for s in ('train', 'valid', 'test', 'post')]
JKP_PANEL = ['datasets/jkp_panel/Char_%s.npz' % s for s in ('train', 'valid', 'test', 'post')]
JKP_THEMES = ['datasets/jkp_themes/Char_%s.npz' % s for s in ('train', 'valid', 'test', 'post')]


def osap_raw():
	from build_osap_panel import MAPPING
	signals = sorted({v for v in MAPPING.values() if not v.startswith(('crsp:', 'osap:'))})
	return ['crsp_download/crsp_msf_1961_2024.csv.gz', 'goyal_classic_download/Data2025.xlsx'] + \
		['osap_download/signals/%s.csv' % s for s in signals]


def jkp_archive():
	return [os.environ.get('JKP_ARCHIVE', 'data_raw/kelly.tar.bz2')]


INPUTS = {
	'tests': lambda: [],
	'synthetic': lambda: [],
	'cpz': lambda: CPZ + CPZ_GAN,
	'cpz_robustness': lambda: CPZ + CPZ_GAN + ['results/paper1_cpz_neural/model_additive.pt'],
	'osap_build': osap_raw,
	'osap': lambda: OSAP_PANEL,
	'osap_robustness': lambda: OSAP_PANEL + ['results/paper1_osap/model_additive.pt'],
	'jkp_build': lambda: jkp_archive() + ['datasets/osap/macro_%s.npz' % s for s in ('train', 'valid', 'test', 'post')],
	'jkp': lambda: JKP_PANEL + JKP_THEMES,
	'jkp_noclip': lambda: JKP_PANEL,
	'consistency': lambda: CPZ + OSAP_PANEL[:3] + ['datasets/jkp/jkp_sample.npz'],
	'compare': lambda: [],
}


def check(stages):
	ok = True
	for st in stages:
		files = INPUTS[st]()
		missing = [f for f in files if not os.path.exists(f)]
		print('%-16s %s' % (st, 'OK' if not missing else 'MISSING %d of %d input(s)' % (len(missing), len(files))))
		for f in missing:
			print('    missing: %s' % f)
		ok &= not missing
	return ok


if __name__ == '__main__':
	stages = sys.argv[1:] or list(INPUTS)
	unknown = [s for s in stages if s not in INPUTS]
	if unknown:
		sys.exit('unknown stage(s): %s; stages: %s' % (', '.join(unknown), ', '.join(INPUTS)))
	sys.exit(0 if check(stages) else 1)
