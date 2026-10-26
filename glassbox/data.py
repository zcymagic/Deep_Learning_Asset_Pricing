"""Panel data container and loaders.

Reads the same ``.npz`` files as ``src/data/data_layer.py``:
	char file : data[T, N, 1 + K] with column 0 = excess return, UNK = -99.99 marks missing
	macro file: data[T, M] with variable names
so the glass-box code runs unchanged on the Chen-Pelger-Zhu data.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import torch

UNK = -99.99


@dataclass
class Panel:
	R: np.ndarray			# T x N excess returns (0 where missing)
	Z: np.ndarray			# T x N x K characteristics, rank-normalized to [-0.5, 0.5]
	mask: np.ndarray		# T x N bool, True = observed
	macro: np.ndarray		# T x M macro series (normalized)
	char_names: list
	macro_names: list
	dates: np.ndarray = None
	extra: dict = field(default_factory=dict)	# optional T x N arrays (e.g. institutional ownership)

	@property
	def T(self):
		return self.R.shape[0]

	@property
	def N(self):
		return self.R.shape[1]

	@property
	def K(self):
		return self.Z.shape[2]

	def char_idx(self, name):
		return self.char_names.index(name)

	def macro_idx(self, name):
		return self.macro_names.index(name)

	def state(self, state_vars):
		"""T x P matrix of the named macro state variables."""
		if not state_vars:
			return np.zeros((self.T, 0), dtype=np.float32)
		return self.macro[:, [self.macro_idx(v) for v in state_vars]].astype(np.float32)

	def subset_time(self, idx):
		idx = np.asarray(idx)
		return Panel(R=self.R[idx], Z=self.Z[idx], mask=self.mask[idx], macro=self.macro[idx],
			char_names=self.char_names, macro_names=self.macro_names,
			dates=None if self.dates is None else self.dates[idx],
			extra={k: v[idx] for k, v in self.extra.items()})

	def to_torch(self, state_vars=(), device='cpu'):
		return {
			'R': torch.as_tensor(self.R, dtype=torch.float32, device=device),
			'Z': torch.as_tensor(self.Z, dtype=torch.float32, device=device),
			'mask': torch.as_tensor(self.mask, dtype=torch.bool, device=device),
			'macro': torch.as_tensor(self.macro, dtype=torch.float32, device=device),
			'S': torch.as_tensor(self.state(list(state_vars)), dtype=torch.float32, device=device),
		}


def _select_macro(names, macro_idx):
	if macro_idx is None or macro_idx == 'all':
		return np.arange(len(names))
	if macro_idx == '178':
		return np.sort(np.concatenate((np.arange(124), np.arange(284, 338))))
	return np.sort(np.array(macro_idx, dtype=int))


def load_npz_panel(char_path, macro_path=None, macro_idx=None, macro_mean=None, macro_std=None):
	"""Load one split. Returns (panel, macro_mean, macro_std); pass the train moments to valid/test."""
	tmp = np.load(char_path, allow_pickle=True)
	data = tmp['data']
	R = data[:, :, 0].astype(np.float32)
	Z = data[:, :, 1:].astype(np.float32)
	mask = R != UNK
	R = np.where(mask, R, 0.0).astype(np.float32)
	Z = np.where(mask[:, :, None], Z, 0.0).astype(np.float32)
	char_names = [str(v) for v in tmp['variable'][1:]]
	dates = tmp['date'] if 'date' in tmp.files else None

	if macro_path is None:
		macro = np.zeros((R.shape[0], 0), dtype=np.float32)
		macro_names = []
	else:
		tmpm = np.load(macro_path, allow_pickle=True)
		idx = _select_macro(tmpm['variable'], macro_idx)
		macro = tmpm['data'][:, idx].astype(np.float64)
		macro_names = [str(v) for v in tmpm['variable'][idx]]
		if macro_mean is None or macro_std is None:
			macro_mean = macro.mean(axis=0)
			macro_std = macro.std(axis=0)
		macro = ((macro - macro_mean) / np.where(macro_std > 0, macro_std, 1.0)).astype(np.float32)
	panel = Panel(R=R, Z=Z, mask=mask, macro=macro, char_names=char_names, macro_names=macro_names, dates=dates)
	return panel, macro_mean, macro_std


def load_splits(config):
	"""Load train / valid / test panels from a config using the repo's key names."""
	macro_idx = config.get('macro_idx', None)
	train, mu, sd = load_npz_panel(config['individual_feature_file'], config.get('macro_feature_file'), macro_idx)
	valid, _, _ = load_npz_panel(config['individual_feature_file_valid'], config.get('macro_feature_file_valid'), macro_idx, mu, sd)
	test, _, _ = load_npz_panel(config['individual_feature_file_test'], config.get('macro_feature_file_test'), macro_idx, mu, sd)
	return train, valid, test


def load_config(path):
	with open(path) as f:
		return json.load(f)


def save_npz_panel(panel, char_path, macro_path):
	"""Write a panel in the repo's npz format (inverse of load_npz_panel)."""
	data = np.concatenate([panel.R[:, :, None], panel.Z], axis=2)
	data = np.where(panel.mask[:, :, None], data, UNK).astype(np.float32)
	dates = panel.dates if panel.dates is not None else np.arange(panel.T)
	np.savez(char_path, data=data, date=dates, variable=np.array(['RET'] + list(panel.char_names)))
	np.savez(macro_path, data=panel.macro, date=dates, variable=np.array(panel.macro_names))


def concat_time(panels):
	"""Stack panels over time (e.g. valid + test as one hold-out sample).

	Splits may hold different stock universes; stocks are not matched across splits, which is fine for
	everything defined month by month (rule portfolios, SDF portfolios, sorts). Columns are padded.
	"""
	N = max(p.N for p in panels)

	def pad(x, fill):
		if x.shape[1] == N:
			return x
		shape = (x.shape[0], N - x.shape[1]) + x.shape[2:]
		return np.concatenate([x, np.full(shape, fill, dtype=x.dtype)], axis=1)

	first = panels[0]
	dates = None if any(p.dates is None for p in panels) else np.concatenate([p.dates for p in panels])
	extra_keys = set.intersection(*[set(p.extra) for p in panels]) if panels else set()
	return Panel(
		R=np.concatenate([pad(p.R, 0.0) for p in panels]),
		Z=np.concatenate([pad(p.Z, 0.0) for p in panels]),
		mask=np.concatenate([pad(p.mask, False) for p in panels]),
		macro=np.concatenate([p.macro for p in panels]),
		char_names=first.char_names, macro_names=first.macro_names, dates=dates,
		extra={k: np.concatenate([pad(p.extra[k], np.nan) for p in panels]) for k in extra_keys})
