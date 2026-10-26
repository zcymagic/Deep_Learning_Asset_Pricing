"""Exact read-outs and figures for glass-box SDFs (no post-hoc approximation)."""
import numpy as np
import pandas as pd
import torch

from .sdf import AdditiveSDF, LinearSDF


def shape_table(model, char_names, state_names=(), state_levels=(-1.0, 0.0, 1.0), grid=None):
	"""Long table of every shape function at each level of each state variable (others at 0)."""
	grid = np.linspace(-0.5, 0.5, 21) if grid is None else grid
	rows = []
	P = len(state_names)
	settings = [('none', 0.0, np.zeros(P))]
	for p, name in enumerate(state_names):
		for lv in state_levels:
			s = np.zeros(P)
			s[p] = lv
			settings.append((name, lv, s))
	for k, cname in enumerate(char_names):
		for sname, lv, s in settings:
			if isinstance(model, AdditiveSDF):
				g, v = model.shape_function(k, s if P else None, grid)
			else:
				coef = model.beta[k].item() + (float(s @ model.B[:, k].detach().numpy()) if getattr(model, 'use_state', False) else 0.0)
				g, v = grid, coef * grid
			for z, val in zip(g, v):
				rows.append({'char': cname, 'state': sname, 'level': lv, 'z': float(z), 'f': float(val)})
	return pd.DataFrame(rows)


def contribution_importance(model, batch, char_names):
	"""Share of the cross-sectional variance of SDF weights due to each characteristic (exact, in-sample)."""
	with torch.no_grad():
		if isinstance(model, AdditiveSDF):
			contrib, pair, _ = model.components(batch)
		else:
			Z, S = batch['Z'], batch['S']
			coef = model.beta[None].expand(Z.shape[0], -1)
			if model.use_state:
				coef = coef + S @ model.B
			contrib, pair = Z * coef[:, None, :], None
		m = batch['mask']
		v = [contrib[..., k][m].var().item() for k in range(contrib.shape[-1])]
		names = list(char_names)
		if pair is not None:
			v += [pair[..., q][m].var().item() for q in range(pair.shape[-1])]
			names += ['%s x %s' % (char_names[u], char_names[w]) for u, w in model.pairs]
	v = np.array(v)
	return pd.DataFrame({'feature': names, 'variance_share': v / v.sum()}).sort_values('variance_share', ascending=False)


def plot_shapes(model, char_names, state_name=None, state_index=0, P=0, path=None, ncols=5, bands=None):
	"""Small multiples of the shape functions; with a state variable, one line per state level.

	bands: optional RashomonEllipsoid; draws its exact band at the neutral state.
	"""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	K = len(char_names)
	nrows = int(np.ceil(K / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), sharex=True, squeeze=False)
	levels = [(-1.0, '#3b6fb6', 'low'), (0.0, '#444444', 'mid'), (1.0, '#c8553d', 'high')] if state_name else [(0.0, '#444444', None)]
	for k in range(K):
		ax = axes[k // ncols][k % ncols]
		if bands is not None:
			g, ref, lo, hi = bands.shape_band(k)
			ax.fill_between(g, lo, hi, color='#bbbbbb', alpha=0.5, lw=0)
		for lv, col, lab in levels:
			s = None
			if P:
				s = np.zeros(P)
				s[state_index] = lv
			if isinstance(model, AdditiveSDF):
				g, v = model.shape_function(k, s)
			else:
				g = np.linspace(-0.5, 0.5, 101)
				coef = model.beta[k].item() + (float(np.asarray(s) @ model.B[:, k].detach().numpy()) if s is not None and model.use_state else 0.0)
				v = coef * g
			ax.plot(g, v, color=col, lw=1.6, label=None if lab is None else '%s %s' % (state_name, lab))
		ax.axhline(0, color='#999999', lw=0.6)
		ax.set_title(char_names[k], fontsize=9)
		ax.tick_params(labelsize=7)
	for k in range(K, nrows * ncols):
		axes[k // ncols][k % ncols].axis('off')
	if state_name:
		axes[0][0].legend(fontsize=7, frameon=False)
	fig.tight_layout()
	if path:
		fig.savefig(path, dpi=150)
	plt.close(fig)
	return fig


def plot_pair(model, q, char_names, path=None):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	g, val = model.pair_surface(q)
	u, v = model.pairs[q]
	fig, ax = plt.subplots(figsize=(3.6, 3.0))
	lim = np.abs(val).max()
	im = ax.imshow(val.T, origin='lower', extent=[g[0], g[-1], g[0], g[-1]], cmap='RdBu_r', vmin=-lim, vmax=lim)
	ax.set_xlabel(char_names[u])
	ax.set_ylabel(char_names[v])
	fig.colorbar(im, ax=ax, shrink=0.8)
	fig.tight_layout()
	if path:
		fig.savefig(path, dpi=150)
	plt.close(fig)
	return fig
