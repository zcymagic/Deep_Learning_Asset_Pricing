"""SDF weight models, from fully transparent to black box.

Every model maps a batch (dict from Panel.to_torch) to portfolio weights w (T x N), so the SDF is
M_t = 1 - sum_i w_ti R_ti. The glass-box family is

	w_ti = c(s_t) + sum_k f_k(z_tik; s_t) + sum_{(j,k) in S} f_jk(z_tij, z_tik)
	f_k(z; s) = sum_m [theta_km + Gamma_km' s_t] B_m(z)        (piecewise-linear hat basis B_m)

so each characteristic contributes through a shape function that can be plotted exactly, the macro state
s_t (a few *named* variables) rescales those shapes linearly, and only a short list of pairwise
interactions is allowed. LinearSDF and NeuralSDF are the two ends of the complexity ladder.
"""
import numpy as np
import torch
import torch.nn as nn

from .pricing import sdf_from_weights

Z_LO, Z_HI = -0.5, 0.5


def _hat_coords(z, n_knots):
	"""Left knot index and interpolation weight for z on a uniform grid over [Z_LO, Z_HI]."""
	pos = (z.clamp(Z_LO, Z_HI) - Z_LO) / (Z_HI - Z_LO) * (n_knots - 1)
	j = pos.floor().clamp(max=n_knots - 2)
	return j.long(), pos - j


class LinearInParamsSDF(nn.Module):
	"""Base for SDFs whose weights are linear in the parameters: w_ti = phi(z_ti, s_t)' theta.

	Then M_t = 1 - sum_i R_ti w_ti = 1 - Psi_t' theta with the design Psi_t = sum_i R_ti phi(z_ti, s_t),
	a small T x D matrix computed once per panel (cached in the batch dict). Training never touches the
	T x N x K panel again, so a fit takes seconds even on the full CRSP cross-section. This is what makes
	the Rashomon exploration (hundreds of refits) feasible.
	"""

	def design(self, batch):
		raise NotImplementedError

	def theta(self):
		raise NotImplementedError

	def _cached_design(self, batch):
		cache = batch.setdefault('_design_cache', {})
		key = self.design_key()
		if key not in cache:
			with torch.no_grad():
				cache[key] = self.design(batch)
		return cache[key]

	def design_key(self):
		return (type(self).__name__,) + tuple(sorted((k, str(v)) for k, v in self.config.items()))

	def sdf(self, batch):
		return 1.0 - self._cached_design(batch) @ self.theta()


class LinearSDF(LinearInParamsSDF):
	"""w = c(s) + z'(beta + B s): characteristics enter linearly, prices may move with the state."""

	def __init__(self, K, P=0, state_interactions=True):
		super().__init__()
		self.K, self.P = K, P
		self.use_state = state_interactions and P > 0
		self.config = {'K': K, 'P': P, 'state': self.use_state}
		self.c0 = nn.Parameter(torch.zeros(1))
		self.beta = nn.Parameter(0.01 * torch.randn(K))
		if self.use_state:
			self.c1 = nn.Parameter(torch.zeros(P))
			self.B = nn.Parameter(torch.zeros(P, K))

	def forward(self, batch):
		Z, S, mask = batch['Z'], batch['S'], batch['mask']
		coef = self.beta[None].expand(Z.shape[0], -1)
		c = self.c0.expand(Z.shape[0])
		if self.use_state:
			coef = coef + S @ self.B
			c = c + S @ self.c1
		w = (Z * coef[:, None, :]).sum(-1) + c[:, None]
		return w * mask

	def design(self, batch):
		R, Z, S, mask = batch['R'], batch['Z'], batch['S'], batch['mask']
		Rm = R * mask
		base = torch.cat([Rm.sum(1, keepdim=True), torch.einsum('tn,tnk->tk', Rm, Z)], dim=1)	# T x (1+K)
		cols = [base[:, :1], base[:, 1:]]
		if self.use_state:
			cols += [S * base[:, :1], (S[:, :, None] * base[:, None, 1:]).reshape(S.shape[0], -1)]
		return torch.cat(cols, dim=1)

	def theta(self):
		parts = [self.c0, self.beta]
		if self.use_state:
			parts += [self.c1, self.B.reshape(-1)]
		return torch.cat(parts)

	def param_groups(self):
		"""Index sets of theta belonging to each characteristic (for importance / identification)."""
		K, P = self.K, self.P
		groups = {k: [1 + k] for k in range(K)}
		if self.use_state:
			off = 1 + K + P
			for p in range(P):
				for k in range(K):
					groups[k].append(off + p * K + k)
		return groups

	def set_theta(self, vec):
		vec = torch.as_tensor(vec, dtype=self.c0.dtype)
		K, P = self.K, self.P
		with torch.no_grad():
			self.c0.copy_(vec[:1])
			self.beta.copy_(vec[1:1 + K])
			if self.use_state:
				self.c1.copy_(vec[1 + K:1 + K + P])
				self.B.copy_(vec[1 + K + P:].view(P, K))
		return self

	def features(self, batch, t):
		"""Feature matrix phi (n_t x D) of the stocks observed in month t, so w_t = phi theta."""
		m = batch['mask'][t]
		z = batch['Z'][t][m]
		ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
		cols = [ones, z]
		if self.use_state:
			s = batch['S'][t]
			cols += [ones * s[None, :], (s[:, None, None] * z[None]).permute(1, 0, 2).reshape(z.shape[0], -1)]
		return torch.cat(cols, dim=1)

	def penalty(self):
		return torch.zeros(())


class AdditiveSDF(LinearInParamsSDF):
	"""Glass-box SDF: additive shape functions, state-varying via named macro variables, sparse pairs."""

	def __init__(self, K, P=0, n_knots=9, pairs=(), n_knots_pair=5, state_varying=True,
			lam_smooth=1e-3, lam_group=0.0, time_chunk=12):
		super().__init__()
		self.K, self.P, self.n_knots = K, P, n_knots
		self.pairs = [tuple(int(i) for i in p) for p in pairs]
		self.n_knots_pair = n_knots_pair
		self.state_varying = state_varying and P > 0
		self.lam_smooth, self.lam_group = lam_smooth, lam_group
		self.time_chunk = time_chunk
		self.config = {'K': K, 'P': P, 'nk': n_knots, 'pairs': self.pairs, 'nkp': n_knots_pair, 'state': self.state_varying}
		self.c0 = nn.Parameter(torch.zeros(1))
		self.theta_k = nn.Parameter(0.01 * torch.randn(K, n_knots))
		if self.state_varying:
			self.c1 = nn.Parameter(torch.zeros(P))
			self.Gamma = nn.Parameter(torch.zeros(P, K, n_knots))
		if self.pairs:
			self.U = nn.Parameter(torch.zeros(len(self.pairs), n_knots_pair, n_knots_pair))

	# --- linear-in-parameters representation -------------------------------------------------------
	def design(self, batch):
		R, Z, S, mask = batch['R'], batch['Z'], batch['S'], batch['mask']
		T, nk = R.shape[0], self.n_knots
		t_idx, i_idx = mask.nonzero(as_tuple=True)
		r = R[t_idx, i_idx]
		z = Z[t_idx, i_idx]									# n_obs x K
		Psi = torch.zeros(T * self.K * nk, dtype=R.dtype, device=R.device)
		for k in range(self.K):
			j, a = _hat_coords(z[:, k], nk)
			base = (t_idx * self.K + k) * nk
			Psi.index_add_(0, base + j, r * (1 - a))
			Psi.index_add_(0, base + j + 1, r * a)
		Psi = Psi.view(T, self.K, nk)
		Psi = Psi - Psi.mean(dim=-1, keepdim=True)			# matches centering of shape functions
		sumR = (R * mask).sum(1, keepdim=True)
		cols = [sumR, Psi.reshape(T, -1)]
		if self.state_varying:
			cols += [S * sumR, (S[:, :, None, None] * Psi[:, None]).reshape(T, -1)]
		if self.pairs:
			m = self.n_knots_pair
			for (u, v) in self.pairs:
				ju, au = _hat_coords(z[:, u], m)
				jv, av = _hat_coords(z[:, v], m)
				Pp = torch.zeros(T * m * m, dtype=R.dtype, device=R.device)
				for du, wu in ((0, 1 - au), (1, au)):
					for dv, wv in ((0, 1 - av), (1, av)):
						Pp.index_add_(0, (t_idx * m + ju + du) * m + jv + dv, r * wu * wv)
				Pp = Pp.view(T, m * m)
				cols.append(Pp - Pp.mean(dim=1, keepdim=True))
		return torch.cat(cols, dim=1)

	def theta(self):
		parts = [self.c0, self.theta_k.reshape(-1)]
		if self.state_varying:
			parts += [self.c1, self.Gamma.reshape(-1)]
		if self.pairs:
			parts.append(self.U.reshape(-1))
		return torch.cat(parts)

	def param_groups(self):
		"""Index sets of theta for each characteristic k and each pair ('pair', p)."""
		K, P, nk = self.K, self.P, self.n_knots
		groups = {k: list(range(1 + k * nk, 1 + (k + 1) * nk)) for k in range(K)}
		off = 1 + K * nk
		if self.state_varying:
			off += P
			for p in range(P):
				for k in range(K):
					start = off + (p * K + k) * nk
					groups[k] += list(range(start, start + nk))
			off += P * K * nk
		m2 = self.n_knots_pair ** 2
		for q in range(len(self.pairs)):
			groups[('pair', q)] = list(range(off + q * m2, off + (q + 1) * m2))
		return groups

	def features(self, batch, t):
		"""Feature matrix phi (n_t x D) of the stocks observed in month t, so w_t = phi theta.

		Hat bases sum to one, so the centered basis is B - 1/n_knots (matching the centered design).
		"""
		m = batch['mask'][t]
		z = batch['Z'][t][m]
		n, nk = z.shape[0], self.n_knots
		j, a = _hat_coords(z, nk)									# n x K
		B = torch.zeros(n, self.K, nk, dtype=z.dtype, device=z.device)
		B.scatter_add_(2, j[..., None], (1 - a)[..., None])
		B.scatter_add_(2, (j + 1)[..., None], a[..., None])
		B = (B - 1.0 / nk).reshape(n, -1)
		ones = torch.ones(n, 1, dtype=z.dtype, device=z.device)
		cols = [ones, B]
		if self.state_varying:
			s = batch['S'][t]
			cols += [ones * s[None, :], (s[None, :, None] * B[:, None, :]).reshape(n, -1)]
		mp = self.n_knots_pair
		for (u, v) in self.pairs:
			ju, au = _hat_coords(z[:, u], mp)
			jv, av = _hat_coords(z[:, v], mp)
			Bp = torch.zeros(n, mp * mp, dtype=z.dtype, device=z.device)
			for du, wu in ((0, 1 - au), (1, au)):
				for dv, wv in ((0, 1 - av), (1, av)):
					Bp.scatter_add_(1, ((ju + du) * mp + jv + dv)[:, None], (wu * wv)[:, None])
			cols.append(Bp - 1.0 / (mp * mp))
		return torch.cat(cols, dim=1)

	def set_theta(self, vec):
		"""Load a flat parameter vector (inverse of theta())."""
		vec = torch.as_tensor(vec, dtype=self.c0.dtype)
		K, P, nk = self.K, self.P, self.n_knots
		with torch.no_grad():
			i = 0
			self.c0.copy_(vec[i:i + 1]); i += 1
			self.theta_k.copy_(vec[i:i + K * nk].view(K, nk)); i += K * nk
			if self.state_varying:
				self.c1.copy_(vec[i:i + P]); i += P
				self.Gamma.copy_(vec[i:i + P * K * nk].view(P, K, nk)); i += P * K * nk
			if self.pairs:
				n = self.U.numel()
				self.U.copy_(vec[i:i + n].view_as(self.U)); i += n
		return self

	# --- exact components -------------------------------------------------------------------------
	def knot_values(self, S):
		"""T x K x n_knots knot values of every shape function in each month (centered per function)."""
		V = self.theta_k[None].expand(S.shape[0], -1, -1)
		if self.state_varying:
			V = V + torch.einsum('tp,pkm->tkm', S, self.Gamma)
		return V - V.mean(dim=-1, keepdim=True)

	def pair_values(self):
		U = self.U
		return U - U.mean(dim=(1, 2), keepdim=True)

	def components(self, batch):
		"""Per-characteristic contributions T x N x K, pair contributions T x N x len(pairs), intercept T."""
		Z, S = batch['Z'], batch['S']
		V = self.knot_values(S)
		j, a = _hat_coords(Z, self.n_knots)						# T x N x K
		Vt = V[:, None].expand(-1, Z.shape[1], -1, -1)			# T x N x K x n_knots (view)
		left = torch.gather(Vt, 3, j[..., None]).squeeze(-1)
		right = torch.gather(Vt, 3, (j + 1)[..., None]).squeeze(-1)
		contrib = left * (1 - a) + right * a
		pair = None
		if self.pairs:
			Up = self.pair_values()
			cols = []
			for p, (u, v) in enumerate(self.pairs):
				ju, au = _hat_coords(Z[..., u], self.n_knots_pair)
				jv, av = _hat_coords(Z[..., v], self.n_knots_pair)
				G = Up[p]
				cols.append(G[ju, jv] * (1 - au) * (1 - av) + G[ju + 1, jv] * au * (1 - av)
					+ G[ju, jv + 1] * (1 - au) * av + G[ju + 1, jv + 1] * au * av)
			pair = torch.stack(cols, dim=-1)
		c = self.c0.expand(Z.shape[0])
		if self.state_varying:
			c = c + S @ self.c1
		return contrib, pair, c

	def _forward_chunk(self, batch):
		contrib, pair, c = self.components(batch)
		w = contrib.sum(-1) + c[:, None]
		if pair is not None:
			w = w + pair.sum(-1)
		return w * batch['mask']

	def forward(self, batch):
		"""Exact stock-level weights (used for evaluation and interpretation; training uses sdf())."""
		T = batch['Z'].shape[0]
		if not self.time_chunk or T <= self.time_chunk:
			return self._forward_chunk(batch)
		out = []
		for s in range(0, T, self.time_chunk):
			sub = {k: batch[k][s:s + self.time_chunk] for k in ('Z', 'S', 'mask')}
			out.append(self._forward_chunk(sub))
		return torch.cat(out, dim=0)

	def penalty(self):
		th = self.theta_k
		pen = self.lam_smooth * (th[:, 2:] - 2 * th[:, 1:-1] + th[:, :-2]).pow(2).sum()
		if self.state_varying:
			G = self.Gamma
			pen = pen + self.lam_smooth * (G[..., 2:] - 2 * G[..., 1:-1] + G[..., :-2]).pow(2).sum()
		if self.lam_group > 0:
			thc = th - th.mean(-1, keepdim=True)
			groups = thc.pow(2).sum(-1)
			if self.state_varying:
				Gc = self.Gamma - self.Gamma.mean(-1, keepdim=True)
				groups = groups + Gc.pow(2).sum(dim=(0, 2))
			pen = pen + self.lam_group * groups.clamp(min=1e-12).sqrt().sum()
			if self.pairs:
				pen = pen + self.lam_group * self.pair_values().pow(2).sum(dim=(1, 2)).clamp(min=1e-12).sqrt().sum()
		return pen

	# --- interpretation ----------------------------------------------------------------------------
	@torch.no_grad()
	def shape_function(self, k, s=None, grid=None):
		"""Exact shape function of characteristic k at state s (P vector; default all zeros)."""
		grid = torch.linspace(Z_LO, Z_HI, 101) if grid is None else torch.as_tensor(grid, dtype=torch.float32)
		S = torch.zeros(1, self.P) if s is None else torch.as_tensor(s, dtype=torch.float32).reshape(1, -1)
		V = self.knot_values(S)[0, k]
		j, a = _hat_coords(grid, self.n_knots)
		return grid.numpy(), (V[j] * (1 - a) + V[j + 1] * a).numpy()

	@torch.no_grad()
	def pair_surface(self, p, n=41):
		g = torch.linspace(Z_LO, Z_HI, n)
		gu, gv = torch.meshgrid(g, g, indexing='ij')
		ju, au = _hat_coords(gu, self.n_knots_pair)
		jv, av = _hat_coords(gv, self.n_knots_pair)
		G = self.pair_values()[p]
		val = G[ju, jv] * (1 - au) * (1 - av) + G[ju + 1, jv] * au * (1 - av) \
			+ G[ju, jv + 1] * (1 - au) * av + G[ju + 1, jv + 1] * au * av
		return g.numpy(), val.numpy()


class NeuralSDF(nn.Module):
	"""Black-box benchmark: the CPZ feed-forward weight network on [z, macro] (no LSTM)."""

	def __init__(self, K, M, hidden=(64, 64), dropout=0.05, use_macro=True):
		super().__init__()
		self.use_macro = use_macro and M > 0
		d = K + (M if self.use_macro else 0)
		layers = []
		for h in hidden:
			layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
			d = h
		layers.append(nn.Linear(d, 1))
		self.net = nn.Sequential(*layers)

	def forward(self, batch):
		Z, mask = batch['Z'], batch['mask']
		x = Z
		if self.use_macro:
			x = torch.cat([Z, batch['macro'][:, None, :].expand(-1, Z.shape[1], -1)], dim=-1)
		return self.net(x).squeeze(-1) * mask

	def sdf(self, batch):
		return sdf_from_weights(self(batch), batch['R'], batch['mask'])

	def penalty(self):
		return torch.zeros(())


class FixedWeightsSDF(nn.Module):
	"""Wrap externally estimated weights (e.g. the CPZ TensorFlow ensemble) so they can be audited."""

	def __init__(self, w):
		super().__init__()
		self.register_buffer('w', torch.as_tensor(np.asarray(w), dtype=torch.float32))

	def forward(self, batch):
		return self.w * batch['mask']

	def sdf(self, batch):
		return sdf_from_weights(self(batch), batch['R'], batch['mask'])

	def penalty(self):
		return torch.zeros(())


def build_sdf(spec, K, P, M_dim):
	"""Build a model from a ladder spec dict, e.g. {'type': 'additive', 'state_varying': true}."""
	kind = spec['type']
	if kind == 'linear':
		return LinearSDF(K, P, state_interactions=spec.get('state_varying', False))
	if kind == 'additive':
		return AdditiveSDF(K, P, n_knots=spec.get('n_knots', 9), pairs=spec.get('pairs', ()),
			n_knots_pair=spec.get('n_knots_pair', 5), state_varying=spec.get('state_varying', False),
			lam_smooth=spec.get('lam_smooth', 1e-3), lam_group=spec.get('lam_group', 0.0),
			time_chunk=spec.get('time_chunk', 12))
	if kind == 'neural':
		return NeuralSDF(K, M_dim, hidden=tuple(spec.get('hidden', (64, 64))), dropout=spec.get('dropout', 0.05),
			use_macro=spec.get('use_macro', True))
	raise ValueError('unknown SDF type %s' % kind)
