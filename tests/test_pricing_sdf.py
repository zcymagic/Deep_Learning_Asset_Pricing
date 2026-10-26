import numpy as np
import torch

from glassbox.pricing import factor_sdf, moment_loss, sdf_from_weights
from glassbox.sdf import AdditiveSDF, LinearSDF


def _random_params(model, seed=0):
	g = torch.Generator().manual_seed(seed)
	with torch.no_grad():
		for p in model.parameters():
			p.copy_(0.1 * torch.randn(p.shape, generator=g))


def test_moment_loss_matches_manual(small_panel):
	panel, _ = small_panel
	b = panel.to_torch()
	w = torch.randn(panel.T, panel.N) * 0.01 * b['mask']
	M = sdf_from_weights(w, b['R'], b['mask'])
	h = torch.tanh(torch.randn(3, panel.T, panel.N))
	got = float(moment_loss(M, b['R'], b['mask'], h, weighted=True))
	R, mask, Mn, hn = panel.R, panel.mask, M.numpy(), h.numpy()
	T_i = np.maximum(mask.sum(0), 1)
	em = np.stack([(R * mask * Mn[:, None] * hn[c]).sum(0) / T_i for c in range(3)])
	want = np.mean(em ** 2 * T_i / T_i.max())
	assert np.isclose(got, want, rtol=1e-4)


def test_factor_sdf_prices_its_factors():
	rng = np.random.default_rng(0)
	F = 0.01 + 0.05 * rng.standard_normal((300, 3))
	M = factor_sdf(F)
	assert np.allclose((M[:, None] * F).mean(0), 0, atol=1e-12)


def test_design_sdf_and_features_are_exact(small_panel):
	panel, _ = small_panel
	b = panel.to_torch(['credit_spread', 'vix'])
	for model in [LinearSDF(panel.K, 2, True), AdditiveSDF(panel.K, 2, pairs=[(0, 4), (1, 2)], state_varying=True)]:
		_random_params(model)
		with torch.no_grad():
			w = model(b)
			assert torch.allclose(model.sdf(b), sdf_from_weights(w, b['R'], b['mask']), atol=1e-5)
			theta = model.theta()
			for t in range(panel.T):
				assert torch.allclose(model.features(b, t) @ theta, w[t][b['mask'][t]], atol=1e-5)
			model.set_theta(theta * 2)
			assert torch.allclose(model.theta(), theta * 2)


def test_shape_function_is_exact(small_panel):
	panel, _ = small_panel
	model = AdditiveSDF(panel.K, 0, state_varying=False)
	_random_params(model)
	b = panel.to_torch()
	with torch.no_grad():
		contrib, _, _ = model.components(b)
	t, i = np.argwhere(panel.mask)[5]
	k = 3
	z = panel.Z[t, i, k]
	_, v = model.shape_function(k, grid=[z])
	assert np.isclose(v[0], float(contrib[t, i, k]), atol=1e-6)
