import numpy as np
import torch

from glassbox.adversary import NeuralAdversary
from glassbox.pricing import moment_loss
from glassbox.rashomon import RashomonEllipsoid, _trs_max, select_shrinkage
from glassbox.sdf import AdditiveSDF, LinearSDF
from glassbox.synthetic import SyntheticSpec, make_synthetic_panel


def test_trust_region_matches_brute_force():
	rng = np.random.default_rng(0)
	for _ in range(5):
		Q = rng.standard_normal((4, 4))
		Mq = Q + Q.T
		c = rng.standard_normal(4)
		val, u = _trs_max(Mq, c)
		U = rng.standard_normal((200000, 4))
		U /= np.linalg.norm(U, axis=1, keepdims=True)
		U *= rng.random((200000, 1)) ** 0.25
		brute = np.max(np.einsum('nd,de,ne->n', U, Mq, U) + 2 * U @ c)
		assert val >= brute - 1e-9 and val <= brute + 0.05 * abs(brute) + 1e-6
		assert np.linalg.norm(u) <= 1 + 1e-8


def _ellipsoid(model, panel, eps=0.01, n_boot=0, states=('credit_spread',)):
	b = panel.to_torch(list(states))
	torch.manual_seed(0)
	adv = NeuralAdversary(panel.K, len(panel.macro_names), 4)
	with torch.no_grad():
		h = adv(b)
	return RashomonEllipsoid(model, b, h, eps=eps, n_boot=n_boot), b, h


def test_closed_form_objective_and_boundary(small_panel):
	panel, _ = small_panel
	model = AdditiveSDF(panel.K, 1, n_knots=5, state_varying=True)
	E, b, h = _ellipsoid(model, panel)
	with torch.no_grad():
		M = model.sdf(b)
		J = moment_loss(M, b['R'], b['mask']) + moment_loss(M, b['R'], b['mask'], h) + model.penalty()
	assert np.isclose(float(J), E.J_star, rtol=1e-4)
	for th in E.sample(5):
		assert np.isclose(E.objective(th), (1 + E.eps) * E.J_star, rtol=1e-6)


def test_linear_range_bounds_samples(small_panel):
	panel, _ = small_panel
	E, _, _ = _ellipsoid(LinearSDF(panel.K, 1, True), panel)
	g = np.random.default_rng(1).standard_normal(E.theta.size)
	mid, lo, hi = E.linear_range(g)
	vals = E.sample(2000) @ g
	assert vals.min() >= lo - 1e-9 and vals.max() <= hi + 1e-9
	assert vals.max() > mid


def test_collinear_characteristics_identified_only_jointly():
	panel, _ = make_synthetic_panel(SyntheticSpec(seed=0, corr_value=0.99, mispricing=0.0, missing_factor=0.0))
	E, _, _ = _ellipsoid(LinearSDF(panel.K, 0, False), panel, eps=0.01, states=())
	tab = E.importance_table(panel.char_names, groups={'value': ['BEME', 'CF2P']}).set_index('feature')
	assert tab.loc['group:value', 'min'] > tab.loc['BEME', 'min']
	assert tab.loc['group:value', 'min'] > tab.loc['CF2P', 'min']


def test_bootstrap_tables_and_ambiguity(small_panel):
	panel, _ = small_panel
	E, b, _ = _ellipsoid(AdditiveSDF(panel.K, 1, n_knots=5, state_varying=True), panel, n_boot=20)
	tab = E.importance_table(panel.char_names, method='bootstrap')
	assert len(tab) == panel.K and (tab['min'] <= tab['max']).all()
	A, sign_amb, index = E.weight_ambiguity(b)
	assert np.isfinite(A[panel.mask]).all() and np.isnan(A[~panel.mask]).all()
	assert index.shape == (panel.T,)


def test_shrinkage_selection_and_penalized_objective(small_panel):
	panel, _ = small_panel
	tr, va = panel.subset_time(range(40)), panel.subset_time(range(40, 60))
	model = AdditiveSDF(panel.K, 1, n_knots=5, state_varying=True)
	b_tr, b_va = tr.to_torch(['credit_spread']), va.to_torch(['credit_spread'])
	torch.manual_seed(0)
	with torch.no_grad():
		h = NeuralAdversary(panel.K, len(panel.macro_names), 4)(b_tr)
	best, df = select_shrinkage(model, b_tr, h, b_va, grid=(0.0, 0.1, 10.0))
	assert best in (0.0, 0.1, 10.0) and len(df) == 3
	assert df.loc[df['shrink'] == best, 'sharpe_valid'].item() == df['sharpe_valid'].max()
	E0 = RashomonEllipsoid(model, b_tr, h, eps=0.01, n_boot=0, shrink=0.0)
	n0 = np.linalg.norm(E0.theta)
	E1 = RashomonEllipsoid(model, b_tr, h, eps=0.01, n_boot=0, shrink=10.0)
	assert np.linalg.norm(E1.theta) < n0
	for th in E1.sample(3):
		assert np.isclose(E1.objective(th), 1.01 * E1.J_star, rtol=1e-6)
