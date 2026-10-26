import numpy as np

from glassbox import audit


def _halves(panel, truth):
	T = panel.T // 2
	A, B = panel.subset_time(np.arange(T)), panel.subset_time(np.arange(T, panel.T))
	return A, B, truth['M_oracle'][:T], truth['M_oracle'][T:]


def test_alpha_moments_mean_is_ols_alpha():
	rng = np.random.default_rng(1)
	T = 200
	M = 1 - 0.3 * rng.standard_normal(T)
	ctrl = rng.standard_normal((T, 1))
	X = 0.002 + 0.5 * (1 - M)[:, None] + 0.3 * ctrl + 0.01 * rng.standard_normal((T, 2))
	G = audit.alpha_moments(X, M, ctrl)
	Fm = np.column_stack([np.ones(T), 1 - M, ctrl])
	coef, *_ = np.linalg.lstsq(Fm, X, rcond=None)
	assert np.allclose(G.mean(0), coef[0])


def test_auditor_finds_and_classifies_planted_mispricing(full_panel):
	panel, truth = full_panel
	A, B, MA, MB = _halves(panel, truth)
	disc = audit.discover(A, MA, panel.macro_names, top=5)
	conf = audit.confirm(B, MB, disc, n_boot=200, state_vars=panel.macro_names)
	table, _ = audit.failure_map(conf, lta_chars=('LME', 'Spread', 'IdioVol', 'LTurnover'))
	top = table.iloc[0]
	feats = {c.feature for c in top['rule'].char_conds}
	assert 'Spread' in feats and feats <= {'LME', 'Spread'}
	assert top['class'] == 'limits-to-arbitrage'
	assert top['p_maxt'] < 0.01 and top['t_holdout'] > 5


def test_auditor_finds_momentum_failure_in_full_sample(full_panel):
	# the planted momentum premium (high-VIX months only) is near the multiple-testing bar in a
	# half sample (t ~ 4-6 vs ~4.7), so discovery power is checked on the full sample
	panel, truth = full_panel
	disc = audit.discover(panel, truth['M_oracle'], panel.macro_names, top=5)
	assert any(any(c.feature == 'r12_2' for c in r.conds) for r in disc['rule'])


def test_state_dependence_test_picks_vix(full_panel):
	panel, truth = full_panel
	_, B, _, MB = _halves(panel, truth)
	rule = audit.Rule((audit.Condition('r12_2', 'char', '>', 0.3),))
	var, t = audit.state_dependence(B, MB, [rule], panel.macro_names)
	assert var[0] == 'vix' and t[0] > 3
	lta = audit.Rule((audit.Condition('LME', 'char', '<', -0.3), audit.Condition('Spread', 'char', '>', 0.3)))
	_, t_lta = audit.state_dependence(B, MB, [lta], panel.macro_names)
	assert abs(t_lta[0]) < 3


def test_global_test_has_size_under_the_null(null_panel):
	panel, truth = null_panel
	rules = audit.fixed_rule_class(panel, panel.macro_names)
	res = audit.global_test(panel, truth['M_oracle'], rules, n_boot=300)
	assert res['p_value'] > 0.05


def test_null_discovery_confirms_nothing(null_panel):
	panel, truth = null_panel
	A, B, MA, MB = _halves(panel, truth)
	disc = audit.discover(A, MA, panel.macro_names, top=5)
	if len(disc):
		conf = audit.confirm(B, MB, disc, n_boot=200)
		assert (conf['p_maxt'] > 0.05).all()
