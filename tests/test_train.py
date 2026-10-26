import torch

from glassbox.adversary import NeuralAdversary, SoftRuleAdversary
from glassbox.sdf import AdditiveSDF, NeuralSDF
from glassbox.synthetic import split_panel
from glassbox.train import TrainConfig, evaluate, fit_gan


def test_gan_training_runs_and_improves(small_panel):
	panel, _ = small_panel
	tr, va, te = split_panel(panel, 40, 10)
	s = ['credit_spread']
	btr, bva = tr.to_torch(s), va.to_torch(s)
	cfg = TrainConfig(epochs_unc=60, epochs_moment=20, epochs_cond=60, lr=3e-3, eval_every=10)
	for model in [AdditiveSDF(tr.K, 1, n_knots=5), NeuralSDF(tr.K, len(tr.macro_names), hidden=(8,))]:
		torch.manual_seed(0)
		before = evaluate(model, btr, full=False)['loss_unc']
		res = fit_gan(model, NeuralAdversary(tr.K, len(tr.macro_names), 4), btr, bva, cfg)
		assert evaluate(res.model, btr, full=False)['loss_unc'] <= before
		assert len(res.history) > 0


def test_soft_rule_adversary_reads_out_rules(small_panel):
	panel, _ = small_panel
	b = panel.to_torch(['vix'])
	adv = SoftRuleAdversary(panel.char_names, ['vix'], n_rules=3, depth=2)
	h = adv(b)
	assert h.shape == (3, panel.T, panel.N) and float(h.detach().abs().max()) <= 1
	rules = adv.hard_rules()
	assert len(rules) == 3 and all(len(r.conds) == 2 for r in rules)
