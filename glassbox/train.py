"""Generic CPZ-style adversarial training for any SDF model and any adversary.

Phases (as in FeedForwardModelWithNA_GAN.train):
	1. unconditional: minimize the unconditional moment loss
	2. adversary: maximize the conditional loss over instruments h with the SDF frozen
	3. conditional: minimize the conditional loss with h frozen
Phases 2-3 can be repeated (n_rounds). Model selection uses validation Sharpe ratio of the SDF factor.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .pricing import moment_loss, normalized_weights, sharpe, explained_variation


@dataclass
class TrainConfig:
	epochs_unc: int = 256
	epochs_moment: int = 64
	epochs_cond: int = 512
	n_rounds: int = 1
	lr: float = 1e-3
	lr_adv: float = 1e-3
	weighted: bool = True
	select_on: str = 'sharpe'		# 'sharpe' or 'loss'
	eval_every: int = 8
	seed: int = 0
	verbose: bool = False
	anneal_every: int = 8			# for SoftRuleAdversary


@dataclass
class FitResult:
	model: object
	adversary: object
	history: list = field(default_factory=list)
	best: dict = field(default_factory=dict)


def evaluate(model, batch, h=None, weighted=True, full=True):
	"""Loss and Sharpe ratio of an SDF model on one panel.

	sharpe_raw uses F = 1 - M (as CPZ do for model selection; cheap). With full=True also the Sharpe ratio
	of the normalized SDF portfolio (sum |w_t| = 1) and EV / XS-R2 with w as loadings.
	"""
	model.eval()
	with torch.no_grad():
		M = model.sdf(batch)
		loss_unc = float(moment_loss(M, batch['R'], batch['mask'], None, weighted))
		loss_cond = float(moment_loss(M, batch['R'], batch['mask'], h, weighted)) if h is not None else float('nan')
		F_raw = (1.0 - M).cpu().numpy()
		out = {'loss_unc': loss_unc, 'loss_cond': loss_cond, 'sharpe_raw': sharpe(F_raw), 'F_raw': F_raw}
		if full:
			w = model(batch)
			wn = normalized_weights(w, batch['mask'])
			F = (wn * batch['R']).sum(dim=1).cpu().numpy()
			R, mask = batch['R'].cpu().numpy(), batch['mask'].cpu().numpy()
			ev, xs = explained_variation(w.cpu().numpy(), R, mask)
			out.update({'sharpe': sharpe(F), 'F': F, 'ev': ev, 'xs_r2': xs})
	return out


def _phase_sdf(model, batch_tr, batch_va, h_tr, h_va, epochs, cfg, state, tag):
	opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	for ep in range(epochs):
		model.train()
		opt.zero_grad()
		M = model.sdf(batch_tr)
		loss = moment_loss(M, batch_tr['R'], batch_tr['mask'], h_tr, cfg.weighted) + model.penalty()
		loss.backward()
		opt.step()
		if (ep + 1) % cfg.eval_every == 0 or ep == epochs - 1:
			ev = evaluate(model, batch_va, h_va, cfg.weighted, full=False)
			score = ev['sharpe_raw'] if cfg.select_on == 'sharpe' else -(ev['loss_cond'] if h_va is not None else ev['loss_unc'])
			state['history'].append({'phase': tag, 'epoch': ep, 'train_loss': loss.item(), 'valid_sharpe': ev['sharpe_raw'],
				'valid_loss_unc': ev['loss_unc']})
			if score > state['best_score']:
				state['best_score'] = score
				state['best_state'] = copy.deepcopy(model.state_dict())
				state['best_phase'] = tag
			if cfg.verbose:
				print('[%s] ep %d loss %.5f valid SR %.3f' % (tag, ep, loss.item(), ev['sharpe_raw']))


def _phase_adversary(model, adversary, batch_tr, epochs, cfg):
	model.eval()
	with torch.no_grad():
		M = model.sdf(batch_tr)
	opt = torch.optim.Adam(adversary.parameters(), lr=cfg.lr_adv)
	for ep in range(epochs):
		opt.zero_grad()
		loss = -moment_loss(M, batch_tr['R'], batch_tr['mask'], adversary(batch_tr), cfg.weighted)
		loss.backward()
		opt.step()
		if hasattr(adversary, 'anneal') and (ep + 1) % cfg.anneal_every == 0:
			adversary.anneal()


def fit_gan(model, adversary, batch_tr, batch_va, cfg=TrainConfig()):
	"""Train an SDF against an adversary; returns the best model (by validation) and the adversary."""
	torch.manual_seed(cfg.seed)
	state = {'history': [], 'best_score': -np.inf, 'best_state': copy.deepcopy(model.state_dict()), 'best_phase': None}
	_phase_sdf(model, batch_tr, batch_va, None, None, cfg.epochs_unc, cfg, state, 'unc')
	if adversary is not None:
		for r in range(cfg.n_rounds):
			model.load_state_dict(state['best_state'])
			_phase_adversary(model, adversary, batch_tr, cfg.epochs_moment, cfg)
			adversary.eval()
			with torch.no_grad():
				h_tr = adversary(batch_tr)
				h_va = adversary(batch_va)
			_phase_sdf(model, batch_tr, batch_va, h_tr, h_va, cfg.epochs_cond, cfg, state, 'cond%d' % r)
	model.load_state_dict(state['best_state'])
	return FitResult(model=model, adversary=adversary, history=state['history'],
		best={'score': state['best_score'], 'phase': state['best_phase']})


def fit_moment_loss(model, batch_tr, h_tr, epochs, lr=1e-3, weighted=True, extra_objective=None):
	"""Plain minimization of the (conditional) moment loss with a fixed instrument set.

	extra_objective(model) -> scalar is added to the loss; used by the Rashomon exploration.
	"""
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	for _ in range(epochs):
		model.train()
		opt.zero_grad()
		M = model.sdf(batch_tr)
		loss = moment_loss(M, batch_tr['R'], batch_tr['mask'], None, weighted) + model.penalty()
		if h_tr is not None:
			loss = loss + moment_loss(M, batch_tr['R'], batch_tr['mask'], h_tr, weighted)
		if extra_objective is not None:
			loss = loss + extra_objective(model)
		loss.backward()
		opt.step()
	return model
