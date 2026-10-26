"""Adversaries for the conditional moment game min_SDF max_h sum_i E[M R_i h_i]^2.

NeuralAdversary is the CPZ moment network (strong, opaque auditor used to *train* SDFs).
SoftRuleAdversary is a differentiable glass-box auditor: each of its C instruments is a conjunction of
D soft threshold conditions on one characteristic or one macro state variable. Annealing the
temperatures makes every instrument a readable rule such as
	"LME < -0.27 AND Spread > 0.21"  or  "r12_2 > 0.30 AND vix > 0.95".
"""
import torch
import torch.nn as nn


class NeuralAdversary(nn.Module):
	def __init__(self, K, M, n_moments=8, hidden=(), use_macro=True):
		super().__init__()
		self.use_macro = use_macro and M > 0
		d = K + (M if self.use_macro else 0)
		layers = []
		for h in hidden:
			layers += [nn.Linear(d, h), nn.ReLU()]
			d = h
		layers += [nn.Linear(d, n_moments), nn.Tanh()]
		self.net = nn.Sequential(*layers)

	def forward(self, batch):
		Z = batch['Z']
		x = Z
		if self.use_macro:
			x = torch.cat([Z, batch['macro'][:, None, :].expand(-1, Z.shape[1], -1)], dim=-1)
		return self.net(x).permute(2, 0, 1)	# C x T x N


class SoftRuleAdversary(nn.Module):
	"""C soft conjunctions of D conditions over K characteristics and P named state variables."""

	def __init__(self, char_names, state_names, n_rules=8, depth=2, temp=0.1, select_temp=1.0):
		super().__init__()
		self.char_names, self.state_names = list(char_names), list(state_names)
		self.F = len(self.char_names) + len(self.state_names)
		self.C, self.D = n_rules, depth
		self.select_logits = nn.Parameter(0.01 * torch.randn(n_rules, depth, self.F))
		self.threshold = nn.Parameter(0.2 * torch.randn(n_rules, depth))
		self.direction = nn.Parameter(torch.randn(n_rules, depth))
		self.sign = nn.Parameter(torch.randn(n_rules))
		self.temp, self.select_temp = temp, select_temp

	def _features(self, batch):
		Z, S = batch['Z'], batch['S']
		feats = [Z]
		if S.shape[1] > 0:
			feats.append(S[:, None, :].expand(-1, Z.shape[1], -1))
		return torch.cat(feats, dim=-1)		# T x N x F

	def forward(self, batch):
		X = self._features(batch)
		sel = torch.softmax(self.select_logits / self.select_temp, dim=-1)	# C x D x F
		x = torch.einsum('tnf,cdf->cdtn', X, sel)
		d = torch.tanh(self.direction)[..., None, None]
		cond = torch.sigmoid(d * (x - self.threshold[..., None, None]) / self.temp)
		return torch.tanh(self.sign)[:, None, None] * cond.prod(dim=1)	# C x T x N

	def anneal(self, factor=0.9, min_temp=0.01, min_select_temp=0.05):
		self.temp = max(self.temp * factor, min_temp)
		self.select_temp = max(self.select_temp * factor, min_select_temp)

	@torch.no_grad()
	def hard_rules(self):
		"""Read out each instrument as a list of (feature, '>'|'<', threshold) conditions."""
		from .audit import Condition, Rule
		names = self.char_names + self.state_names
		rules = []
		for c in range(self.C):
			conds = []
			for d in range(self.D):
				f = int(self.select_logits[c, d].argmax())
				op = '>' if float(self.direction[c, d]) > 0 else '<'
				kind = 'char' if f < len(self.char_names) else 'state'
				conds.append(Condition(names[f], kind, op, float(self.threshold[c, d])))
			rules.append(Rule(tuple(conds)))
		return rules
