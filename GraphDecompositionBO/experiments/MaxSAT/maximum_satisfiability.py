import numpy as np
import torch

from GraphDecompositionBO.experiments.exp_utils import sample_init_points


class MaxSAT(object):

	def __init__(self, data_filename, random_seed=None):
		f = open(data_filename, 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		self.nbvar = int(line_str.split(' ')[2])
		self.nbclause = int(line_str.split(' ')[3])
		clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
		self.weights = np.array([elm[0] for elm in clauses])
		self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for weight, clause in clauses]
		f.close()

		self.suggested_init = torch.empty(0).long()
		self.suggested_init = torch.cat([self.suggested_init, sample_init_points([2] * self.nbvar, 20 - self.suggested_init.size(0), random_seed).long()], dim=0)

	def evaluate(self, x):
		assert x.numel() == self.nbvar
		if x.dim() == 2:
			x = x.squeeze(0)
		x_np = (x.cpu() if x.is_cuda else x).numpy().astype(np.bool)
		satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
		return np.sum(self.weights * satisfied)


if __name__ == '__main__':
	import torch as _torch
	_maxsat = MaxSAT('/home/coh1/Downloads/' + 'mse18-complete-weighted-benchmarks/frb-frb10-6-1.wcnf')
	_x = _torch.from_numpy(np.random.randint(0, 2, _maxsat.nbvar))
	_eval = _maxsat.evaluate(_x)
	weight_sum = np.sum(_maxsat.weights)
	print(weight_sum, _eval, weight_sum - _eval)