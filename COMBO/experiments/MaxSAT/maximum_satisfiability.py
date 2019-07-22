import os
import numpy as np

import torch

from COMBO.experiments.exp_utils import sample_init_points

MAXSAT_DIR_NAME = os.path.join(os.path.split(__file__)[0], 'maxsat2018_data')


class _MaxSAT(object):
	def __init__(self, data_filename, random_seed):
		f = open(os.path.join(MAXSAT_DIR_NAME, data_filename), 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		self.n_variables = int(line_str.split(' ')[2])
		self.n_clauses = int(line_str.split(' ')[3])
		self.n_vertices = np.array([2] * self.n_variables)
		clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
		f.close()
		weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
		weight_mean = np.mean(weights)
		weight_std = np.std(weights)
		self.weights = (weights - weight_mean) / weight_std
		self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses]

		self.suggested_init = sample_init_points(self.n_vertices, 20, random_seed)
		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		self.random_seed_info = 'R%04d' % random_seed
		for i in range(self.n_variables):
			adjmat = torch.diag(torch.ones(1), -1) + torch.diag(torch.ones(1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x):
		assert x.numel() == self.n_variables
		if x.dim() == 2:
			x = x.squeeze(0)
		x_np = (x.cpu() if x.is_cuda else x).numpy().astype(np.bool)
		satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
		return -np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)


class MaxSAT28(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='maxcut-johnson8-2-4.clq.wcnf', random_seed=random_seed)


class MaxSAT43(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='maxcut-hamming8-2.clq.wcnf', random_seed=random_seed)


class MaxSAT60(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='frb-frb10-6-4.wcnf', random_seed=random_seed)


if __name__ == '__main__':
	import torch as torch_
	maxsat_ = MaxSAT60()
	x_ = torch_.from_numpy(np.random.randint(0, 2, maxsat_.nbvar))
	eval_ = maxsat_.evaluate(x_)
	weight_sum_ = np.sum(maxsat_.weights)
	print(weight_sum_, eval_, weight_sum_ - eval_)
