import math

import numpy as np
import sampyl as smp

import torch
from torch.nn import Parameter
from CombinatorialBO.graphGP.kernels.graphkernel import GraphKernel, log_lower_bnd, log_upper_bnd


class DiffusionKernel(GraphKernel):
	"""
	Usually Graph Kernel means a kernel between graphs, here this kernel is a kernel between vertices on a graph
	"""
	def __init__(self, fourier_coef_list, fourier_basis_list, ard=True):
		super(DiffusionKernel, self).__init__(fourier_coef_list, fourier_basis_list)
		self.ard = ard
		self.log_beta = Parameter(torch.zeros(self.n_factors if ard else 1))
		self.beta_min = 1e-4
		self.beta_max = 2.0

	def init_beta(self, init_beta):
		self.log_beta.data.fill_(math.log(init_beta))

	def init_parameters(self, amp):
		super(DiffusionKernel, self).init_parameters(amp)
		self.init_beta(0.5)

	def n_params(self):
		return self.log_beta.numel() + super(DiffusionKernel, self).n_params()

	def param_to_vec(self):
		flat_param_list = [super(DiffusionKernel, self).param_to_vec()]
		flat_param_list += [self.log_beta.data.clone()]
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		n_super = super(DiffusionKernel, self).n_params()
		super(DiffusionKernel, self).vec_to_param(vec[:n_super])
		self.log_beta.data = vec[n_super:n_super + self.log_beta.numel()]

	def prior_log_lik(self, vec):
		n_super = super(DiffusionKernel, self).n_params()
		log_lik = super(DiffusionKernel, self).prior_log_lik(vec[:n_super])
		# log_lik += smp.uniform(np.exp(vec[n_super:]), lower=self.beta_min, upper=self.beta_max)
		log_lik += smp.uniform(vec[n_super:], lower=math.log(self.beta_min), upper=math.log(self.beta_max))
		return log_lik

	def out_of_bounds(self, vec=None):
		n_super = super(DiffusionKernel, self).n_params()
		if vec is None:
			if (self.log_beta.data <= math.log(self.beta_max)).all():
				return False
		else:
			if (vec[n_super:n_super + self.log_beta.numel()] <= math.log(self.beta_max)).all():
				return super(DiffusionKernel, self).out_of_bounds(vec[:n_super])
		return True

	def forward(self, x1, x2=None, diagonal=False):
		"""
		:param x1, x2: each row is a vector with vertex numbers starting from 0 for each 
		:return: 
		"""
		if diagonal:
			assert x2 is None

		stabilizer = 0
		if x2 is None:
			x2 = x1
			if diagonal:
				stabilizer = 1e-6 * x1.new_ones(x1.size(0), 1, dtype=torch.float32)
			else:
				stabilizer = torch.diag(1e-6 * x1.new_ones(x1.size(0), dtype=torch.float32))

		regularizer_inv_summand = [1]
		for i in range(self.n_factors):
			subvec1 = self.fourier_basis[i][x1[:, i]]
			subvec2 = self.fourier_basis[i][x2[:, i]]
			freq_transform = torch.exp(-torch.exp(self.log_beta[i] if self.ard else self.log_beta) * self.fourier_coef[i])
			if diagonal:
				factor_gram = torch.sum(subvec1 * freq_transform.unsqueeze(0) * subvec2, dim=1, keepdim=True)
			else:
				factor_gram = torch.matmul(subvec1 * freq_transform.unsqueeze(0), subvec2.t())
			regularizer_inv_summand[0] *= factor_gram / torch.mean(freq_transform)
		return torch.exp(self.log_amp) * (torch.sum(torch.stack(regularizer_inv_summand), dim=0) + stabilizer)


if __name__ == '__main__':
	weight_mat_list = []
	graph_size = []
	n_variables = 24
	n_data = 10

	fourier_coef_list = []
	fourier_basis_list = []
	for i in range(n_variables):
		n_v = int(torch.randint(50, 51, (1,))[0])
		graph_size.append(n_v)
		# type_num = int(torch.randint(0, 3, (1,))[0])
		type_num = 1
		if type_num < 0:
			adjmat = (torch.rand(n_v, n_v) * 0.5 - 0.25).clamp(min=0).tril(-1)
			adjmat = adjmat + adjmat.t()
		elif type_num == 0:
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			adjmat *= n_v * (n_v - 1) / 2.0
		elif type_num == 1:
			adjmat = torch.ones(n_v, n_v) - torch.eye(n_v)
		wgtsum = torch.sum(adjmat, dim=0)
		laplacian = (torch.diag(wgtsum) - adjmat)# / wgtsum ** 0.5 / wgtsum.unsqueeze(1) ** 0.5
		eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
		fourier_coef_list.append(eigval)
		fourier_basis_list.append(eigvec)
	k = DiffusionKernel(fourier_coef_list=fourier_coef_list, fourier_basis_list=fourier_basis_list)
	k.log_amp.data.fill_(0)
	k.log_beta.data.fill_(math.log(0.05))
	input_data = torch.empty([0, n_variables])
	for i in range(n_data):
		datum = torch.zeros([1, n_variables])
		for e, c in enumerate(graph_size):
			datum[0, e] = torch.randint(0, c, (1,))[0]
		input_data = torch.cat([input_data, datum], dim=0)
	input_data = input_data.long()
	data = k(input_data)

	# data = factor_gram / torch.mean(freq_transform)
	for r in range(data.size(0)):
		print(' '.join([' %+.2E' % data[r, i] for i in range(data.size(1))]))
