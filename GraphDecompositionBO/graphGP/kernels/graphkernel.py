import numpy as np

import torch
from torch.nn.parameter import Parameter
from GraphDecompositionBO.graphGP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd


class GraphKernel(GPModule):

	def __init__(self, fourier_coef_list, fourier_basis_list):
		super(GraphKernel, self).__init__()
		self.log_amp = Parameter(torch.FloatTensor(1))
		self.amp_scale = 1.0
		self.n_factors = len(fourier_coef_list)
		self.fourier_coef = fourier_coef_list
		self.fourier_basis = fourier_basis_list
		self.n_vertices = torch.Tensor([elm.numel() for elm in self.fourier_coef])

	def reset_parameters(self):
		self.log_amp.data.normal_()

	def init_parameters(self, amp):
		self.log_amp.data.fill_(np.log(amp + 1e-4))

	def out_of_bounds(self, vec=None):
		if vec is None:
			if (log_lower_bnd <= self.log_amp.data).all() and (self.log_amp.data <= log_upper_bnd).all():
				return False
		else:
			if log_lower_bnd <= vec[0] <= log_upper_bnd:
				return False
		return True

	def n_params(self):
		return 1

	def param_to_vec(self):
		flat_param_list = [self.log_amp.data.clone()]
		return torch.cat(flat_param_list)

	def vec_to_param(self, vec):
		assert vec.numel() == 1
		self.log_amp.data = vec[:1]

	def prior_log_lik(self, vec):
		assert vec.size == 1
		return -0.5 * 0.25 * vec ** 2 / self.amp_scale ** 2

	def forward(self, input1, input2=None):
		raise NotImplementedError
