import torch

from GraphDecompositionBO.graphGP.modules.gp_modules import GPModule


class GraphKernel(GPModule):

	def __init__(self, grouped_log_beta, fourier_freq_list, fourier_basis_list):
		super(GraphKernel, self).__init__()
		self.log_amp = torch.FloatTensor(1)
		self.grouped_log_beta = grouped_log_beta
		self.fourier_freq_list = fourier_freq_list
		self.fourier_basis_list = fourier_basis_list

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.log_amp.clone()

	def vec_to_param(self, vec):
		assert vec.numel() == 1
		self.log_amp = vec[:1].clone()

	def forward(self, input1, input2=None):
		raise NotImplementedError
