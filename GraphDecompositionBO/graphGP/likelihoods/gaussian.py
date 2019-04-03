import numpy as np
import torch
from torch.nn.parameter import Parameter

from GraphDecompositionBO.graphGP.likelihoods.likelihood import Likelihood, log_lower_bnd


class GaussianLikelihood(Likelihood):

	def __init__(self):
		super(GaussianLikelihood, self).__init__()
		self.log_noise_var = Parameter(torch.FloatTensor(1))
		self.noise_scale = 0.1

	def reset_parameters(self):
		self.log_noise_var.data.normal_(std=np.abs(np.random.standard_cauchy()) * self.noise_scale).pow_(2).log_()

	def out_of_bounds(self, vec=None):
		if vec is None:
			return (self.log_noise_var.data < log_lower_bnd).any() or (self.log_noise_var.data > 16 + np.log(self.noise_scale/0.1)).any()
		else:
			return (vec < log_lower_bnd).any() or (vec > 16 + np.log(self.noise_scale / 0.1)).any()

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.log_noise_var.data.clone()

	def vec_to_param(self, vec):
		self.log_noise_var.data = vec

	def prior_log_lik(self, vec):
		if (vec < log_lower_bnd).any() or (vec > 16 + np.log(self.noise_scale/0.1)).any():
			return -np.inf
		return np.log(np.log(1.0 + (self.noise_scale / np.exp(vec)) ** 2)).sum()

	def forward(self, input):
		return torch.exp(self.log_noise_var).repeat(input.size(0))

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = GaussianLikelihood()
	print(list(likelihood.parameters()))
