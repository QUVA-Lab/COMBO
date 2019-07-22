import torch

from COMBO.graphGP.likelihoods.likelihood import Likelihood


class GaussianLikelihood(Likelihood):

	def __init__(self):
		super(GaussianLikelihood, self).__init__()
		self.log_noise_var = torch.FloatTensor(1)
		self.noise_scale = 0.1

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.log_noise_var.clone()

	def vec_to_param(self, vec):
		self.log_noise_var = vec.clone()

	def forward(self, input):
		return torch.exp(self.log_noise_var).repeat(input.size(0))

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = GaussianLikelihood()
	print(list(likelihood.parameters()))
