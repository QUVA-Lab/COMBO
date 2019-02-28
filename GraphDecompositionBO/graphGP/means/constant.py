import torch
from torch.nn.parameter import Parameter

from CombinatorialBO.graphGP.means.mean import Mean


class ConstantMean(Mean):

	def __init__(self):
		super(ConstantMean, self).__init__()
		self.const_mean = Parameter(torch.FloatTensor(1))

	def reset_parameters(self):
		self.const_mean.data.normal_(std=10.0) # approximation to uniform

	def out_of_bounds(self, vec=None):
		if vec is None:
			return False
		else:
			return False

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.const_mean.data.clone()

	def vec_to_param(self, vec):
		self.const_mean.data = vec

	def prior_log_lik(self, vec):
		return 0

	def forward(self, input):
		return self.const_mean * torch.ones(input.size(0), 1, device=input.device)

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = ConstantMean()
	print(list(likelihood.parameters()))