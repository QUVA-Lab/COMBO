import torch

from GraphDecompositionBO.graphGP.means.mean import Mean


class ConstantMean(Mean):

	def __init__(self):
		super(ConstantMean, self).__init__()
		self.const_mean = torch.FloatTensor(1)

	def n_params(self):
		return 1

	def param_to_vec(self):
		return self.const_mean.clone()

	def vec_to_param(self, vec):
		self.const_mean = vec.clone()

	def forward(self, input):
		return self.const_mean * torch.ones(input.size(0), 1, device=input.device)

	def __repr__(self):
		return self.__class__.__name__


if __name__ == '__main__':
	likelihood = ConstantMean()
	print(list(likelihood.parameters()))