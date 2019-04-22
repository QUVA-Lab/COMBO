import torch

from GraphDecompositionBO.graphGP.likelihoods.likelihood import Likelihood


class DegeneratedLikelihood(Likelihood):

	def __init__(self):
		super(DegeneratedLikelihood, self).__init__()
		self.dummy = torch.empty(0)

	def n_params(self):
		return 0

	def param_to_vec(self):
		return self.dummy.clone()

	def vec_to_param(self, vec):
		pass

	def forward(self, input):
		return input.new_ones(input.size(0))

	def __repr__(self):
		return self.__class__.__name__
