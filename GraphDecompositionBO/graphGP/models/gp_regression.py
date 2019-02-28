from CombinatorialBO.graphGP.likelihoods.gaussian import GaussianLikelihood
from CombinatorialBO.graphGP.means.constant import ConstantMean
from CombinatorialBO.graphGP.models.gp import GP


class GPRegression(GP):

	def __init__(self, kernel, mean=ConstantMean()):
		super(GPRegression, self).__init__()
		self.kernel = kernel
		self.mean = mean
		self.likelihood = GaussianLikelihood()