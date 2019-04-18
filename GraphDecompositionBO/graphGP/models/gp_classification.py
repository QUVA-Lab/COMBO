from GraphDecompositionBO.graphGP.likelihoods.degenerated import DegeneratedLikelihood
from GraphDecompositionBO.graphGP.means.constant import ConstantMean
from GraphDecompositionBO.graphGP.models.gp import GP


class GPClassification(GP):

	def __init__(self, kernel, mean=ConstantMean()):
		super(GPClassification, self).__init__()
		self.kernel = kernel
		self.mean = mean
		self.likelihood = DegeneratedLikelihood()
