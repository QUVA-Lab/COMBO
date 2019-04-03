from GraphDecompositionBO.graphGP.modules.gp_modules import GPModule, log_lower_bnd, log_upper_bnd


class Likelihood(GPModule):

	def __init__(self):
		super(Likelihood, self).__init__()
