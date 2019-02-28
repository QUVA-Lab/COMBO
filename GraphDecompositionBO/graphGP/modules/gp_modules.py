from torch.nn.modules.module import Module

log_lower_bnd = -12.0
log_upper_bnd = 20.0


class GPModule(Module):

	def __init__(self):
		super(GPModule, self).__init__()

	def reset_parameters(self):
		raise NotImplementedError

	def init_parameters(self):
		raise NotImplementedError

	def out_of_bounds(self, vec=None):
		raise NotImplementedError

	def n_params(self):
		raise NotImplementedError

	def param_to_vec(self):
		raise NotImplementedError

	def vec_to_param(self, vec):
		raise NotImplementedError

	def prior_log_lik(self, vec):
		raise NotImplementedError

	def __repr__(self):
		return self.__class__.__name__
