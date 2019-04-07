import numpy as np
from GraphDecompositionBO.sampler.tool_partition import compute_group_size

GRAPH_SIZE_LIMIT = 1024 + 2


# TODO define a prior prior for (scalar) log_beta


def log_prior_constmean(constmean):
	return 0


def log_prior_noisevar(log_noise_var):
	return 0


def log_prior_kernelamp(log_amp):
	return 0


def log_prior_edgeweight(log_beta_ind):
	'''
	:param log_beta_ind: numeric(float), ind-th element of 'log_beta'
	:return: numeric(float)
	'''
	if np.exp(log_beta_ind) > 2.0:
		return -float('inf')
	else:
		return np.log(1.0 / 2.0)


def log_prior_partition(sorted_partition, categories):
	'''
	Log of unnormalized density of given partition
	this prior prefers well-spread partition, which is quantified by induced entropy as below
	:param sorted_partition:
	:return:
	'''
	if compute_group_size(sorted_partition=sorted_partition, categories=categories) > GRAPH_SIZE_LIMIT:
		return -float('inf')
	else:
		prob_mass = np.array([np.sum(np.log(categories[subset])) for subset in sorted_partition])
		prob_mass /= np.sum(prob_mass)
		return np.log(np.sum(-prob_mass * np.log(prob_mass)))
