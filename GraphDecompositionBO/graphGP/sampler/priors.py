import numpy as np

import torch

from GraphDecompositionBO.sampler.tool_partition import compute_group_size

# For numerical stability in exponential
LOG_LOWER_BND = -12.0
LOG_UPPER_BND = 20.0
# For sampling stability
STABLE_MEAN_RNG = 1.0
# Hyperparameter for graph factorization
GRAPH_SIZE_LIMIT = 1024 + 2


# TODO define a prior prior for (scalar) log_beta


def init_parameters(model, output_data):
	amp = float(torch.std(output_data))
	model.kernel.init_parameters(amp)
	model.mean.const_mean.data.fill_(float(torch.mean(output_data)))
	model.likelihood.log_noise_var.data.fill_(np.log(amp / 1000.0))


def log_prior_constmean(constmean, output_min, output_max):
	'''
	:param constmean: numeric(float)
	:param output_min: numeric(float)
	:param output_max: numeric(float)
	:return:
	'''
	output_mid = (output_min + output_max) / 2.0
	stable_dev = (output_max - output_min) * STABLE_MEAN_RNG / 2.0
	# Unstable parameter in sampling
	if constmean < output_mid - stable_dev or output_mid + stable_dev < constmean:
		return -float('inf')
	return 0


def log_prior_noisevar(log_noise_var):
	if log_noise_var < LOG_LOWER_BND or log_noise_var > 16:
		return -float('inf')
	return np.log(np.log(1.0 + (0.1 / np.exp(log_noise_var)) ** 2))


def log_prior_kernelamp(log_amp):
	if log_amp < LOG_LOWER_BND or LOG_UPPER_BND < log_amp:
		return -float('inf')
	return -0.5 * 0.25 * log_amp ** 2


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
	this prior prefers well-spread partition, which is quantified by induced entropy.
	Density is proportional to the entropy of a unnormalized probability vector consisting of [log(n_vertices in subgraph_i)]_i=1...N
	:param sorted_partition:
	:return:
	'''
	if compute_group_size(sorted_partition=sorted_partition, categories=categories) > GRAPH_SIZE_LIMIT:
		return -float('inf')
	else:
		prob_mass = np.array([np.sum(np.log(categories[subset])) for subset in sorted_partition])
		prob_mass /= np.sum(prob_mass)
		return np.log(np.sum(-prob_mass * np.log(prob_mass)))
