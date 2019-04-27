import numpy as np
from scipy.special import gammaln

from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_group_size

# For numerical stability in exponential
LOG_LOWER_BND = -12.0
LOG_UPPER_BND = 20.0
# For sampling stability
STABLE_MEAN_RNG = 1.0
# Hyperparameter for graph factorization
GRAPH_SIZE_LIMIT = 1024 + 2


# TODO define a prior prior for (scalar) log_beta


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
	# Uniform prior
	# return 0
	# Truncated Gaussian
	return -np.log(stable_dev / 2.0) - 0.5 * (constmean - output_mid) ** 2 / (stable_dev / 2.0) ** 2


def log_prior_noisevar(log_noise_var):
	if log_noise_var < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_noise_var:
		return -float('inf')
	return np.log(np.log(1.0 + (0.1 / np.exp(log_noise_var)) ** 2))


def log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max):
	if log_amp < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_amp:
		return -float('inf')
	# LogNormal
	log_amp_lower = np.log(output_var / kernel_max)
	log_amp_upper = np.log(output_var / kernel_min)
	log_amp_mid = 0.5 * (log_amp_upper + log_amp_lower)
	log_amp_std = 0.5 * (log_amp_upper - log_amp_lower) / 2.0
	return -np.log(log_amp_std) - 0.5 * (log_amp - log_amp_mid) ** 2 / log_amp_std ** 2
	# return
	# Uniform
	# return 0 if kernel_min < output_var / amp < kernel_max else -float('inf')
	# Gamma
	# shape = output_var
	# rate = 1.0
	# return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * log_amp - rate * np.exp(log_amp)


def log_prior_edgeweight(log_beta_i, ind, sorted_partition):
	'''
	:param log_beta_i: numeric(float), ind-th element of 'log_beta'
	:return: numeric(float)
	'''
	# Gamma prior
	shape = 1.0
	rate = 1.0 / len(sorted_partition) ** 0.5
	if log_beta_i < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_beta_i:
		return -float('inf')
	return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * log_beta_i - rate * np.exp(log_beta_i)
	# Uniform prior
	# if og_beta_i < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(2.0)) < log_beta_i:
	# 	return -float('inf')
	# else:
	# 	return np.log(1.0 / 2.0)


def log_prior_partition(sorted_partition, n_vertices):
	'''
	Log of unnormalized density of given partition
	this prior prefers well-spread partition, which is quantified by induced entropy.
	Density is proportional to the entropy of a unnormalized probability vector consisting of [log(n_vertices in subgraph_i)]_i=1...N
	:param sorted_partition:
	:return:
	'''
	if len(sorted_partition) == 1 or compute_group_size(sorted_partition=sorted_partition, n_vertices=n_vertices) > GRAPH_SIZE_LIMIT:
		return -float('inf')
	else:
		prob_mass = np.array([np.sum(np.log(n_vertices[subset])) for subset in sorted_partition])
		prob_mass /= np.sum(prob_mass)
		return np.log(np.sum(-prob_mass * np.log(prob_mass)))
