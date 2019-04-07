import numpy as np

import torch


from GraphDecompositionBO.sampler.tool_slice_sampling import univariate_slice_sampling
from GraphDecompositionBO.sampler.priors import log_prior_constmean, log_prior_noisevar, log_prior_kernelamp


def slice_constmean(inference):
	'''
	Slice sampling const_mean, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.mean.const_mean'
	:param inference:
	:return:
	'''
	def logp(constmean):
		'''
		:param constmean: numeric(float)
		:return: numeric(float)
		'''
		log_prior = log_prior_constmean(constmean)
		if np.isinf(log_prior):
			return log_prior
		inference.model.mean.const_mean.data.fill_(constmean)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(model.mean.const_mean)
	x1 = univariate_slice_sampling(logp, x0)
	model.mean.const_mean.data.fill_(x1)
	return


def slice_noisevar(inference):
	'''
	Slice sampling log_noise_var, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.likelihood.log_noise_var'
	:param inference:
	:return:
	'''

	def logp(log_noise_var):
		'''
		:param log_noise_var: numeric(float)
		:return: numeric(float)
		'''
		log_prior = log_prior_noisevar(log_noise_var)
		if np.isinf(log_prior):
			return log_prior
		inference.model.likelihood.log_noise_var.data.fill_(log_noise_var)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(model.likelihood.log_noise_var)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.likelihood.log_noise_var.data.fill_(x1)
	return


def slice_kernelamp(inference):
	'''
	Slice sampling log_amp, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.kernel.log_amp'
	:param inference:
	:return:
	'''

	def logp(log_amp):
		'''
		:param log_amp: numeric(float)
		:return: numeric(float)
		'''
		log_prior = log_prior_kernelamp(log_amp)
		if np.isinf(log_prior):
			return log_prior
		inference.model.kernel.log_amp.data.fill_(log_amp)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(model.kernel.log_amp)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.kernel.log_amp.data.fill_(x1)
	return


if __name__ == '__main__':
	pass
	import progressbar
	import time
	from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
	from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
	from GraphDecompositionBO.sampler.tool_partition import sort_partition, compute_unit_in_group, group_input, ungroup_input
	n_vars = 50
	n_data = 60
	categories = np.random.randint(2, 3, n_vars)
	list_of_adjacency = []
	for d in range(n_vars):
		adjacency = torch.ones(categories[d], categories[d])
		adjacency[range(categories[d]), range(categories[d])] = 0
		list_of_adjacency.append(adjacency)
	input_data = torch.zeros(n_data, n_vars).long()
	output_data = torch.randn(n_data, 1)
	for a in range(n_vars):
		input_data[:, a] = torch.randint(0, categories[a], (n_data,))
	inds = range(n_vars)
	np.random.shuffle(inds)
	b = 0
	random_partition = []
	while b < n_vars:
		subset_size = np.random.poisson(2) + 1
		random_partition.append(inds[b:b + subset_size])
		b += subset_size
	sorted_partition = sort_partition(random_partition)
	unit_in_group = compute_unit_in_group(sorted_partition, categories)
	grouped_input_data = group_input(input_data, sorted_partition, unit_in_group)
	input_data_re = ungroup_input(grouped_input_data, sorted_partition, unit_in_group)
	amp = torch.std(output_data, dim=0)
	log_beta = torch.randn(n_vars)
	model = GPRegression(kernel=DiffusionKernel(fourier_freq_list=[], fourier_basis_list=[]))
	model.kernel.log_amp.data = torch.log(amp)
	model.mean.const_mean.data = torch.mean(output_data, dim=0)
	model.likelihood.log_noise_var.data = torch.log(amp / 1000.)