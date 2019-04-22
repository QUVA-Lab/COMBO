import numpy as np

import torch

from GraphDecompositionBO.graphGP.inference.inference import Inference
from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_unit_in_group, group_input
from GraphDecompositionBO.graphGP.sampler.tool_slice_sampling import univariate_slice_sampling
from GraphDecompositionBO.graphGP.sampler.priors import log_prior_constmean, log_prior_noisevar, log_prior_kernelamp


def slice_hyper(model, input_data, output_data, categories, sorted_partition):
	'''

	:param model:
	:param input_data:
	:param output_data:
	:return:
	'''
	unit_in_group = compute_unit_in_group(sorted_partition=sorted_partition, categories=categories)
	grouped_input_data = group_input(input_data=input_data, sorted_partition=sorted_partition, unit_in_group=unit_in_group)
	inference = Inference(train_data=(grouped_input_data, output_data), model=model)
	# Randomly shuffling order can be considered, here the order is in const_mean, kernel_amp, noise_var
	slice_constmean(inference)
	slice_kernelamp(inference)
	slice_noisevar(inference)


def slice_constmean(inference):
	'''
	Slice sampling const_mean, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.mean.const_mean'
	:param inference:
	:return:
	'''
	output_min = torch.min(inference.train_y).item()
	output_max = torch.max(inference.train_y).item()
	def logp(constmean):
		'''
		:param constmean: numeric(float)
		:return: numeric(float)
		'''
		log_prior = log_prior_constmean(constmean, output_min=output_min, output_max=output_max)
		if np.isinf(log_prior):
			return log_prior
		inference.model.mean.const_mean.fill_(constmean)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(inference.model.mean.const_mean)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.mean.const_mean.fill_(x1)
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
		inference.model.likelihood.log_noise_var.fill_(log_noise_var)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(inference.model.likelihood.log_noise_var)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.likelihood.log_noise_var.fill_(x1)
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
		inference.model.kernel.log_amp.fill_(log_amp)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(inference.model.kernel.log_amp)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.kernel.log_amp.fill_(x1)
	return


if __name__ == '__main__':
	pass
	from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
	from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
	from GraphDecompositionBO.sampler.tool_partition import sort_partition, compute_unit_in_group, group_input, ungroup_input
	n_vars_ = 50
	n_data_ = 60
	categories_ = np.random.randint(2, 3, n_vars_)
	list_of_adjacency_ = []
	for d_ in range(n_vars_):
		adjacency_ = torch.ones(categories_[d_], categories_[d_])
		adjacency_[range(categories_[d_]), range(categories_[d_])] = 0
		list_of_adjacency_.append(adjacency_)
	input_data_ = torch.zeros(n_data_, n_vars_).long()
	output_data_ = torch.randn(n_data_, 1)
	for a_ in range(n_vars_):
		input_data_[:, a_] = torch.randint(0, categories_[a_], (n_data_,))
	inds_ = range(n_vars_)
	np.random.shuffle(inds_)
	b_ = 0
	random_partition_ = []
	while b_ < n_vars_:
		subset_size_ = np.random.poisson(2) + 1
		random_partition_.append(inds_[b_:b_ + subset_size_])
		b_ += subset_size_
	sorted_partition_ = sort_partition(random_partition_)
	unit_in_group_ = compute_unit_in_group(sorted_partition_, categories_)
	grouped_input_data_ = group_input(input_data_, sorted_partition_, unit_in_group_)
	input_data_re_ = ungroup_input(grouped_input_data_, sorted_partition_, unit_in_group_)
	amp_ = torch.std(output_data_, dim=0)
	log_beta_ = torch.randn(n_vars_)
	model_ = GPRegression(kernel=DiffusionKernel(fourier_freq_list=[], fourier_basis_list=[]))
	model_.kernel.log_amp = torch.log(amp_)
	model_.mean.const_mean = torch.mean(output_data_, dim=0)
	model_.likelihood.log_noise_var = torch.log(amp_ / 1000.)