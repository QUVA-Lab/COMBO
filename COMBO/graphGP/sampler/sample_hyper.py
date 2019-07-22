import numpy as np

import torch

from COMBO.graphGP.inference.inference import Inference
from COMBO.graphGP.sampler.tool_partition import group_input
from COMBO.graphGP.sampler.tool_slice_sampling import univariate_slice_sampling
from COMBO.graphGP.sampler.priors import log_prior_constmean, log_prior_noisevar, log_prior_kernelamp


def slice_hyper(model, input_data, output_data, n_vertices, sorted_partition):
	"""

	:param model:
	:param input_data:
	:param output_data:
	:return:
	"""
	grouped_input_data = group_input(input_data=input_data, sorted_partition=sorted_partition, n_vertices=n_vertices)
	inference = Inference(train_data=(grouped_input_data, output_data), model=model)
	# Randomly shuffling order can be considered, here the order is in const_mean, kernel_amp, noise_var
	slice_constmean(inference)
	slice_kernelamp(inference)
	slice_noisevar(inference)


def slice_constmean(inference):
	"""
	Slice sampling const_mean, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.mean.const_mean'
	:param inference:
	:return:
	"""
	output_min = torch.min(inference.train_y).item()
	output_max = torch.max(inference.train_y).item()
	def logp(constmean):
		"""
		:param constmean: numeric(float)
		:return: numeric(float)
		"""
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
	"""
	Slice sampling log_noise_var, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.likelihood.log_noise_var'
	:param inference:
	:return:
	"""

	def logp(log_noise_var):
		"""
		:param log_noise_var: numeric(float)
		:return: numeric(float)
		"""
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
	"""
	Slice sampling log_amp, this function does not need to return a sampled value
	This directly modifies parameters in the argument 'inference.model.kernel.log_amp'
	:param inference:
	:return:
	"""
	output_var = torch.var(inference.train_y).item()
	kernel_min = np.prod([torch.mean(torch.exp(-fourier_freq[-1])).item() / torch.mean(torch.exp(-fourier_freq)).item() for fourier_freq in inference.model.kernel.fourier_freq_list])
	kernel_max = np.prod([torch.mean(torch.exp(-fourier_freq[0])).item() / torch.mean(torch.exp(-fourier_freq)).item() for fourier_freq in inference.model.kernel.fourier_freq_list])
	def logp(log_amp):
		"""
		:param log_amp: numeric(float)
		:return: numeric(float)
		"""
		log_prior = log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max)
		if np.isinf(log_prior):
			return log_prior
		inference.model.kernel.log_amp.fill_(log_amp)
		log_likelihood = float(-inference.negative_log_likelihood(hyper=inference.model.param_to_vec()))
		return log_prior + log_likelihood

	x0 = float(inference.model.kernel.log_amp)
	x1 = univariate_slice_sampling(logp, x0)
	inference.model.kernel.log_amp.fill_(x1)
	return