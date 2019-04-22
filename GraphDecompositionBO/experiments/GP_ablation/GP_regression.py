import copy
import argparse

import numpy as np

import torch

from GraphDecompositionBO.graphGP.sampler.sample_hyper import slice_hyper
from GraphDecompositionBO.graphGP.sampler.sample_edgeweight import slice_edgeweight
from GraphDecompositionBO.graphGP.sampler.sample_partition import gibbs_partition

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference

from GraphDecompositionBO.experiments.GP_ablation.data_loader import load_highorderbinary


def GP_regression_posterior_sampling(model, input_data, output_data, categories, list_of_adjacency, log_beta, sorted_partition, n_sample, n_burn=100, thinning=1):
	'''

	:param model:
	:param input_data:
	:param output_data:
	:param categories:  1d np.array
	:param list_of_adjacency:
	:param log_beta:
	:param sorted_partition:
	:param n_sample:
	:param n_burn:
	:param thinning:
	:return:
	'''
	sample_hyper = []
	sample_log_beta = []
	sample_partition = []
	sample_freq = []
	sample_basis = []
	partition_sample = sorted_partition
	log_beta_sample = log_beta

	for _ in range(n_burn):
		slice_hyper(model=model, input_data=input_data, output_data=output_data, categories=categories, sorted_partition=sorted_partition)

		fourier_freq_list = model.kernel.fourier_freq_list
		fourier_basis_list = model.kernel.fourier_basis_list
		shuffled_ind = range(len(list_of_adjacency))
		np.random.shuffle(shuffled_ind)
		for ind in shuffled_ind:
			log_beta_sample, fourier_freq_list, fourier_basis_list = slice_edgeweight(model=model,
			                                                                          input_data=input_data,
			                                                                          output_data=output_data,
			                                                                          categories=categories,
			                                                                          list_of_adjacency=list_of_adjacency,
			                                                                          log_beta=log_beta_sample,
			                                                                          sorted_partition=partition_sample,
			                                                                          fourier_freq_list=fourier_freq_list,
			                                                                          fourier_basis_list=fourier_basis_list,
			                                                                          ind=ind)

	cnt = 0
	while len(sample_hyper) < n_sample:
		slice_hyper(model=model, input_data=input_data, output_data=output_data, categories=categories, sorted_partition=sorted_partition)

		fourier_freq_list = model.kernel.fourier_freq_list
		fourier_basis_list = model.kernel.fourier_basis_list
		# In 'Batched High-dimensional Bayesian Optimization via Structural Kernel Learning', similar additive structure is updated for every 50 iterations(evaluations)
		# This may be due to too much variability if decomposition is learned every iterations.
		# Thus, when multiple points are sampled, sweeping all inds for each sample may be not a good practice
		# For example if there are 50 variables and 10 samples are needed, then after shuffling indices, and 50/10 thinning can be used.
		shuffled_ind = range(len(list_of_adjacency))
		np.random.shuffle(shuffled_ind)
		for ind in shuffled_ind:
			# In each sampler, model.kernel fourier_freq_list, fourier_basis_list are updated.
			partition_sample, fourier_freq_list, fourier_basis_list = gibbs_partition(model=model,
			                                                                          input_data=input_data,
			                                                                          output_data=output_data,
			                                                                          categories=categories,
			                                                                          list_of_adjacency=list_of_adjacency,
			                                                                          log_beta=log_beta_sample,
			                                                                          sorted_partition=partition_sample,
			                                                                          fourier_freq_list=fourier_freq_list,
			                                                                          fourier_basis_list=fourier_basis_list,
			                                                                          ind=ind)
			log_beta_sample, fourier_freq_list, fourier_basis_list = slice_edgeweight(model=model,
			                                                                          input_data=input_data,
			                                                                          output_data=output_data,
			                                                                          categories=categories,
			                                                                          list_of_adjacency=list_of_adjacency,
			                                                                          log_beta=log_beta_sample,
			                                                                          sorted_partition=partition_sample,
			                                                                          fourier_freq_list=fourier_freq_list,
			                                                                          fourier_basis_list=fourier_basis_list,
			                                                                          ind=ind)
			cnt += 1
			if cnt == int(round(thinning * (len(sample_hyper) + 1))):
				sample_hyper.append(model.param_to_vec())
				sample_log_beta.append(log_beta_sample.clone())
				sample_partition.append(copy.deepcopy(sorted_partition))
				sample_freq.append([elm.clone() for elm in fourier_freq_list])
				sample_basis.append([elm.clone() for elm in fourier_basis_list])
			if len(sample_hyper) == n_sample:
				return sample_hyper, sample_log_beta, sample_partition, sample_freq, sample_basis


def marginal_likelihood(model, input_data, output_data, sample_hyper, sample_freq, sample_basis):
	assert len(sample_hyper) == len(sample_freq) == len(sample_basis)
	n_sample = len(sample_hyper)
	inference = Inference((input_data, output_data), model)
	ml_sum = 0
	for i in range(n_sample):
		model.kernel.fourier_freq_list = [elm.clone() for elm in sample_freq[i]]
		model.kernel.fourier_basis_list = [elm.clone() for elm in sample_basis[i]]
		ml_sum += -inference.negative_log_likelihood(hyper=sample_hyper[i])
	return ml_sum / float(n_sample)


def prediction_likelihood(model, train_input, train_output, test_input, test_output, sample_hyper, sample_freq, sample_basis):
	assert len(sample_hyper) == len(sample_freq) == len(sample_basis)
	n_sample = len(sample_hyper)
	inference = Inference((train_input, train_output), model)
	pl_sum = 0
	for i in range(n_sample):
		model.kernel.fourier_freq_list = [elm.clone() for elm in sample_freq[i]]
		model.kernel.fourier_basis_list = [elm.clone() for elm in sample_basis[i]]
		sample_pred_mean, sample_pred_var = inference.predict(test_input, hyper=sample_hyper[i])
		pl_sum = gaussian_likelihood(test_output, sample_pred_mean, sample_pred_var)
	pl_avg = pl_sum / float(n_sample)
	return torch.sum(pl_avg)


def gaussian_likelihood(data, mean, var):
	return 0.5 / (np.pi * var) ** 0.5 * torch.exp(-0.5 * (data - mean) ** 2 / var)


if __name__ == '__main__':
	parser_ = argparse.ArgumentParser(description='GOLD : Gaussian Process Regression')
	parser_.add_argument('-d', '--data_type', dest='data_type', type=int)
	parser_.add_argument('-s', '--train_data_scale', dest='train_data_scale', type=int)
	parser_.add_argument('-r', '--random_seed', dest='random_seed', type=int)
	args_ = parser_.parse_args()

	(train_input_, train_output_), (test_input_, test_output_) = load_highorderbinary(data_type=args_.data_type,
	                                                                                  train_data_scale=args_.train_data_scale,
	                                                                                  random_seed=args_.random_seed)
	n_variables_ = train_input_.size(1)
	categories_ = np.array([2 for _ in range(n_variables_)])
	list_of_adjacency_ = []
	init_log_beta_ = torch.zeros(n_variables_)
	init_sorted_partition_ = [[m_] for m_ in range(n_variables_)]

	fourier_freq_list_ = []
	fourier_basis_list_ = []
	for i_ in range(n_variables_):
		adjmat_ = torch.diag(torch.ones(1), -1) + torch.diag(torch.ones(1), 1)
		list_of_adjacency_.append(adjmat_)
		laplacian_ = torch.diag(torch.sum(adjmat_, dim=0)) - adjmat_
		eigval_, eigvec_ = torch.symeig(laplacian_, eigenvectors=True)
		fourier_freq_list_.append(eigval_)
		fourier_basis_list_.append(eigvec_)
	kernel_ = DiffusionKernel(fourier_freq_list=fourier_freq_list_, fourier_basis_list=fourier_basis_list_)

	model_ = GPRegression(kernel=kernel_)
	model_.init_param(train_output_)

	posterior_sample_ = GP_regression_posterior_sampling(model=model_, input_data=train_input_, output_data=train_output_,
	                                                     categories=categories_, list_of_adjacency=list_of_adjacency_,
	                                                     log_beta=init_log_beta_, sorted_partition=init_sorted_partition_,
	                                                     n_sample=10, n_burn=10)
	sample_hyper_, sample_log_beta_, sample_partition_, sample_freq_, sample_basis_ = posterior_sample_