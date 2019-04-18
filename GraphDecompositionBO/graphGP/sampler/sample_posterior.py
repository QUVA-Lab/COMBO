import copy
import numpy as np

from GraphDecompositionBO.graphGP.sampler.sample_hyper import slice_hyper
from GraphDecompositionBO.graphGP.sampler.sample_edgeweight import slice_edgeweight
from GraphDecompositionBO.graphGP.sampler.sample_partition import gibbs_partition


def posterior_sampling(model, input_data, output_data, categories, list_of_adjacency, log_beta, sorted_partition, n_sample, thinning=None):
	'''

	:param model: model.kernel members fourier_freq_list, fourier_basis_list is grouped
	:param input_data:
	:param output_data:
	:param list_of_adjacency: list of 2D torch.Tensor of adjacency matrix
	:param log_beta:
	:param sorted_partition: Partition of {0, ..., K-1}, list of subsets(list)
	:param n_sample:
	:param thinning:
	:return:
	'''
	sample_hyper = []
	sample_log_beta = []
	sample_partition = []
	sample_freq = []
	sample_basis = []
	if thinning is None:
		thinning = n_sample
	partition_sample = sorted_partition
	log_beta_sample = log_beta

	cnt = 0
	while len(sample_hyper) < n_sample:
		slice_hyper(model=model, input_data=input_data, output_data=output_data)

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
			partition_sample, fourier_freq_list, fourier_basis_list = gibbs_partition(model=model, input_data=input_data, output_data=output_data, categories=categories,
			                                                                          list_of_adjacency=list_of_adjacency, log_beta=log_beta_sample, sorted_partition=partition_sample,
			                                                                          fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list, ind=ind)
			log_beta_sample, fourier_freq_list, fourier_basis_list = slice_edgeweight(model=model, input_data=input_data, output_data=output_data,
			                                                                          list_of_adjacency=list_of_adjacency, log_beta=log_beta_sample, sorted_partition=partition_sample,
			                                                                          fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list, ind=ind)
			cnt += 1
			if cnt == int(round(thinning * (len(sample_hyper) + 1))):
				sample_hyper.append(model.vec_to_param())
				sample_log_beta.append(log_beta_sample.clone())
				sample_partition.append(copy.deepcopy(sorted_partition))
				sample_freq.append([elm.clone() for elm in fourier_freq_list])
				sample_basis.append([elm.clone() for elm in fourier_basis_list])
			if len(sample_hyper) == n_sample:
				return sample_hyper, sample_log_beta, sample_partition, sample_freq, sample_basis
