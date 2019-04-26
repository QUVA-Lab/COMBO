import sys
import copy
import progressbar
import numpy as np

from GraphDecompositionBO.graphGP.sampler.sample_hyper import slice_hyper
from GraphDecompositionBO.graphGP.sampler.sample_edgeweight import slice_edgeweight
from GraphDecompositionBO.graphGP.sampler.sample_partition import gibbs_partition
from GraphDecompositionBO.graphGP.sampler.tool_partition import ind_to_perturb, strong_product


def posterior_sampling(model, input_data, output_data, n_vertex, adj_mat_list,
                       log_beta, sorted_partition, n_sample, n_burn=0, n_thin=1):
	"""

	:param model:
	:param input_data:
	:param output_data:
	:param n_vertex:
	:param adj_mat_list:
	:param log_beta:
	:param sorted_partition:
	:param n_sample:
	:param n_burn:
	:param n_thin:
	:return:
	"""
	hyper_samples = []
	log_beta_samples = []
	partition_samples = []
	freq_samples = []
	basis_samples = []
	edge_mat_samples = []

	partition_sample = sorted_partition
	log_beta_sample = log_beta
	fourier_freq_list = model.kernel.fourier_freq_list
	fourier_basis_list = model.kernel.fourier_basis_list
	edge_mat_list = [strong_product(adj_mat_list, input_data.new_ones(len(adj_mat_list)), subset) for subset in
	                 sorted_partition]

	shuffled_partition_ind = []
	bar = progressbar.ProgressBar(max_value=n_sample * n_thin + n_burn)
	for s in range(0, n_sample * n_thin + n_burn):
		# In 'Batched High-dimensional Bayesian Optimization via Structural Kernel Learning',
		# similar additive structure is updated for every 50 iterations(evaluations)
		# This may be due to too much variability if decomposition is learned every iterations.
		# Thus, when multiple points are sampled, sweeping all inds for each sample may be not a good practice
		# For example if there are 50 variables and 10 samples are needed,
		# then after shuffling indices, and 50/10 thinning can be used.
		if len(shuffled_partition_ind) == 0:
			shuffled_partition_ind = range(len(adj_mat_list))
			np.random.shuffle(shuffled_partition_ind)
		partition_ind = ind_to_perturb(sorted_partition=partition_sample, n_vertex=n_vertex)
		# partition_ind = shuffled_partition_ind.pop()
		gibbs_tuple = gibbs_partition(model, input_data, output_data, n_vertex, adj_mat_list,
		                              log_beta=log_beta_sample, sorted_partition=partition_sample,
		                              fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list,
		                              edge_mat_list=edge_mat_list, ind=partition_ind)
		partition_sample, fourier_freq_list, fourier_basis_list, edge_mat_list = gibbs_tuple
		slice_hyper(model, input_data, output_data, n_vertex, sorted_partition=partition_sample)

		shuffled_beta_ind = range(len(adj_mat_list))
		np.random.shuffle(shuffled_beta_ind)
		for beta_ind in shuffled_beta_ind:
			# In each sampler, model.kernel fourier_freq_list, fourier_basis_list are updated.
			slice_tuple = slice_edgeweight(model, input_data, output_data, n_vertex, adj_mat_list,
			                               log_beta=log_beta_sample, sorted_partition=partition_sample,
			                               fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list,
			                               ind=beta_ind)
			log_beta_sample, fourier_freq_list, fourier_basis_list = slice_tuple
		if s >= n_burn and (s - n_burn + 1) % n_thin == 0:
			hyper_samples.append(model.param_to_vec())
			log_beta_samples.append(log_beta_sample.clone())
			partition_samples.append(copy.deepcopy(partition_sample))
			freq_samples.append([elm.clone() for elm in fourier_freq_list])
			basis_samples.append([elm.clone() for elm in fourier_basis_list])
			edge_mat_samples.append([elm.clone() for elm in edge_mat_list])
		bar.update(value=s + 1)
		sys.stdout.flush()
	return hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples
