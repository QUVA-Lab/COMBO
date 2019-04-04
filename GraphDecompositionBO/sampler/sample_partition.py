import time
import numpy as np

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference
from GraphDecompositionBO.sampler.grouping_utils import sort_partition, compute_unit_in_group, compute_group_size, group_input, ungroup_input, strong_product, neighbor_partitions


GRAPH_SIZE_LIMIT = 1024 + 2


def partition_log_prior(sorted_partition, categories):
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


def gibbs_partition(input_data, output_data, categories, list_of_adjacency, mean, log_amp, log_beta, log_noise_var, sorted_partition, fourier_freq_list, fourier_basis_list, ind):
	"""
	Gibbs sampling from a given partition by relocating ind
	:param input_data:
	:param output_data:
	:param categories: list of the number of categories in each of K categorical variables
	:param log_amp:
	:param log_beta:
	:param sorted_partition: Partition of {1, ..., K}
	:param fourier_freq_list: frequencies for subsets in sorted_partition
	:param fourier_basis_list: basis for subsets in sorted_partition
	:param ind: the index of the variable to be relocated in the sorted_partition
	:param freq_basis:
	:return:
	"""
	# TODO : using beta and using log_beta are very different, consider slices for each method
	candidate_sorted_partitions = neighbor_partitions(sorted_partition, ind)
	unnormalized_log_posterior = []
	# TODO : eigen_decompositions itself can be given if all betas are sampled first and all partitions are sampled afterward
	# TODO : if beta and partition are sampled alternatively, but still passing eigen_decomposition may be passed with some tuning.
	#        As long as subset does not contain ind, it is reusable.
	#        check below try and except, in which checking that ind belongs a subset should be checked
	eigen_decompositions = {}
	for subset, fourier_freq, fourier_basis in zip(sorted_partition, fourier_freq_list, fourier_basis_list):
		eigen_decompositions[tuple(subset)] = (fourier_freq, fourier_basis)
	model = GPRegression(kernel=None)
	model.mean.const_mean.data = mean
	model.likelihood.log_noise_var.data = log_noise_var
	inference = Inference(train_data=(None, output_data), model=model)
	for cand_sorted_partition in candidate_sorted_partitions:
		log_prior = partition_log_prior(sorted_partition=cand_sorted_partition, categories=categories)
		if np.isinf(log_prior):
			unnormalized_log_posterior.append(log_prior)
		else:
			unit_in_group = compute_unit_in_group(sorted_partition=cand_sorted_partition, categories=categories)
			grouped_input_data = group_input(input_data=input_data, sorted_partition=cand_sorted_partition, unit_in_group=unit_in_group)
			fourier_freq_list = []
			fourier_basis_list = []
			for subset in cand_sorted_partition:
				try:
					fourier_freq, fourier_basis = eigen_decompositions[tuple(subset)]
				except KeyError:
					adj_mat = strong_product(list_of_adjacency=list_of_adjacency, beta=torch.exp(log_beta), subset=subset)
					deg_mat = torch.diag(torch.sum(adj_mat, dim=0))
					laplacian = deg_mat - adj_mat
					fourier_freq, fourier_basis = torch.symeig(laplacian, eigenvectors=True)
					eigen_decompositions[tuple(subset)] = (fourier_freq, fourier_basis)
				fourier_freq_list.append(fourier_freq)
				fourier_basis_list.append(fourier_basis)
			inference.train_x = grouped_input_data
			kernel = DiffusionKernel(fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
			kernel.log_amp.data = log_amp
			model.kernel = kernel
			ll = -inference.negative_log_likelihood(hyper=model.param_to_vec())
			unnormalized_log_posterior.append(log_prior + ll)
	# Gumbel Max trick : No need to calculate the normalizing constant for multinomial random variables
	unnormalized_log_posterior = np.array(unnormalized_log_posterior)
	gumbel_max_rv = np.argmax(-np.log(-np.log(np.random.uniform(0, 1, unnormalized_log_posterior.shape))) + unnormalized_log_posterior)
	sampled_sorted_partition = candidate_sorted_partitions[gumbel_max_rv]

	fourier_freq_list = []
	fourier_basis_list = []
	for subset in sampled_sorted_partition:
		fourier_freq, fourier_basis = eigen_decompositions[tuple(subset)]
		fourier_freq_list.append(fourier_freq)
		fourier_basis_list.append(fourier_basis)
	return sampled_sorted_partition, fourier_freq_list, fourier_basis_list


if __name__ == '__main__':
	import progressbar
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
	grouped_input = group_input(input_data, sorted_partition, unit_in_group)
	input_data_re = ungroup_input(grouped_input, sorted_partition, unit_in_group)
	mean = torch.mean(output_data, dim=0)
	amp = torch.std(output_data, dim=0)
	log_amp = torch.log(amp)
	log_noise_var = torch.log(amp / 1000.)
	log_beta = torch.randn(n_vars)

	start_time = time.time()
	fourier_freq_list = []
	fourier_basis_list = []
	for subset in sorted_partition:
		adj_mat = strong_product(list_of_adjacency=list_of_adjacency, beta=torch.exp(log_beta), subset=subset)
		deg_mat = torch.diag(torch.sum(adj_mat, dim=0))
		laplacian = deg_mat - adj_mat
		fourier_freq, fourier_basis = torch.symeig(laplacian, eigenvectors=True)
		fourier_freq_list.append(fourier_freq)
		fourier_basis_list.append(fourier_basis)
	print('init elapsed time', time.time() - start_time)

	start_time = time.time()
	print('%d variables' % n_vars)
	print(len(sorted_partition))
	print(sorted([len(elm) for elm in sorted_partition]))
	bar = progressbar.ProgressBar(max_value=n_vars)
	for e in range(n_vars):
		bar.update(e)
		sorted_partition, fourier_freq_list, fourier_basis_list = gibbs_partition(input_data, output_data, categories, list_of_adjacency, mean, log_amp, log_beta, log_noise_var, sorted_partition, fourier_freq_list, fourier_basis_list, ind=e, freq_basis=True)
	print(time.time() - start_time)
	print(sorted([len(elm) for elm in sorted_partition]))
