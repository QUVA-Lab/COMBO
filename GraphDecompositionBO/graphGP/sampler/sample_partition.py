import numpy as np

import torch

from GraphDecompositionBO.graphGP.inference.inference import Inference
from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_unit_in_group, group_input, strong_product, neighbor_partitions
from GraphDecompositionBO.graphGP.sampler.priors import log_prior_partition


def gibbs_partition(model, input_data, output_data, categories, list_of_adjacency, log_beta,
					sorted_partition, fourier_freq_list, fourier_basis_list, ind):
	"""
	Gibbs sampling from a given partition by relocating 'ind' in 'sorted_partition'
	Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
	:param model:
	:param input_data:
	:param output_data:
	:param categories: list of the number of categories in each of K categorical variables
	:param list_of_adjacency:
	:param log_beta:
	:param sorted_partition: Partition of {1, ..., K}
	:param fourier_freq_list: frequencies for subsets in sorted_partition
	:param fourier_basis_list: basis for subsets in sorted_partition
	:param ind: the index of the variable to be relocated in the sorted_partition
	:return:
	"""
	candidate_sorted_partitions = neighbor_partitions(sorted_partition, ind)
	unnormalized_log_posterior = []
	# TODO : eigen_decompositions itself can be given if all betas are sampled first and all partitions are sampled afterward
	# TODO : if beta and partition are sampled alternatively, but still passing eigen_decomposition may be passed with some tuning.
	#        As long as subset does not contain ind, it is reusable.
	#        check below try and except, in which checking that ind belongs a subset should be checked
	eigen_decompositions = {}
	for subset, fourier_freq, fourier_basis in zip(sorted_partition, fourier_freq_list, fourier_basis_list):
		eigen_decompositions[tuple(subset)] = (fourier_freq, fourier_basis)
	inference = Inference(train_data=(None, output_data), model=model)
	for cand_sorted_partition in candidate_sorted_partitions:
		log_prior = log_prior_partition(sorted_partition=cand_sorted_partition, categories=categories)
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
			model.kernel.fourier_freq_list = fourier_freq_list
			model.kernel.fourier_basis_list = fourier_basis_list
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
	pass
	import progressbar
	import time
	from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
	from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
	from GraphDecompositionBO.graphGP.sampler.tool_partition import sort_partition, ungroup_input
	n_vars_ = 100
	n_data_ = 60
	categories_ = np.random.randint(5, 6, n_vars_)
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
	while True:
		random_partition_ = []
		b_ = 0
		while b_ < n_vars_:
			subset_size_ = np.random.randint(1, 5)
			random_partition_.append(inds_[b_:b_ + subset_size_])
			b_ += subset_size_
		sorted_partition_ = sort_partition(random_partition_)
		print(sorted_partition_)
		if np.isinf(log_prior_partition(sorted_partition_, categories_)):
			print('Infeasible partition')
		else:
			print('Feasible partition')
			break
	sorted_partition_ = sort_partition(random_partition_)
	unit_in_group_ = compute_unit_in_group(sorted_partition_, categories_)
	grouped_input_ = group_input(input_data_, sorted_partition_, unit_in_group_)
	input_data_re_ = ungroup_input(grouped_input_, sorted_partition_, unit_in_group_)
	amp_ = torch.std(output_data_, dim=0)
	log_beta_ = torch.randn(n_vars_)
	model_ = GPRegression(kernel=DiffusionKernel(fourier_freq_list=[], fourier_basis_list=[]))
	model_.kernel.log_amp.data = torch.log(amp_)
	model_.mean.const_mean.data = torch.mean(output_data_, dim=0)
	model_.likelihood.log_noise_var.data = torch.log(amp_ / 1000.)

	start_time_ = time.time()
	fourier_freq_list_ = []
	fourier_basis_list_ = []
	for subset_ in sorted_partition_:
		adj_mat_ = strong_product(list_of_adjacency=list_of_adjacency_, beta=torch.exp(log_beta_), subset=subset_)
		deg_mat_ = torch.diag(torch.sum(adj_mat_, dim=0))
		laplacian_ = deg_mat_ - adj_mat_
		fourier_freq_, fourier_basis_ = torch.symeig(laplacian_, eigenvectors=True)
		fourier_freq_list_.append(fourier_freq_)
		fourier_basis_list_.append(fourier_basis_)
	print('init elapsed time', time.time() - start_time_)

	start_time_ = time.time()
	print('%d variables' % n_vars_)
	print(len(sorted_partition_))
	print(sorted([len(elm_) for elm_ in sorted_partition_]))
	bar_ = progressbar.ProgressBar(max_value=n_vars_)
	for e_ in range(n_vars_):
		bar_.update(e_)
		sorted_partition_, fourier_freq_list_, fourier_basis_list_ = gibbs_partition(model_, input_data_, output_data_, categories_, list_of_adjacency_, log_beta_, sorted_partition_, fourier_freq_list_, fourier_basis_list_, ind=e_)
	print('\n%f' % (time.time() - start_time_))