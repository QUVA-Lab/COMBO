import time
import copy
import numpy as np

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference


GRAPH_SIZE_LIMIT = 2048 + 2


def np_kron(mat1, mat2):
	'''
	In order to check the function below 'def kronecker', numpy kron is used
	:param mat1: 2d torch.Tensor
	:param mat2: 2d torch.Tensor
	:return: kronecker product of mat1 and mat2
	'''
	np_mat1 = mat1.numpy()
	np_mat2 = mat2.numpy()
	return torch.from_numpy(np.kron(np_mat1, np_mat2))


def kronecker(mat1, mat2):
	'''
	kronecker product between 2 2D tensors
	:param mat1: 2d torch.Tensor
	:param mat2: 2d torch.Tensor
	:return: kronecker product of mat1 and mat2
	'''
	s1 = mat1.size()
	s2 = mat2.size()
	return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(s1 + s2)).permute([0, 2, 1, 3]).reshape(s1[0] * s2[0], s1[1] * s2[1])


def sort_partition(partition):
	'''
	given partition is ordered to have unique representation
	:param partition: list of subsets
	:return: each subset is represented as an ordered list, subsets are ordered according to their smallest elements
	'''
	lowest_ind_key_dict = {min(subset): sorted(subset) for subset in partition}
	sorted_partition = []
	for k in sorted(lowest_ind_key_dict.keys()):
		sorted_partition.append(lowest_ind_key_dict[k])
	return sorted_partition


def compute_unit_in_group(sorted_partition, categories):
	'''
	In order to convert between grouped variable and original ungrouped variable, units are necessary
	e.g) when C1, C2, C5 are grouped and C1, C2, C5 have 3, 4, 5 categories respectively,
	then (c1, c2, c5) in ungrouped value is equivalent to c1 x 4 x 5 + c2 x 5 + c3 in grouped value,
	here 4 x 5 is the unit for c1, 5 is unit for c2, 1 is unit for c3
	:param sorted_partition: list of subsets, elements in a subset are ordered, subsets are ordered by their smallest elements
	:param categories: 1d np.array
	:return: unit is given as the same order corresponding to sorted_partition
	'''
	unit_in_group = []
	for subset in sorted_partition:
		ind_units = list(np.flip(np.cumprod((categories[subset][1:][::-1])))) + [1]
		unit_in_group.append(ind_units)
	return unit_in_group


def compute_group_size(sorted_partition, categories):
	'''
	Return the size of the largest subgraph (a product of strong product)
	:param sorted_partition:
	:param categories:
	:return:
	'''
	complexity = sum([np.prod(categories[subset]) ** 3 for subset in sorted_partition])
	max_size = max([np.prod(categories[subset]) for subset in sorted_partition])
	return max_size


def group_input(input_data, sorted_partition, unit_in_group):
	'''

	:param input_data: 2D torch.Tensor, size(0) : number of data, size(1) : number of original variables
	:param sorted_partition: list of subsets, elements in each subset are ordered, subsets are ordered by their smallest elements
	:param unit_in_group: compute_unit_in_group(sorted_partition, categories)
	:return: 2D torch.Tensor, size(0) : number of data, size(1) : number of grouped variables
	'''
	grouped_input = input_data.new_zeros(input_data.size(0), len(sorted_partition))
	for g in range(len(sorted_partition)):
		for ind, unit in zip(sorted_partition[g], unit_in_group[g]):
			grouped_input[:, g] += input_data[:, ind] * unit
	return grouped_input


def ungroup_input(grouped_input, sorted_partition, unit_in_group):
	'''

	:param grouped_input: 2D torch.Tensor, size(0) : number of data, size(1) : number of grouped variables
	:param sorted_partition: list of subsets, elements in each subset are ordered, subsets are ordered by their smallest elements
	:param unit_in_group: compute_unit_in_group(sorted_partition, categories)
	:return: 2D torch.Tensor, size(0) : number of data, size(1) : number of original variables
	'''
	input_data = grouped_input.new_zeros(grouped_input.size(0), sum([len(subset) for subset in sorted_partition]))
	for g in range(len(sorted_partition)):
		subset = sorted_partition[g]
		unit = unit_in_group[g]
		elm_ind = 0
		remainder = grouped_input[:, g]
		while elm_ind < len(subset) - 1:
			input_data[:, subset[elm_ind]] = remainder // unit[elm_ind]
			remainder = remainder % unit[elm_ind]
			elm_ind += 1
		input_data[:, subset[-1]] = remainder
	return input_data


def strong_product(list_of_adjacency, beta, subset):
	'''
	Adjacency matrix of strong product of G1 with edge-weight beta1 and G2 with edge-weight beta2 is (beta1 x A1 + id) kron (beta2 x A2 + id) - id
	https://pdfs.semanticscholar.org/c534/af029958ba0882c04e136ceec99cbcba508f.pdf
	each subgraph has the same edge-weight within the graph
	:param list_of_adjacency: list of 2D tensors or adjacency matrices
	:param sorted_partition: list of subsets, elements in each subset are ordered, subsets are ordered by their smallest elements
	:return:
	'''
	elm = subset[0]
	grouped_adjacency_id_added = beta[elm] * list_of_adjacency[elm] + torch.diag(list_of_adjacency[elm].new_ones(list_of_adjacency[elm].size(0)))
	for ind in range(1, len(subset)):
		elm = subset[ind]
		mat1 = grouped_adjacency_id_added
		mat2 = beta[elm] * list_of_adjacency[elm] + torch.diag(list_of_adjacency[elm].new_ones(list_of_adjacency[elm].size(0)))
		grouped_adjacency_id_added = kronecker(mat1, mat2)
	return grouped_adjacency_id_added - torch.diag(grouped_adjacency_id_added.new_ones(grouped_adjacency_id_added.size(0)))


def neighbor_partitions(sorted_partition, ind):
	'''
	For given partition, relocation of ind into other subsets is considered as a neighbor
	:param sorted_partition:
	:param ind: the element which is relocated in the partition
	:return: list of neighbors
	'''
	n_subsets = len(sorted_partition)
	neighbors = []
	for i in range(n_subsets):
		if ind in sorted_partition[i]:
			break
	for s in range(n_subsets):
		if ind in sorted_partition[s]:
			neighbors.append(copy.deepcopy(sorted_partition))
		else:
			neighbor = copy.deepcopy(sorted_partition)
			neighbor[i].remove(ind)
			neighbor[s].append(ind)
			if len(neighbor[i]) == 0:
				neighbor.pop(i)
			neighbors.append(sort_partition(neighbor))
	return neighbors


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


def gibbs_partition(input_data, output_data, categories, list_of_adjacency, mean, log_amp, log_beta, log_noise_var, sorted_partition, ind):
	"""

	:param input_data:
	:param output_data:
	:param categories: list of the number of categories in each of K categorical variables
	:param log_amp:
	:param log_beta:
	:param sorted_partition: Partition of {1, ..., K}
	:return:
	"""
	# TODO : using beta and using log_beta are very different, consider slices coming from each method
	candidate_partitions = neighbor_partitions(sorted_partition, ind)
	unnormalized_log_posterior = []
	eigen_decompositions = {}
	model = GPRegression(kernel=None)
	model.mean.const_mean.data = mean
	model.likelihood.log_noise_var.data = log_noise_var
	for candidate in candidate_partitions:
		log_prior = partition_log_prior(sorted_partition=candidate, categories=categories)
		if np.isinf(log_prior):
			continue
		unit_in_group = compute_unit_in_group(sorted_partition=candidate, categories=categories)
		grouped_input_data = group_input(input_data=input_data, sorted_partition=candidate, unit_in_group=unit_in_group)
		fourier_freq_list = []
		fourier_basis_list = []
		for subset in candidate:
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
		kernel = DiffusionKernel(fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
		kernel.log_amp.data = log_amp
		model.kernel = kernel
		inference = Inference(train_data=(grouped_input_data, output_data), model=model)
		ll = -inference.negative_log_likelihood(hyper=model.param_to_vec())
		unnormalized_log_posterior.append(log_prior + ll)
	# Gumbel Max trick no need to calculate normalizing constant for multinomial random variables
	unnormalized_log_posterior = np.array(unnormalized_log_posterior)
	sampled_partition = candidate_partitions[np.argmax(-np.log(-np.log(np.random.uniform(0, 1, unnormalized_log_posterior.shape))) + unnormalized_log_posterior)]
	return sampled_partition


if __name__ == '__main__':
	n_vars = 100
	n_data = 100
	categories = np.random.randint(2, 6, n_vars)
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
		random_partition.append(inds[b:b+subset_size])
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
	print('%d variables' % n_vars)
	print(sorted_partition)
	for e in range(n_vars):
		print(e)
		gibbs_partition(input_data, output_data, categories, list_of_adjacency, mean, log_amp, log_beta, log_noise_var, sorted_partition, ind=e)
	print(time.time() - start_time)
