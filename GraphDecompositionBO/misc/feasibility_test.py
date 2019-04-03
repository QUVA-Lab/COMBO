import time
import sys
import numpy as np

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference


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


def strong_product_of_adjacency(list_of_adjmats, partition):
	'''

	:param list_of_adjmats: list of tensors
	:param ind_groups: list of list of inds
	:return:
	'''
	sorted_partition = sort_partition(partition)
	list_of_grouped_adjmats = []
	for subset in sorted_partition:
		grouped_adjmats = list_of_adjmats[subset[0]]
		for i in range(1, len(subset)):
			grouped_adjmats = kronecker(grouped_adjmats, list_of_adjmats[subset[i]])
		list_of_grouped_adjmats.append(grouped_adjmats)
	return list_of_grouped_adjmats


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


def group_input(input_data, sorted_partition, unit_in_group):
	'''

	:param input_data: 2D torch.Tensor, size(0) : number of data, size(1) : number of original variables
	:param sorted_partition: list of subsets, elements in a subset are ordered, subsets are ordered by their smallest elements
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
	:param sorted_partition: list of subsets, elements in a subset are ordered, subsets are ordered by their smallest elements
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


def feasibility(input_data, output_data, log_amp, log_beta, categories, grouping):
	"""

	:param input_data:
	:param output_data:
	:param log_amp:
	:param log_beta:
	:param categories: list of the number of categories in each of K categorical variables
	:param grouping: Partition of {1, ..., K}
	:return:
	"""
	pass


if __name__ == '__main__':
	n_vars = 10
	n_data = 3
	categories = np.random.randint(2, 6, n_vars)
	random_input_data = torch.zeros(n_data, n_vars)
	for i in range(n_vars):
		random_input_data[:, i] = torch.randint(0, categories[i], (n_data,))
	inds = range(n_vars)
	np.random.shuffle(inds)
	i = 0
	random_partition = []
	while i < n_vars:
		subset_size = np.random.poisson(2) + 1
		random_partition.append(inds[i:i+subset_size])
		i += subset_size
	print(random_partition)
	sorted_partition = sort_partition(random_partition)
	unit_in_group = compute_unit_in_group(sorted_partition, categories)
	print(sorted_partition)
	print(unit_in_group)
	grouped_input = group_input(random_input_data, sorted_partition, unit_in_group)
	random_input_data_re = ungroup_input(grouped_input, sorted_partition, unit_in_group)
	print(torch.sum((random_input_data - random_input_data_re) ** 2))