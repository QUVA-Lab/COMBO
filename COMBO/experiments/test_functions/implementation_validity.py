import numpy as np

import torch

from COMBO.experiments.test_functions.binary_categorical import _contamination


def interaction_matlab2python(bocs_representation):
	assert bocs_representation.size(0) == bocs_representation.size(1)
	grid_size = int(bocs_representation.size(0) ** 0.5)
	horizontal_interaction = torch.zeros(grid_size, grid_size-1)
	vertical_interaction = torch.zeros(grid_size-1, grid_size)
	for i in range(bocs_representation.size(0)):
		r_i = i // grid_size
		c_i = i % grid_size
		for j in range(i + 1, bocs_representation.size(1)):
			r_j = j // grid_size
			c_j = j % grid_size
			if abs(r_i - r_j) + abs(c_i - c_j) > 1:
				assert bocs_representation[i, j] == 0
			elif abs(r_i - r_j) == 1:
				vertical_interaction[min(r_i, r_j), c_i] = bocs_representation[i, j]
			else:
				horizontal_interaction[r_i, min(c_i, c_j)] = bocs_representation[i, j]
	return horizontal_interaction, vertical_interaction


def interaction_python2matlab(horizontal_interaction, vertical_interaction):
	grid_size = horizontal_interaction.size(0)
	bocs_representation = torch.zeros(grid_size ** 2, grid_size ** 2)
	for i in range(bocs_representation.size(0)):
		r_i = i // grid_size
		c_i = i % grid_size
		for j in range(i + 1, bocs_representation.size(1)):
			r_j = j // grid_size
			c_j = j % grid_size
			if abs(r_i - r_j) + abs(c_i - c_j) > 1:
				assert bocs_representation[i, j] == 0
			elif abs(r_i - r_j) == 1:
				bocs_representation[i, j] = vertical_interaction[min(r_i, r_j), c_i]
			else:
				bocs_representation[i, j] = horizontal_interaction[r_i, min(c_i, c_j)]
	return bocs_representation


def matlab_matstr_reader(filename):
	matstr_file = open(filename, 'rt')
	matstr = matstr_file.read()
	rows = [elm.strip() for elm in matstr.split('\n')]
	result_list = []
	for row in rows:
		result_list.append([float(elm.strip()) for elm in row.split()])
	return torch.from_numpy(np.array(result_list))


def random_dynamics_from_csv():
	init_Z = np.genfromtxt('init_X.csv', delimiter=',')
	lambdas = np.genfromtxt('Lambdas.csv', delimiter=',')
	gammas = np.genfromtxt('Gammas.csv', delimiter=',')
	print(_contamination(np.hstack([np.ones(5), np.zeros(10), np.ones(10)]), np.ones(25), init_Z=init_Z, lambdas=lambdas, gammas=gammas, U=0.1, epsilon=0.05))


if __name__ == '__main__':
	grid_size = 4
	horizontal_interaction = torch.randn(grid_size, grid_size - 1)
	vertical_interaction = torch.randn(grid_size - 1, grid_size)
	matlab_representation = interaction_python2matlab(horizontal_interaction, vertical_interaction)
	python_representation = interaction_matlab2python(matlab_representation)
	print(torch.sum(horizontal_interaction != python_representation[0]))
	print(torch.sum(vertical_interaction != python_representation[1]))