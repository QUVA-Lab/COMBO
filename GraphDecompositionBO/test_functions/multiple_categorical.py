from operator import mul
import functools
import itertools

import numpy as np

import torch

from CombinatorialBO.test_functions.experiment_configuration import sample_init_points, generate_ising_interaction, generate_random_seed_pair_centroid
from CombinatorialBO.test_functions.binary_categorical import spin_covariance, partition, ising_dense

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25
CENTROID_N_CHOICE = 3
CENTROID_GRID = (4, 4)
CENTROID_N_EDGES = CENTROID_GRID[0] * (CENTROID_GRID[1] - 1) + (CENTROID_GRID[0] - 1) * CENTROID_GRID[1]


def edge_choice(x, interaction_list):
	edge_weight = np.zeros(x.shape)
	for i in range(len(interaction_list)):
		edge_weight[x == i] = np.hstack([interaction_list[i][0].reshape(-1), interaction_list[i][1].reshape(-1)])[x == i]
	grid_h, grid_w = CENTROID_GRID
	split_ind = grid_h * (grid_w - 1)
	return edge_weight[:split_ind].reshape((grid_h, grid_w - 1)), edge_weight[split_ind:].reshape((grid_h - 1, grid_w))


class Centroid(object):
	"""
	Ising Sparsification Problem with the simplest graph
	"""
	def __init__(self, random_seed_pair=(None, None)):
		self.n_vertices = [CENTROID_N_CHOICE] * CENTROID_N_EDGES
		self.suggested_init = torch.empty(0).long()
		self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1]).long()], dim=0)
		self.adjacency_mat = []
		self.fourier_coef = []
		self.fourier_basis = []
		self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
		for i in range(len(self.n_vertices)):
			n_v = self.n_vertices[i]
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_coef.append(eigval)
			self.fourier_basis.append(eigvec)
		self.interaction_list = []
		self.covariance_list = []
		self.partition_original_list = []
		self.n_ising_models = 3
		ising_seeds = np.random.RandomState(random_seed_pair[0]).randint(0, 10000, (self.n_ising_models,))
		for i in range(self.n_ising_models):
			interaction = generate_ising_interaction(CENTROID_GRID[0], CENTROID_GRID[1], ising_seeds[i])
			interaction = (interaction[0].numpy(), interaction[1].numpy())
			covariance, partition_original = spin_covariance(interaction, CENTROID_GRID)
			self.interaction_list.append(interaction)
			self.covariance_list.append(covariance)
			self.partition_original_list.append(partition_original)

	def evaluate(self, x):
		assert x.numel() == len(self.n_vertices)
		if x.dim() == 2:
			x = x.squeeze(0)
		interaction_mixed = edge_choice(x.numpy(), self.interaction_list)
		partition_mixed = partition(interaction_mixed, CENTROID_GRID)
		kld_sum = 0
		for i in range(self.n_ising_models):
			kld = ising_dense(interaction_sparsified=interaction_mixed, interaction_original=self.interaction_list[i],
			                  covariance=self.covariance_list[i], partition_sparsified=partition_mixed,
			                  partition_original=self.partition_original_list[i], grid_h=CENTROID_GRID[0])
			kld_sum += kld
		return float(kld_sum / float(self.n_ising_models)) * x.new_ones((1,)).float()


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
	if apply_control:
		next_pest_frac = (1.0 - control_rate) * curr_pest_frac
	else:
		next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
	return next_pest_frac


def _pest_control_score(x):
	U = 0.1
	n_stages = x.size
	n_simulations = 100

	init_pest_frac_alpha = 1.0
	init_pest_frac_beta = 30.0
	spread_alpha = 1.0
	spread_beta = 17.0 / 3.0

	control_alpha = 1.0
	control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
	tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
	control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
	# below two changes over stages according to x
	control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

	payed_price_sum = 0
	above_threshold = 0

	init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
	curr_pest_frac = init_pest_frac
	for i in range(n_stages):
		spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
		do_control = x[i] > 0
		if do_control:
			control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
			next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
			# torelance has been developed for pesticide type 1
			control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
			# you will get discount
			payed_price = control_price[x[i]] * (1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
		else:
			next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
			payed_price = 0
		payed_price_sum += payed_price
		above_threshold += np.mean(curr_pest_frac > U)
		curr_pest_frac = next_pest_frac

	return payed_price_sum + above_threshold


class PestControl(object):
	"""
	Ising Sparsification Problem with the simplest graph
	"""
	def __init__(self, random_seed=None):
		self.n_vertices = [PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES
		self.suggested_init = torch.empty(0).long()
		self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed).long()], dim=0)
		self.adjacency_mat = []
		self.fourier_coef = []
		self.fourier_basis = []
		self.random_seed_info = str(random_seed).zfill(4)
		for i in range(len(self.n_vertices)):
			n_v = self.n_vertices[i]
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_coef.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x):
		assert x.numel() == len(self.n_vertices)
		if x.dim() == 2:
			x = x.squeeze(0)
		evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy())
		return float(evaluation) * x.new_ones((1,)).float()


if __name__ == '__main__':
	pass
	# evaluator = PestControl(5355)
	# # x = np.random.RandomState(123).randint(0, 5, (PESTCONTROL_N_STAGES, ))
	# # print(_pest_control_score(x))
	# n_evals = 2000
	# for _ in range(10):
	# 	best_pest_control_loss = float('inf')
	# 	for i in range(n_evals):
	# 		if i < evaluator.suggested_init.size(0):
	# 			random_x = evaluator.suggested_init[i]
	# 		else:
	# 			random_x = torch.Tensor([np.random.randint(0, 5) for h in range(len(evaluator.n_vertices))]).long()
	# 		pest_control_loss = evaluator.evaluate(random_x).item()
	# 		if pest_control_loss < best_pest_control_loss:
	# 			best_pest_control_loss = pest_control_loss
	# 	print('With %d random search, the pest control objective(%d stages) is %f' % (n_evals, PESTCONTROL_N_STAGES, best_pest_control_loss))

	# for _ in range(10):
	# 	x = torch.from_numpy(np.random.RandomState(None).randint(0, 3, (ALTERATION_N_EDGES, )))
	# 	print(evaluator.evaluate(x))
	n_evals = 100
	for _ in range(10):
		evaluator = Centroid((9154, None))
		min_eval = float('inf')
		for i in range(n_evals):
			if i < 2:#evaluator.suggested_init.size(0):
				random_x = evaluator.suggested_init[i]
			else:
				random_x = torch.Tensor([np.random.randint(0, 5) for h in range(len(evaluator.n_vertices))]).long()
			evaluation = evaluator.evaluate(random_x).item()
			if evaluation < min_eval:
				min_eval = evaluation
		print('With %d random search, the ising alteration objective(%d edges) is %f' % (n_evals, CENTROID_N_EDGES, min_eval))

