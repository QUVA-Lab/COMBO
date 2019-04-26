import itertools
import numpy as np
import torch

from GraphDecompositionBO.experiments.exp_utils import sample_init_points


def generate_function_on_highorderbinary(n_variables, highest_order, random_seed=None):
	init_seeds = np.random.RandomState(random_seed).randint(100, 100000, 4)
	choice_ratio = (np.random.RandomState(init_seeds[0]).rand(highest_order) + np.random.RandomState(init_seeds[1]).rand(highest_order)) / 2.0 * 0.3
	choice_shuffle = np.random.RandomState(init_seeds[2]).randint(100, 10000, highest_order)
	coef_seed = np.random.RandomState(init_seeds[3]).randint(100, 10000, highest_order)
	interaction_coef = []
	for o in range(1, highest_order + 1):
		combinations = list(itertools.combinations(range(n_variables), o))
		n_choices = len(combinations)
		n_chosen = int(n_choices * choice_ratio[o - 1])
		choice_inds = range(n_choices)
		np.random.RandomState(choice_shuffle[o - 1]).shuffle(choice_inds)
		chosen_interaction = [combinations[i] for i in choice_inds[:n_chosen]]
		chosen_coefficient = list(np.random.RandomState(coef_seed[o - 1]).uniform(-1.0, 1.0, n_chosen))
		interaction_coef.extend(zip(chosen_interaction, chosen_coefficient))
	return interaction_coef


def highorder_interaction_function(x, interaction_coef):
	'''
	:param x: np.array 2 dimensional array
	:param interaction: list of tuple, tuple of interactions and coefficient
	:return:
	'''
	output = 0
	for interaction, coef in interaction_coef:
		output += np.any(x[:, interaction], axis=1) * coef
	return output


class HighOrderBinary(object):
	def __init__(self, n_variables, highest_order, random_seed_pair=(None, None)):
		case_seed, init_seed = random_seed_pair
		self.suggested_init = torch.empty(0).long()
		self.suggested_init = torch.cat([self.suggested_init, sample_init_points([2] * n_variables, 20 - self.suggested_init.size(0), init_seed).long()], dim=0)
		self.n_variables = n_variables
		self.highest_order = highest_order
		self.interaction_coef = generate_function_on_highorderbinary(n_variables=n_variables, highest_order=highest_order, random_seed=case_seed)
		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
		for i in range(self.n_variables):
			adjmat = torch.diag(torch.ones(1), -1) + torch.diag(torch.ones(1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x):
		if x.dim() == 1:
			x = x.unsqueeze(0)
		assert x.size(1) == self.n_variables
		evaluation = torch.from_numpy(highorder_interaction_function(x.numpy(), self.interaction_coef).astype(np.float32))
		return evaluation