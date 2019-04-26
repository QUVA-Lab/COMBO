import os
import pickle
import numpy as np
import progressbar

from simanneal import Annealer

import torch

from GraphDecompositionBO.experiments.test_functions import generate_random_seed_pestcontrol, generate_random_seed_pair_centroid
from GraphDecompositionBO.experiments.test_functions import PESTCONTROL_N_STAGES, CENTROID_N_EDGES, CENTROID_N_CHOICE, PESTCONTROL_N_CHOICE
from GraphDecompositionBO.experiments.test_functions import _pest_control_score, Centroid
from GraphDecompositionBO.experiments.test_functions import Branin
from GraphDecompositionBO.experiments.test_functions import sample_init_points
from GraphDecompositionBO.baselines.utils import result_dir


RESULT_DIR = result_dir()


class SA_PestControl(Annealer):

	def __init__(self, initial_state):
		super(SA_PestControl, self).__init__(initial_state)
		self.eval_history = []

	def move(self):
		self.state[np.random.randint(0, PESTCONTROL_N_STAGES)] = np.random.randint(0, PESTCONTROL_N_CHOICE)

	def energy(self):
		evaluation = _pest_control_score(np.array(self.state))
		self.eval_history.append(evaluation)
		return evaluation


class SA_Centroid(Annealer):

	def __init__(self, initial_state, random_seed_pair):
		super(SA_Centroid, self).__init__(initial_state)
		self.eval_history = []
		self.evaluator = Centroid(random_seed_pair)

	def move(self):
		self.state[np.random.randint(0, CENTROID_N_EDGES)] = np.random.randint(0, CENTROID_N_CHOICE)

	def energy(self):
		evaluation = self.evaluator.evaluate(torch.from_numpy(np.array(self.state))).item()
		self.eval_history.append(evaluation)
		return evaluation


class SA_Branin(Annealer):

	def __init__(self, initial_state):
		super(SA_Branin, self).__init__(initial_state)
		self.eval_history = []
		self.evaluator = Branin()

	def move(self):
		which_dim = np.random.randint(0, len(self.evaluator.n_vertices))
		if self.state[which_dim] == 0:
			self.state[which_dim] = 1
		elif self.state[which_dim] == self.evaluator.n_vertices[which_dim] - 1:
			self.state[which_dim] -= 1
		else:
			self.state[which_dim] += int(np.random.randint(0, 2) * 2 - 1)

	def energy(self):
		evaluation = self.evaluator.evaluate(torch.from_numpy(np.array(self.state))).item()
		self.eval_history.append(evaluation)
		return evaluation


def pest_control(n_eval, random_seed):
	init_points = sample_init_points([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES, 20, random_seed).long().numpy()
	evaluations = []
	for i in range(init_points.shape[0]):
		evaluations.append(_pest_control_score(init_points[i]))
	initial_state = list(init_points[np.argmin(evaluations)])
	sa_runner = SA_PestControl(initial_state)
	# configuration equivalent to that of BOCS' SA implementation
	sa_runner.Tmax = 1.0
	sa_runner.Tmin = 0.8 ** n_eval
	sa_runner.steps = n_eval
	sa_runner.anneal()

	evaluations, optimum = evaluations_from_annealer(sa_runner)
	return optimum


def centroid(n_eval, random_seed_pair):
	evaluator = Centroid(random_seed_pair)
	init_points = sample_init_points([CENTROID_N_CHOICE] * CENTROID_N_EDGES, 20, random_seed_pair[1]).long().numpy()

	def evaluate(x):
		return evaluator.evaluate(torch.from_numpy(x)).item()

	evaluations = []
	for i in range(init_points.shape[0]):
		evaluations.append(evaluate(init_points[i]))
	initial_state = list(init_points[np.argmin(evaluations)])
	sa_runner = SA_Centroid(initial_state, random_seed_pair)
	# configuration equivalent to that of BOCS' SA implementation
	sa_runner.Tmax = 1.0
	sa_runner.Tmin = 0.8 ** n_eval
	sa_runner.steps = n_eval
	sa_runner.anneal()

	evaluations, optimum = evaluations_from_annealer(sa_runner)
	return optimum


def branin(n_eval, random_seed):
	evaluator = Branin()
	init_points = sample_init_points(evaluator.n_vertices, 1, random_seed).long().numpy()
	evaluations = []
	for i in range(init_points.shape[0]):
		evaluations.append(evaluator.evaluate(torch.from_numpy(init_points[i])).item())
	initial_state = list(init_points[np.argmin(evaluations)])
	sa_runner = SA_Branin(initial_state)
	# configuration equivalent to that of BOCS' SA implementation
	sa_runner.Tmax = 1.0
	sa_runner.Tmin = 0.8 ** n_eval
	sa_runner.steps = n_eval
	sa_runner.anneal()

	evaluations, optimum = evaluations_from_annealer(sa_runner)
	return optimum


def multiple_runs(problem):
	print('Optimizing %s' % problem)
	if problem == 'pestcontrol':
		n_eval = 250
		random_seeds = sorted(generate_random_seed_pestcontrol())
		runs = None
		bar = progressbar.ProgressBar(max_value=len(random_seeds))
		for i in range(len(random_seeds)):
			optimum = pest_control(n_eval, random_seeds[i])
			bar.update(i)
			if runs is None:
				runs = optimum.reshape(-1, 1)
			else:
				runs = np.hstack([runs, optimum.reshape(-1, 1)])
	elif problem == 'centroid':
		n_eval = 200
		random_seed_pairs = generate_random_seed_pair_centroid()
		runs = None
		n_runs = sum([len(elm) for elm in random_seed_pairs.values()])
		bar = progressbar.ProgressBar(max_value=n_runs)
		bar_cnt = 0
		for i in range(len(random_seed_pairs.keys())):
			case_seed = sorted(random_seed_pairs.keys())[i]
			for j in range(len(random_seed_pairs[case_seed])):
				init_seed = random_seed_pairs[case_seed][j]
				optimum = centroid(n_eval, (case_seed, init_seed))
				bar_cnt += 1
				bar.update(bar_cnt)
				if runs is None:
					runs = optimum.reshape(-1, 1)
				else:
					runs = np.hstack([runs, optimum.reshape(-1, 1)])
	elif problem == 'branin':
		n_eval = 100
		random_seeds = range(25)
		runs = None
		bar = progressbar.ProgressBar(max_value=len(random_seeds))
		for i in range(len(random_seeds)):
			optimum = branin(n_eval, random_seeds[i])
			bar.update(i)
			if runs is None:
				runs = optimum.reshape(-1, 1)
			else:
				runs = np.hstack([runs, optimum.reshape(-1, 1)])
	else:
		raise NotImplementedError
	print('\nOptimized %s' % problem)

	mean = np.mean(runs, axis=1)
	std = np.std(runs, axis=1)
	simulatedannealing_file = open(os.path.join(RESULT_DIR, problem + '_baseline_result_simulatedannealing.pkl'), 'wb')
	pickle.dump({'mean': mean, 'std': std}, simulatedannealing_file)
	simulatedannealing_file.close()

	return np.mean(runs, axis=1), np.std(runs, axis=1)


def evaluations_from_annealer(annealer):
	n_trials = len(annealer.eval_history)
	evaluations = np.array(annealer.eval_history)
	optimum = np.zeros((n_trials, ))
	for i in range(n_trials):
		optimum[i] = np.min(evaluations[:i+1])
	return evaluations, optimum


if __name__ == '__main__':
	mean, std = multiple_runs('branin')
	print(np.hstack([mean.reshape(-1, 1), std.reshape(-1, 1)]))