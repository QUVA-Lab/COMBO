import os
import pickle
import numpy as np
import progressbar

from hyperopt import hp, base
from hyperopt import fmin, tpe

import torch

from COMBO.experiments.test_functions import generate_random_seed_pestcontrol, generate_random_seed_pair_centroid
from COMBO.experiments.test_functions import PESTCONTROL_N_STAGES, CENTROID_N_EDGES, CENTROID_GRID, CENTROID_N_CHOICE, PESTCONTROL_N_CHOICE
from COMBO.experiments.test_functions import _pest_control_score, Centroid, ising_dense, partition, edge_choice
from COMBO.experiments.test_functions import Branin
from COMBO.experiments.test_functions import sample_init_points
from COMBO.baselines.utils import result_dir


RESULT_DIR = result_dir()


def pest_control(n_eval, random_seed):
	space = [hp.choice('x' + str(i + 1).zfill(2), range(PESTCONTROL_N_CHOICE)) for i in range(PESTCONTROL_N_STAGES)]

	init_points = sample_init_points([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES, 20, random_seed).long().numpy()
	init_points = [{'x' + str(j + 1).zfill(2): init_points[i][j] for j in range(PESTCONTROL_N_STAGES)} for i in range(20)]

	def evaluate(x):
		return _pest_control_score(np.array(x))

	trials = base.Trials()
	fmin(evaluate, space, algo=tpe.suggest, max_evals=n_eval, points_to_evaluate=init_points, trials=trials)
	evaluations, optimum = evaluations_from_trials(trials)

	return optimum


def centroid(n_eval, random_seed_pair):
	space = [hp.choice('x' + str(i + 1).zfill(2), range(CENTROID_N_CHOICE)) for i in range(CENTROID_N_EDGES)]

	init_points = sample_init_points([CENTROID_N_CHOICE] * CENTROID_N_EDGES, 20, random_seed_pair[1]).long().numpy()
	init_points = [{'x' + str(j + 1).zfill(2): init_points[i][j] for j in range(CENTROID_N_EDGES)} for i in range(20)]

	evaluator = Centroid(random_seed_pair)

	def evaluate(x):
		interaction_mixed = edge_choice(np.array(x), evaluator.interaction_list)
		partition_mixed = partition(interaction_mixed, CENTROID_GRID)
		kld_sum = 0
		for i in range(evaluator.n_ising_models):
			kld = ising_dense(interaction_sparsified=interaction_mixed, interaction_original=evaluator.interaction_list[i],
			                  covariance=evaluator.covariance_list[i], partition_sparsified=partition_mixed,
			                  partition_original=evaluator.partition_original_list[i], grid_h=CENTROID_GRID[0])
			kld_sum += kld
		return kld_sum / float(evaluator.n_ising_models)

	trials = base.Trials()
	fmin(evaluate, space, algo=tpe.suggest, max_evals=n_eval, points_to_evaluate=init_points, trials=trials)
	evaluations, optimum = evaluations_from_trials(trials)

	return optimum


def branin(n_eval, random_seed):
	evaluator = Branin()
	space = [hp.randint('x' + str(i), evaluator.n_vertices[i]) for i in range(len(evaluator.n_vertices))]

	init_points = sample_init_points(evaluator.n_vertices, 1, random_seed).long().numpy()
	init_points = [{'x' + str(j): init_points[i][j] for j in range(len(evaluator.n_vertices))} for i in range(init_points.shape[0])]

	def evaluate(x):
		return evaluator.evaluate(torch.from_numpy(np.array(x))).item()

	trials = base.Trials()
	fmin(evaluate, space, algo=tpe.suggest, max_evals=n_eval, points_to_evaluate=init_points, trials=trials)
	evaluations, optimum = evaluations_from_trials(trials)

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
	tpe_file = open(os.path.join(RESULT_DIR, problem + '_baseline_result_tpe.pkl'), 'wb')
	pickle.dump({'mean': mean, 'std': std}, tpe_file)
	tpe_file.close()

	return np.mean(runs, axis=1), np.std(runs, axis=1)


def evaluations_from_trials(trials):
	n_trials = len(trials.trials)
	evaluations = np.array([trials.trials[i]['result']['loss'] for i in range(n_trials)])
	optimum = np.zeros((n_trials, ))
	for i in range(n_trials):
		optimum[i] = np.min(evaluations[:i+1])
	return evaluations, optimum


if __name__ == '__main__':
	mean, std = multiple_runs('branin')
	print(np.hstack([mean.reshape(-1, 1), std.reshape(-1, 1)]))

