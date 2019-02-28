import os
import numpy as np
import pickle
from datetime import datetime
import progressbar

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunHistory

import torch

from CombinatorialBO.test_functions.experiment_configuration import generate_random_seed_pair_ising, generate_random_seed_pair_contamination, generate_random_seed_aerostruct, generate_random_seed_pair_travelplan, generate_random_seed_pestcontrol, generate_random_seed_pair_centroid
from CombinatorialBO.test_functions.binary_categorical import Ising1, Ising2, Contamination1, AeroStruct1, AeroStruct2, AeroStruct3
from CombinatorialBO.test_functions.multiple_categorical import PESTCONTROL_N_STAGES, CENTROID_GRID, CENTROID_N_EDGES, CENTROID_N_CHOICE, PESTCONTROL_N_CHOICE
from CombinatorialBO.test_functions.multiple_categorical import _pest_control_score, Centroid, edge_choice, partition, ising_dense
from CombinatorialBO.test_functions.discretized_continuous import Branin
from CombinatorialBO.test_functions.experiment_configuration import sample_init_points
from CombinatorialBO.baselines.utils import exp_dir, result_dir


EXP_DIR = exp_dir()
RESULT_DIR = result_dir()


def pest_control(n_eval, random_seed):
	name_tag = 'pestcontrol_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
	cs = ConfigurationSpace()
	for i in range(PESTCONTROL_N_STAGES):
		car_var = CategoricalHyperparameter('x' + str(i + 1).zfill(2), [str(elm) for elm in range(PESTCONTROL_N_CHOICE)], default_value='0')
		cs.add_hyperparameter(car_var)

	init_points_numpy = sample_init_points([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES, 20, random_seed).long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1).zfill(2): str(init_points_numpy[i][j]) for j in range(PESTCONTROL_N_STAGES)}))

	def evaluate(x):
		return _pest_control_score(np.array([int(x['x' + str(j + 1).zfill(2)]) for j in range(PESTCONTROL_N_STAGES)]))

	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	return optimum


def centroid(n_eval, random_seed_pair):
	name_tag = 'centroid_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
	cs = ConfigurationSpace()
	for i in range(CENTROID_N_EDGES):
		car_var = CategoricalHyperparameter('x' + str(i + 1).zfill(2), [str(elm) for elm in range(CENTROID_N_CHOICE)], default_value='0')
		cs.add_hyperparameter(car_var)

	init_points_numpy = sample_init_points([CENTROID_N_CHOICE] * CENTROID_N_EDGES, 20, random_seed_pair[1]).long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1).zfill(2): str(init_points_numpy[i][j]) for j in range(CENTROID_N_EDGES)}))

	evaluator = Centroid(random_seed_pair)
	interaction_list = evaluator.interaction_list
	covariance_list = evaluator.covariance_list
	partition_original_list = evaluator.partition_original_list

	def evaluate(x):
		interaction_mixed = edge_choice(np.array([int(x['x' + str(j + 1).zfill(2)]) for j in range(CENTROID_N_EDGES)]), interaction_list)
		partition_mixed = partition(interaction_mixed, CENTROID_GRID)
		kld_sum = 0
		for i in range(evaluator.n_ising_models):
			kld = ising_dense(interaction_sparsified=interaction_mixed, interaction_original=interaction_list[i],
			                  covariance=covariance_list[i], partition_sparsified=partition_mixed,
			                  partition_original=partition_original_list[i], grid_h=CENTROID_GRID[0])
			kld_sum += kld
		return kld_sum / float(evaluator.n_ising_models)

	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	return optimum


def branin(n_eval, random_seed):
	name_tag = 'branin_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
	evaluator = Branin()
	cs = ConfigurationSpace()
	for i in range(len(evaluator.n_vertices)):
		var = UniformIntegerHyperparameter('x' + str(i + 1), int(0), int(evaluator.n_vertices[i] - 1), default_value=25)
		cs.add_hyperparameter(var)

	init_points_numpy = sample_init_points(evaluator.n_vertices, 2, random_seed).long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1): int(init_points_numpy[i][j]) for j in range(len(evaluator.n_vertices))}))

	def evaluate(x):
		return evaluator.evaluate(torch.from_numpy(np.array([x['x' + str(j + 1)] for j in range(len(evaluator.n_vertices))]))).item()

	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
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
	smac_file = open(os.path.join(RESULT_DIR, problem + '_baseline_result_smac.pkl'), 'wb')
	pickle.dump({'mean': mean, 'std': std}, smac_file, protocol=2)
	smac_file.close()

	return mean, std


def evaluations_from_smac(smac):
	evaluations = smac.get_X_y()[1]
	n_evals = evaluations.size
	optimum = np.zeros((n_evals, ))
	for i in range(n_evals):
		optimum[i] = np.min(evaluations[:i+1])
	return evaluations, optimum


if __name__ == '__main__':
	mean, std = multiple_runs('branin')
	print(np.hstack([mean.reshape(-1, 1), std.reshape(-1, 1)]))
