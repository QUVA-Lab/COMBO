import os
import sys
import numpy as np
import pickle
from datetime import datetime
import progressbar

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace, Configuration
from ConfigSpace import CategoricalHyperparameter, UniformIntegerHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import torch

from GraphDecompositionBO.experiments.random_seed_config import generate_random_seed_pestcontrol, \
	generate_random_seed_pair_centroid, generate_random_seed_maxsat, generate_random_seed_pair_ising, \
	generate_random_seed_pair_contamination
from GraphDecompositionBO.experiments.test_functions.binary_categorical import ISING_N_EDGES, CONTAMINATION_N_STAGES
from GraphDecompositionBO.experiments.test_functions.binary_categorical import Ising, Contamination
from GraphDecompositionBO.experiments.test_functions.multiple_categorical import PESTCONTROL_N_STAGES, CENTROID_GRID, \
	CENTROID_N_EDGES, CENTROID_N_CHOICE, PESTCONTROL_N_CHOICE
from GraphDecompositionBO.experiments.test_functions.multiple_categorical import PestControl, Centroid, \
	edge_choice, partition, ising_dense
from GraphDecompositionBO.experiments.MaxSAT.maximum_satisfiability import MaxSAT28, MaxSAT43, MaxSAT60
from GraphDecompositionBO.experiments.exp_utils import sample_init_points
from GraphDecompositionBO.config import experiment_directory, SMAC_exp_dir


EXP_DIR = experiment_directory()
RESULT_DIR = SMAC_exp_dir()


def ising(n_eval, lamda, random_seed_pair):
	evaluator = Ising(random_seed_pair)
	evaluator.lamda = lamda
	name_tag = '_'.join(['ising',  ('%.2E' % lamda), datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")])
	cs = ConfigurationSpace()
	for i in range(ISING_N_EDGES):
		car_var = CategoricalHyperparameter('x' + str(i + 1).zfill(2), [str(elm) for elm in range(2)], default_value='0')
		cs.add_hyperparameter(car_var)

	init_points_numpy = evaluator.suggested_init.long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1).zfill(2): str(init_points_numpy[i][j]) for j in range(ISING_N_EDGES)}))

	def evaluate(x):
		x_tensor = torch.LongTensor([int(x['x' + str(j + 1).zfill(2)]) for j in range(ISING_N_EDGES)])
		return evaluator.evaluate(x_tensor).item()

	print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	print('Finished at ' + datetime.now().strftime("%H:%M:%S"))
	return optimum


def contamination(n_eval, lamda, random_seed_pair):
	evaluator = Contamination(random_seed_pair)
	evaluator.lamda = lamda
	name_tag = '_'.join(['contamination',  ('%.2E' % lamda), datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")])
	cs = ConfigurationSpace()
	for i in range(CONTAMINATION_N_STAGES):
		car_var = CategoricalHyperparameter('x' + str(i + 1).zfill(2), [str(elm) for elm in range(2)], default_value='0')
		cs.add_hyperparameter(car_var)

	init_points_numpy = evaluator.suggested_init.long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1).zfill(2): str(init_points_numpy[i][j]) for j in range(CONTAMINATION_N_STAGES)}))

	def evaluate(x):
		x_tensor = torch.LongTensor([int(x['x' + str(j + 1).zfill(2)]) for j in range(CONTAMINATION_N_STAGES)])
		return evaluator.evaluate(x_tensor).item()

	print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	print('Finished at ' + datetime.now().strftime("%H:%M:%S"))
	return optimum


def maxsat(n_eval, n_variables, random_seed):
	assert n_variables in [28, 43, 60]
	if n_variables == 28:
		evaluator = MaxSAT28(random_seed)
	elif n_variables == 43:
		evaluator = MaxSAT43(random_seed)
	elif n_variables == 60:
		evaluator = MaxSAT60(random_seed)
	name_tag = 'maxsat' + str(n_variables) + '_' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")
	cs = ConfigurationSpace()
	for i in range(n_variables):
		car_var = CategoricalHyperparameter('x' + str(i + 1).zfill(2), [str(elm) for elm in range(2)], default_value='0')
		cs.add_hyperparameter(car_var)
	init_points_numpy = evaluator.suggested_init.long().numpy()
	init_points = []
	for i in range(init_points_numpy.shape[0]):
		init_points.append(Configuration(cs, {'x' + str(j + 1).zfill(2): str(init_points_numpy[i][j]) for j in range(n_variables)}))

	def evaluate(x):
		x_tensor = torch.LongTensor([int(x['x' + str(j + 1).zfill(2)]) for j in range(n_variables)])
		return evaluator.evaluate(x_tensor).item()

	print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	print('Finished at ' + datetime.now().strftime("%H:%M:%S"))
	return optimum


def pest_control(n_eval, random_seed):
	evaluator = PestControl(random_seed)
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
		x_tensor = torch.LongTensor([int(x['x' + str(j + 1).zfill(2)]) for j in range(PESTCONTROL_N_STAGES)])
		return evaluator.evaluate(x_tensor).item()

	print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	print('Finished at ' + datetime.now().strftime("%H:%M:%S"))
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

	print('Began    at ' + datetime.now().strftime("%H:%M:%S"))
	scenario = Scenario({"run_obj": "quality", "runcount-limit": n_eval, "cs": cs, "deterministic": "true",
	                     'output_dir': os.path.join(EXP_DIR, name_tag)})
	smac = SMAC(scenario=scenario, tae_runner=evaluate, initial_configurations=init_points)
	smac.optimize()

	evaluations, optimum = evaluations_from_smac(smac)
	print('Finished at ' + datetime.now().strftime("%H:%M:%S"))
	return optimum


def multiple_runs(problem):
	print('Optimizing %s' % problem)
	if problem[:5] == 'ising':
		n_eval = 170
		lamda = float(problem.split('_')[1])
		random_seed_pairs = generate_random_seed_pair_ising()
		runs = None
		n_runs = sum([len(elm) for elm in random_seed_pairs.values()])
		bar = progressbar.ProgressBar(max_value=n_runs)
		bar_cnt = 0
		for i in range(len(random_seed_pairs.keys())):
			case_seed = sorted(random_seed_pairs.keys())[i]
			for j in range(len(random_seed_pairs[case_seed])):
				init_seed = sorted(random_seed_pairs[case_seed])[j]
				optimum = ising(n_eval, lamda, (case_seed, init_seed))
				bar_cnt += 1
				bar.update(bar_cnt)
				if runs is None:
					runs = optimum.reshape(-1, 1)
				else:
					runs = np.hstack([runs, optimum.reshape(-1, 1)])
	elif problem[:13] == 'contamination':
		n_eval = 270
		lamda = float(problem.split('_')[1])
		random_seed_pairs = generate_random_seed_pair_contamination()
		runs = None
		n_runs = sum([len(elm) for elm in random_seed_pairs.values()])
		bar = progressbar.ProgressBar(max_value=n_runs)
		bar_cnt = 0
		for i in range(len(random_seed_pairs.keys())):
			case_seed = sorted(random_seed_pairs.keys())[i]
			for j in range(len(random_seed_pairs[case_seed])):
				init_seed = sorted(random_seed_pairs[case_seed])[j]
				optimum = contamination(n_eval, lamda, (case_seed, init_seed))
				bar_cnt += 1
				bar.update(bar_cnt)
				if runs is None:
					runs = optimum.reshape(-1, 1)
				else:
					runs = np.hstack([runs, optimum.reshape(-1, 1)])
	elif problem == 'pestcontrol':
		n_eval = 320
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
		n_eval = 220
		random_seed_pairs = generate_random_seed_pair_centroid()
		runs = None
		n_runs = sum([len(elm) for elm in random_seed_pairs.values()])
		bar = progressbar.ProgressBar(max_value=n_runs)
		bar_cnt = 0
		for i in range(len(random_seed_pairs.keys())):
			case_seed = sorted(random_seed_pairs.keys())[i]
			for j in range(len(random_seed_pairs[case_seed])):
				init_seed = sorted(random_seed_pairs[case_seed])[j]
				optimum = centroid(n_eval, (case_seed, init_seed))
				bar_cnt += 1
				bar.update(bar_cnt)
				if runs is None:
					runs = optimum.reshape(-1, 1)
				else:
					runs = np.hstack([runs, optimum.reshape(-1, 1)])
	elif problem in ['maxsat28', 'maxsat43', 'maxsat60']:
		n_variables = int(problem[-2:])
		n_eval = 270
		random_seeds = sorted(generate_random_seed_maxsat())
		runs = None
		n_runs = 10

		bar = progressbar.ProgressBar(max_value=n_runs)
		bar_cnt = 0
		for i in range(n_runs):
			init_seed = random_seeds[i]
			optimum = maxsat(n_eval, n_variables, init_seed)
			bar_cnt += 1
			bar.update(bar_cnt)
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
	mean, std = multiple_runs(sys.argv[1])
	print(np.hstack([mean.reshape(-1, 1), std.reshape(-1, 1)]))
