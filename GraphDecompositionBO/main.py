import time
import argparse

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.sampler.sample_posterior import posterior_sampling

from GraphDecompositionBO.acquisition.acquisition_optimization import next_evaluation
from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement
from GraphDecompositionBO.acquisition.acquisition_marginalization import inference_sampling

from GraphDecompositionBO.utils import experiment_directory, model_data_filenames, load_model_data, displaying_and_logging

from GraphDecompositionBO.experiments.test_functions.experiment_configuration import generate_random_seed_pair_ising, \
	generate_random_seed_pair_contamination, generate_random_seed_pestcontrol, generate_random_seed_pair_centroid
from GraphDecompositionBO.experiments.test_functions.binary_categorical import Ising, Contamination
from GraphDecompositionBO.experiments.test_functions.multiple_categorical import PestControl, Centroid


def GOLD(objective=None, n_eval=200, path=None, parallel=False, **kwargs):
	"""
	
	:param objective: function to be optimized, it should provide information about search space by list of adjacency matrices 
	:param n_eval: 
	:return: 
	"""
	# GOLD continues from info given in 'path' or starts minimization of 'objective'
	assert (path is None) != (objective is None)
	acquisition_func = expected_improvement

	n_vertices = adj_mat_list = None
	eval_inputs = eval_outputs = log_beta = sorted_partition = None
	time_list = elapse_list = pred_mean_list = pred_std_list = pred_var_list = None

	if objective is not None:
		exp_dir = experiment_directory()
		objective_name = '_'.join([objective.__class__.__name__, objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none', ('%.1E' % objective.lamda) if hasattr(objective, 'lamda') else ''])
		model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(exp_dir=exp_dir, objective_name=objective_name)

		n_vertices = objective.n_vertices
		adj_mat_list = objective.adjacency_mat
		fourier_freq_list = objective.fourier_freq
		fourier_basis_list = objective.fourier_basis
		suggested_init = objective.suggested_init # suggested_init should be 2d tensor
		n_init = suggested_init.size(0)

		kernel = DiffusionKernel(fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
		surrogate_model = GPRegression(kernel=kernel)

		eval_inputs = suggested_init
		eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device)
		for i in range(eval_inputs.size(0)):
			eval_outputs[i] = objective.evaluate(eval_inputs[i])
		assert not torch.isnan(eval_outputs).any()
		log_beta = eval_outputs.new_zeros(eval_inputs.size(1))
		sorted_partition = [[m] for m in range(eval_inputs.size(1))]

		time_list = [time.time()] * n_init
		elapse_list = [0] * n_init
		pred_mean_list = [0] * n_init
		pred_std_list = [0] * n_init
		pred_var_list = [0] * n_init

		surrogate_model.init_param(eval_outputs)
		sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
		                                      log_beta, sorted_partition, n_sample=1, n_burn=2, n_thin=1)
		log_beta = sample_posterior[1][0]
		sorted_partition = sample_posterior[2][0]
	else:
		surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=experiment_directory())

	for _ in range(n_eval):

		reference = torch.min(eval_outputs, dim=0)[0].item()
		sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
		                                      log_beta, sorted_partition, n_sample=10, n_burn=0, n_thin=1)
		hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
		log_beta = log_beta_samples[-1]
		sorted_partition = partition_samples[-1]

		x_opt = eval_inputs[torch.argmin(eval_outputs)]
		inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
		                                       hyper_samples, partition_samples, freq_samples, basis_samples)
		suggestion = next_evaluation(x_opt, eval_inputs, inference_samples, partition_samples, edge_mat_samples, n_vertices,
		                             acquisition_func, reference, parallel)
		next_eval, pred_mean, pred_std, pred_var = suggestion

		eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
		eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
		assert not torch.isnan(eval_outputs).any()

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])
		pred_mean_list.append(float(pred_mean))
		pred_std_list.append(float(pred_std))
		pred_var_list.append(float(pred_var))

		displaying_and_logging(logfile_dir, eval_inputs, eval_outputs,
		                       pred_mean_list, pred_std_list, pred_var_list, time_list, elapse_list)
		print('Optimizing %s with regularization %.2E up to %4d visualization random seed : %s' %
		      (objective.__class__.__name__, objective.lamda if hasattr(objective, 'lamda') else 0, n_eval,
		       objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='GOLD : Graph Bayesian optimization with Learned Dependencies for Combinatorial Structures')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-p', '--path', dest='path')
	parser.add_argument('-o', '--objective', dest='objective')
	parser.add_argument('-s', '--random_seed_config', dest='random_seed_config', type=int, default=None)
	parser.add_argument('--parallel', dest='parallel', action='store_true', default=False)

	args = parser.parse_args()
	print(args)
	assert (args.path is None) != (args.objective is None)
	args_dict = vars(args)
	if args.objective == 'ising':
		assert 1 <= int(args.random_seed_config) <= 25
		random_seed_pair = generate_random_seed_pair_ising()
		random_seed_config = args.random_seed_config - 1
		case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
		init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
		args.objective = Ising(random_seed_pair=(case_seed, init_seed))
	elif args.objective == 'contamination':
		assert 1 <= int(args.random_seed_config) <= 25
		random_seed_pair = generate_random_seed_pair_contamination()
		random_seed_config = args.random_seed_config - 1
		case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
		init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
		args.objective = Contamination(random_seed_pair=(case_seed, init_seed))
	elif args.objective == 'pestcontrol':
		assert 1 <= int(args.random_seed_config) <= 25
		random_seed = sorted(generate_random_seed_pestcontrol())[args.random_seed_config - 1]
		args.objective = PestControl(random_seed=random_seed)
	elif args.objective == 'centroid':
		assert 1 <= int(args.random_seed_config) <= 25
		random_seed_pair = generate_random_seed_pair_centroid()
		random_seed_config = args.random_seed_config - 1
		case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
		init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
		print(case_seed, init_seed)
		args.objective = Centroid(random_seed_pair=(case_seed, init_seed))
	else:
		raise NotImplementedError
	GOLD(**vars(args))