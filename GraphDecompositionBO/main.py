import time
import sys
import argparse

import torch
import torch.multiprocessing as multiprocessing

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference

from GraphDecompositionBO.acquisition.acquisition_utils import deepcopy_inference, next_evaluation
from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement

from GraphDecompositionBO.experiment_utils import experiment_directory, model_data_filenames, load_model_data, displaying_and_logging

from GraphDecompositionBO.test_functions.experiment_configuration import generate_random_seed_pair_ising, generate_random_seed_pair_contamination, generate_random_seed_aerostruct, generate_random_seed_pair_travelplan, generate_random_seed_pestcontrol, generate_random_seed_pair_centroid
from GraphDecompositionBO.test_functions.discretized_continuous import Branin, Hartmann6
from GraphDecompositionBO.test_functions.binary_categorical import Ising1, Ising2, Contamination1, AeroStruct1, AeroStruct2, AeroStruct3
from GraphDecompositionBO.test_functions.multiple_categorical import PestControl, Centroid


def GRASB(objective=None, n_eval=200, path=None, parallel=False, **kwargs):
	"""
	
	:param objective: function to be optimized, it should provide information about search space by list of adjacency matrices 
	:param n_eval: 
	:return: 
	"""
	# GRASB continues from info given in 'path' or starts minimization of 'objective'
	assert (path is None) != (objective is None)

	if objective is not None:
		exp_dir = experiment_directory()
		objective_name = '_'.join([objective.__class__.__name__, objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none', ('%.1E' % objective.lamda) if hasattr(objective, 'lamda') else ''])
		model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(exp_dir=exp_dir, objective_name=objective_name)

		adjacency_mat_list = objective.adjacency_mat
		fourier_coef_list = objective.fourier_coef
		fourier_basis_list = objective.fourier_basis
		suggested_init = objective.suggested_init # suggested_init should be 2d tensor
		n_BO_init = suggested_init.size(0)

		surrogate_model = GPRegression(kernel=DiffusionKernel(fourier_coef_list=fourier_coef_list, fourier_basis_list=fourier_basis_list, ard=True))

		eval_inputs = suggested_init
		eval_outputs = torch.zeros(eval_inputs.size(0), 1, device=eval_inputs.device)
		for i in range(eval_inputs.size(0)):
			eval_outputs[i] = objective.evaluate(eval_inputs[i])
		assert not torch.isnan(eval_outputs).any()

		time_list = [time.time()] * n_BO_init
		elapse_list = [0] * n_BO_init
		pred_mean_list = [0] * n_BO_init
		pred_std_list = [0] * n_BO_init
		pred_var_list = [0] * n_BO_init

		print(eval_inputs)
		print(eval_outputs)

		inference = Inference((eval_inputs, eval_outputs), surrogate_model)
		inference.init_parameters()
		if 'TravelPlan' in objective.__class__.__name__:
			surrogate_model.kernel.init_beta(0.1)
			surrogate_model.kernel.beta_min = 1e-8
			surrogate_model.kernel.beta_max = 1.0
		inference.sampling(n_sample=1, n_burnin=99, n_thin=1)
	else:
		surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=experiment_directory())

	for _ in range(n_eval):
		inference = Inference((eval_inputs, eval_outputs), surrogate_model)

		reference = torch.min(eval_outputs, dim=0)[0].item()
		graphGP_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
		inferences = deepcopy_inference(inference, graphGP_hyper_params)

		x_best = eval_inputs[torch.argmin(eval_outputs)]
		next_eval, pred_mean, pred_std, pred_var = next_evaluation(x_best, adjacency_mat_list, inferences, acquisition_func=expected_improvement, reference=reference, verbose=False, parallel=parallel)

		eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
		eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
		assert not torch.isnan(eval_outputs).any()

		time_list.append(time.time())
		elapse_list.append(time_list[-1] - time_list[-2])
		pred_mean_list.append(float(pred_mean))
		pred_std_list.append(float(pred_std))
		pred_var_list.append(float(pred_var))

		displaying_and_logging(logfile_dir=logfile_dir, eval_inputs=eval_inputs, eval_outputs=eval_outputs,
		                       pred_mean_list=pred_mean_list, pred_std_list=pred_std_list, pred_var_list=pred_var_list,
		                       time_list=time_list, elapse_list=elapse_list)
		print('Optimizing %s with regularization %.2E up to %4d visualization random seed : %s' %
		      (objective.__class__.__name__, objective.lamda if hasattr(objective, 'lamda') else 0, n_eval,
		       objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none'))


if __name__ == '__main__':
	if sys.version_info.major == 3:
		multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser(description='GRASB : GRAph Signal Bayesian optimization')
	parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=1)
	parser.add_argument('-p', '--path', dest='path')
	parser.add_argument('-o', '--objective', dest='objective')
	parser.add_argument('-s', '--random_seed_config', dest='random_seed_config', type=int, default=None)
	parser.add_argument('--parallel', dest='parallel', action='store_true', default=False)

	args = parser.parse_args()
	print(args)
	assert (args.path is None) != (args.objective is None)
	args_dict = vars(args)
	if args.objective is not None:
		if args.objective == 'branin':
			args.objective = Branin()
		elif args.objective == 'hartmann6':
			args.objective = Hartmann6()
		elif args.objective.split('_')[0] == 'ising1':
			assert 1 <= int(args.random_seed_config) <= 25
			random_seed_pair = generate_random_seed_pair_ising()
			random_seed_config = args.random_seed_config - 1
			case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
			init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
			args.objective = Ising1(lamda=float(args.objective.split('_')[1]), random_seed_pair=(case_seed, init_seed))
		elif args.objective.split('_')[0] == 'ising2':
			assert 1 <= int(args.random_seed_config) <= 25
			random_seed_pair = generate_random_seed_pair_ising()
			random_seed_config = args.random_seed_config - 1
			case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
			init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
			args.objective = Ising2(lamda=float(args.objective.split('_')[1]), random_seed_pair=(case_seed, init_seed))
		elif args.objective.split('_')[0] == 'contamination1':
			assert 1 <= int(args.random_seed_config) <= 25
			random_seed_pair = generate_random_seed_pair_contamination()
			random_seed_config = args.random_seed_config - 1
			case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
			init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
			args.objective = Contamination1(lamda=float(args.objective.split('_')[1]), random_seed_pair=(case_seed, init_seed))
		elif args.objective.split('_')[0] == 'aerostruct1':
			assert 1 <= int(args.random_seed_config) <= 10
			random_seed = sorted(generate_random_seed_aerostruct())[args.random_seed_config - 1]
			args.objective = AeroStruct1(lamda=float(args.objective.split('_')[1]), random_seed=random_seed)
		elif args.objective.split('_')[0] == 'aerostruct2':
			assert 1 <= int(args.random_seed_config) <= 10
			random_seed = sorted(generate_random_seed_aerostruct())[args.random_seed_config - 1]
			args.objective = AeroStruct2(lamda=float(args.objective.split('_')[1]), random_seed=random_seed)
		elif args.objective.split('_')[0] == 'aerostruct3':
			assert 1 <= int(args.random_seed_config) <= 10
			random_seed = sorted(generate_random_seed_aerostruct())[args.random_seed_config - 1]
			args.objective = AeroStruct3(lamda=float(args.objective.split('_')[1]), random_seed=random_seed)
		elif args.objective.split('_')[0] == 'pestcontrol':
			assert 1 <= int(args.random_seed_config) <= 25
			random_seed = sorted(generate_random_seed_pestcontrol())[args.random_seed_config - 1]
			args.objective = PestControl(random_seed=random_seed)
		elif args.objective.split('_')[0] == 'centroid':
			assert 1 <= int(args.random_seed_config) <= 25
			random_seed_pair = generate_random_seed_pair_centroid()
			random_seed_config = args.random_seed_config - 1
			case_seed = sorted(random_seed_pair.keys())[int(random_seed_config / 5)]
			init_seed = sorted(random_seed_pair[case_seed])[int(random_seed_config % 5)]
			print(case_seed, init_seed)
			args.objective = Centroid(random_seed_pair=(case_seed, init_seed))
		else:
			raise NotImplementedError
	GRASB(**vars(args))