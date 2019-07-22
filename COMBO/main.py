import sys
import time
import argparse

import torch

from COMBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from COMBO.graphGP.models.gp_regression import GPRegression
from COMBO.graphGP.sampler.sample_posterior import posterior_sampling

from COMBO.acquisition.acquisition_optimization import next_evaluation
from COMBO.acquisition.acquisition_functions import expected_improvement
from COMBO.acquisition.acquisition_marginalization import inference_sampling

from COMBO.config import experiment_directory
from COMBO.utils import model_data_filenames, load_model_data, displaying_and_logging

from COMBO.experiments.random_seed_config import generate_random_seed_pair_ising, \
    generate_random_seed_pair_contamination, generate_random_seed_pestcontrol, generate_random_seed_pair_centroid, \
    generate_random_seed_maxsat
from COMBO.experiments.test_functions.discretized_continuous import Branin, Hartmann6
from COMBO.experiments.test_functions.binary_categorical import Ising, Contamination
from COMBO.experiments.test_functions.multiple_categorical import PestControl, Centroid
from COMBO.experiments.MaxSAT.maximum_satisfiability import MaxSAT28, MaxSAT43, MaxSAT60
from COMBO.experiments.NAS.nas_binary import NASBinary


def COMBO(objective=None, n_eval=200, path=None, parallel=False, store_data=False, **kwargs):
    """

    :param objective:
    :param n_eval:
    :param path:
    :param parallel:
    :param kwargs:
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
        objective_id_list = [objective.__class__.__name__]
        if hasattr(objective, 'random_seed_info'):
            objective_id_list.append(objective.random_seed_info)
        if hasattr(objective, 'lamda'):
            objective_id_list.append('%.1E' % objective.lamda)
        if hasattr(objective, 'data_type'):
            objective_id_list.append(objective.data_type)
        objective_id_list.append('COMBO')
        objective_name = '_'.join(objective_id_list)
        model_filename, data_cfg_filaname, logfile_dir = model_data_filenames(exp_dir=exp_dir,
                                                                              objective_name=objective_name)

        n_vertices = objective.n_vertices
        adj_mat_list = objective.adjacency_mat
        grouped_log_beta = torch.ones(len(objective.fourier_freq))
        fourier_freq_list = objective.fourier_freq
        fourier_basis_list = objective.fourier_basis
        suggested_init = objective.suggested_init  # suggested_init should be 2d tensor
        n_init = suggested_init.size(0)

        kernel = DiffusionKernel(grouped_log_beta=grouped_log_beta,
                                 fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
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
        print('(%s) Burn-in' % time.strftime('%H:%M:%S', time.gmtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              log_beta, sorted_partition, n_sample=1, n_burn=99, n_thin=1)
        log_beta = sample_posterior[1][0]
        sorted_partition = sample_posterior[2][0]
        print('')
    else:
        surrogate_model, cfg_data, logfile_dir = load_model_data(path, exp_dir=experiment_directory())

    for _ in range(n_eval):
        start_time = time.time()
        reference = torch.min(eval_outputs, dim=0)[0].item()
        print('(%s) Sampling' % time.strftime('%H:%M:%S', time.gmtime()))
        sample_posterior = posterior_sampling(surrogate_model, eval_inputs, eval_outputs, n_vertices, adj_mat_list,
                                              log_beta, sorted_partition, n_sample=10, n_burn=0, n_thin=1)
        hyper_samples, log_beta_samples, partition_samples, freq_samples, basis_samples, edge_mat_samples = sample_posterior
        log_beta = log_beta_samples[-1]
        sorted_partition = partition_samples[-1]
        print('')

        x_opt = eval_inputs[torch.argmin(eval_outputs)]
        inference_samples = inference_sampling(eval_inputs, eval_outputs, n_vertices,
                                               hyper_samples, log_beta_samples, partition_samples,
                                               freq_samples, basis_samples)
        suggestion = next_evaluation(x_opt, eval_inputs, inference_samples, partition_samples, edge_mat_samples,
                                     n_vertices, acquisition_func, reference, parallel)
        next_eval, pred_mean, pred_std, pred_var = suggestion

        processing_time = time.time() - start_time

        eval_inputs = torch.cat([eval_inputs, next_eval.view(1, -1)], 0)
        eval_outputs = torch.cat([eval_outputs, objective.evaluate(eval_inputs[-1]).view(1, 1)])
        assert not torch.isnan(eval_outputs).any()

        time_list.append(time.time())
        elapse_list.append(processing_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())

        displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                               time_list, elapse_list, store_data)
        print('Optimizing %s with regularization %.2E up to %4d visualization random seed : %s'
              % (objective.__class__.__name__, objective.lamda if hasattr(objective, 'lamda') else 0, n_eval,
                 objective.random_seed_info if hasattr(objective, 'random_seed_info') else 'none'))


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='COMBO : Combinatorial Bayesian Optimization using the graph Cartesian product')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=1)
    parser_.add_argument('--path', dest='path')
    parser_.add_argument('--objective', dest='objective')
    parser_.add_argument('--lamda', dest='lamda', type=float, default=None)
    parser_.add_argument('--random_seed_config', dest='random_seed_config', type=int, default=None)
    parser_.add_argument('--parallel', dest='parallel', action='store_true', default=False)
    parser_.add_argument('--device', dest='device', type=int, default=None)

    args_ = parser_.parse_args()
    print(args_)
    kwag_ = vars(args_)
    path_ = kwag_['path']
    objective_ = kwag_['objective']
    random_seed_config_ = kwag_['random_seed_config']
    parallel_ = kwag_['parallel']
    if args_.device is None:
        del kwag_['device']
    print(kwag_)
    if random_seed_config_ is not None:
        assert 1 <= int(random_seed_config_) <= 25
        random_seed_config_ -= 1
    assert (path_ is None) != (objective_ is None)

    if objective_ == 'branin':
        kwag_['objective'] = Branin()
    elif objective_ == 'hartmann6':
        kwag_['objective'] = Hartmann6()
    elif objective_ == 'ising':
        random_seed_pair_ = generate_random_seed_pair_ising()
        case_seed_ = sorted(random_seed_pair_.keys())[int(random_seed_config_ / 5)]
        init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
        kwag_['objective'] = Ising(lamda=args_.lamda, random_seed_pair=(case_seed_, init_seed_))
    elif objective_ == 'contamination':
        random_seed_pair_ = generate_random_seed_pair_contamination()
        case_seed_ = sorted(random_seed_pair_.keys())[int(random_seed_config_ / 5)]
        init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
        kwag_['objective'] = Contamination(lamda=args_.lamda, random_seed_pair=(case_seed_, init_seed_))
    elif objective_ == 'centroid':
        random_seed_pair_ = generate_random_seed_pair_centroid()
        case_seed_ = sorted(random_seed_pair_.keys())[int(random_seed_config_ / 5)]
        init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
        kwag_['objective'] = Centroid(random_seed_pair=(case_seed_, init_seed_))
    elif objective_ == 'pestcontrol':
        random_seed_ = sorted(generate_random_seed_pestcontrol())[random_seed_config_]
        kwag_['objective'] = PestControl(random_seed=random_seed_)
    elif objective_ == 'maxsat28':
        random_seed_ = sorted(generate_random_seed_maxsat())[random_seed_config_]
        kwag_['objective'] = MaxSAT28(random_seed=random_seed_)
    elif objective_ == 'maxsat43':
        random_seed_ = sorted(generate_random_seed_maxsat())[random_seed_config_]
        kwag_['objective'] = MaxSAT43(random_seed=random_seed_)
    elif objective_ == 'maxsat60':
        random_seed_ = sorted(generate_random_seed_maxsat())[random_seed_config_]
        kwag_['objective'] = MaxSAT60(random_seed=random_seed_)
    elif objective_ == 'nasbinary':
        kwag_['objective'] = NASBinary(data_type='CIFAR10', device=args_.device)
        kwag_['store_data'] = True
    else:
        raise NotImplementedError
    COMBO(**kwag_)
