import time
import psutil

import numpy as np

import torch
import torch.multiprocessing as multiprocessing

from GraphDecompositionBO.acquisition.acquisition_optimizers.starting_points import optim_inits
from GraphDecompositionBO.acquisition.acquisition_optimizers.greedy_ascent import greedy_ascent
from GraphDecompositionBO.acquisition.acquisition_optimizers.simulated_annealing import simulated_annealing
from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement
from GraphDecompositionBO.acquisition.acquisition_utils import suggestion_statistic

N_AVAILABLE_CORE = 8# When there is this many available cpu cores new optimization is started
MAX_N_ASCENT = float('inf')


def next_evaluation(x_opt, input_data, inference_samples, partition_samples, edge_mat_samples, n_vertex,
                    acquisition_func=expected_improvement, reference=None, parallel=None):
    """

    :param x_opt:
    :param input_data:
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertex:
    :param acquisition_func:
    :param reference:
    :param verbose:
    :param parallel:
    :return:
    """

    start_time = time.time()
    print('Acqusition function optimization initial points selection %s has begun'
          % (time.strftime('%H:%M:%S', time.gmtime(start_time))))

    x_inits = optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertex, acquisition_func, reference)
    n_inits = x_inits.size(0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Acqusition function optimization initial points selection ended %s(%s)'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    start_time = time.time()
    print('Acqusition function optimization with %2d inits %s has begun'
          % (x_inits.size(0), time.strftime('%H:%M:%S', time.gmtime(start_time))))

    opt_vrt = []
    opt_acq = []
    if parallel:
        pool = multiprocessing.Pool(int(psutil.cpu_count() / 9)) if parallel else None
        results = []
        for i in range(0, n_inits):
            args_ga = (x_inits[i], inference_samples, edge_mat_samples, n_vertex,
                       acquisition_func, float('inf'), reference)
            proc_ga = pool.apply_async(greedy_ascent, args=args_ga)
            args_sa = (x_inits[i + n_inits], inference_samples, partition_samples, edge_mat_samples, n_vertex,
                       acquisition_func, reference)
            proc_sa = pool.apply_async(simulated_annealing, args=args_sa)
            results.append(proc_ga)
            results.append(proc_sa)

        for res_vrt, res_acq in [res.get() for res in results]:
            if not (input_data == res_vrt).all(dim=1).any():
                opt_vrt.append(res_vrt)
                opt_acq.append(res_acq)
        pool.close()
        pool.join()
    else:
        for i in range(0, n_inits):
            max_vrt_ga, max_acq_ga = greedy_ascent(x_inits[i], inference_samples, edge_mat_samples, n_vertex,
                                                   acquisition_func, MAX_N_ASCENT, reference)
            max_vrt_sa, max_acq_sa = simulated_annealing(x_inits[i + n_inits], inference_samples, partition_samples,
                                                         edge_mat_samples, n_vertex, acquisition_func, reference)
            if not torch.isnan(max_acq_ga) and not (input_data == max_vrt_ga).all(dim=1).any():
                opt_vrt.append(max_vrt_ga)
                opt_acq.append(max_acq_ga)
            if not torch.isnan(max_acq_sa) and not (input_data == max_vrt_sa).all(dim=1).any():
                opt_vrt.append(max_vrt_sa)
                opt_acq.append(max_acq_sa)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Acqusition function optimization ended %s(%s)'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    suggestion = opt_vrt[np.nanargmax(opt_acq)]
    mean, std, var = suggestion_statistic(suggestion, inference_samples, partition_samples, n_vertex)
    return suggestion, mean, std, var
