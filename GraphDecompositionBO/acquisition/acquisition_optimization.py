import os
import sys
import time
import psutil

import numpy as np

import torch
import torch.multiprocessing as mp

from GraphDecompositionBO.acquisition.acquisition_optimizers.starting_points import optim_inits
from GraphDecompositionBO.acquisition.acquisition_optimizers.greedy_ascent import greedy_ascent
from GraphDecompositionBO.acquisition.acquisition_optimizers.simulated_annealing import simulated_annealing
from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement
from GraphDecompositionBO.acquisition.acquisition_marginalization import prediction_statistic

MAX_N_ASCENT = float('inf')
N_CPU = os.cpu_count()
N_AVAILABLE_CORE = min(10, N_CPU)
N_SA_RUN = 10


def next_evaluation(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                    acquisition_func=expected_improvement, reference=None, parallel=None):
    """
    In case of '[Errno 24] Too many open files', check 'nofile' limit by typing 'ulimit -n' in a terminal
    if it is too small then add lines to '/etc/security/limits.conf'
        *               soft    nofile          [Large Number e.g 65536]
        *               soft    nofile          [Large Number e.g 65536]
    Rebooting may be needed.
    :param x_opt: 1D Tensor
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices: 1d np.array
    :param acquisition_func:
    :param reference: numeric(float)
    :param parallel:
    :return:
    """
    id_digit = np.ceil(np.log(np.prod(n_vertices)) / np.log(10))
    id_unit = torch.from_numpy(np.cumprod(np.concatenate([np.ones(1), n_vertices[:-1]])).astype(np.int))
    fmt_str = '\t %5.2f (id:%' + str(id_digit) + 'd) ==> %5.2f (id:%' + str(id_digit) + 'd)'

    start_time = time.time()
    print('(%s) Acquisition function optimization initial points selection began'
          % (time.strftime('%H:%M:%S', time.gmtime(start_time))))

    x_inits, acq_inits = optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                                     acquisition_func, reference)
    n_inits = x_inits.size(0)
    assert n_inits % 2 == 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(%s) Acquisition function optimization initial points selection ended - %s'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    start_time = time.time()
    print('(%s) Acquisition function optimization with %2d inits'
          % (time.strftime('%H:%M:%S', time.gmtime(start_time)), x_inits.size(0)))

    ga_args_list = [(x_inits[i], inference_samples, partition_samples, edge_mat_samples,
                     n_vertices, acquisition_func, MAX_N_ASCENT, reference) for i in range(n_inits)]
    ga_start_time = time.time()
    sys.stdout.write('    Greedy Ascent  began at %s ' % time.strftime('%H:%M:%S', time.gmtime(ga_start_time)))
    if parallel:
        with mp.Pool(processes=min(n_inits, N_CPU // 3)) as pool:
            ga_result = []
            process_started = [False] * n_inits
            process_running = [False] * n_inits
            process_index = 0
            while process_started.count(False) > 0:
                cpu_usage = psutil.cpu_percent(0.25)
                run_more = (100.0 - cpu_usage) * float(psutil.cpu_count()) > 100.0 * N_AVAILABLE_CORE
                if run_more:
                    ga_result.append(pool.apply_async(greedy_ascent, args=ga_args_list[process_index]))
                    process_started[process_index] = True
                    process_running[process_index] = True
                    process_index += 1
            while [not res.ready() for res in ga_result].count(True) > 0:
                time.sleep(1)
            ga_return_values = [res.get() for res in ga_result]
    else:
        ga_return_values = [greedy_ascent(*(ga_args_list[i])) for i in range(n_inits)]
    ga_opt_vrt, ga_opt_acq = zip(*ga_return_values)
    sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - ga_start_time)))
    print('  '.join(['%4.2f' % ga_opt_acq[i] for i in range(n_inits)]))

    opt_vrt = list(ga_opt_vrt[:])
    opt_acq = list(ga_opt_acq[:])

    ## Optimization using simulated annealing,
    ## 1. First optimize with Greedy Ascent, then do additional optimization with that results with Simulated Annealing
    ## 2. Optimize with a different set of initial points for Greedy Ascent and Simulated Annealing and choose the best
    ## Both does not any improvement on the result solely from Greedy Ascent
    # x_rands = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(N_SA_RUN, 1))
    #                            for n_v in n_vertices]), dim=1).long()
    # sa_args_list = [(x_rands[i], inference_samples, partition_samples, edge_mat_samples,
    #                  n_vertices, acquisition_func, reference) for i in range(N_SA_RUN)]
    # sa_start_time = time.time()
    # sys.stdout.write('    Sim. Annealing began at %s ' % time.strftime('%H:%M:%S', time.gmtime(sa_start_time)))
    # if parallel:
    #     with mp.Pool(processes=min(N_SA_RUN, N_CPU // 2)) as pool:
    #         sa_result = []
    #         process_started = [False] * N_SA_RUN
    #         process_running = [False] * N_SA_RUN
    #         process_index = 0
    #         while process_started.count(False) > 0:
    #             cpu_usage = psutil.cpu_percent(1.0)
    #             run_more = (100.0 - cpu_usage) * float(psutil.cpu_count()) > 100.0 * N_AVAILABLE_CORE
    #             if run_more:
    #                 sa_result.append(pool.apply_async(simulated_annealing, args=sa_args_list[process_index]))
    #                 process_started[process_index] = True
    #                 process_running[process_index] = True
    #                 process_index += 1
    #         while [not res.ready() for res in sa_result].count(True) > 0:
    #             time.sleep(1)
    #
    #         sa_return_values = [res.get() for res in sa_result]
    # else:
    #     sa_return_values = [simulated_annealing(*(sa_args_list[i])) for i in range(N_SA_RUN)]
    # sa_opt_vrt, sa_opt_acq = zip(*sa_return_values)
    # sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - sa_start_time)))
    # print('  '.join(['%4.2f' % sa_opt_acq[i] for i in range(N_SA_RUN)]))
    #
    # opt_vrt = opt_vrt + list(sa_opt_vrt[:])
    # opt_acq = opt_acq + list(sa_opt_acq[:])

    acq_max_inds = np.where(np.max(opt_acq) == np.array(opt_acq))[0]
    acq_max_ind = acq_max_inds[np.random.randint(0, acq_max_inds.size)]
    suggestion = opt_vrt[acq_max_ind]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(%s) Acquisition function optimization ended %s'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    mean, std, var = prediction_statistic(suggestion, inference_samples, partition_samples, n_vertices)
    return suggestion, mean, std, var
