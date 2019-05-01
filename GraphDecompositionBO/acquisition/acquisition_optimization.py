import os
import sys
import time

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
            ga_result = pool.starmap(greedy_ascent, ga_args_list)
        ga_opt_vrt = [elm[0] for elm in ga_result]
        ga_opt_acq = [elm[1] for elm in ga_result]
    else:
        ga_opt_vrt, ga_opt_acq = zip(*[greedy_ascent(*(ga_args_list[i])) for i in range(n_inits)])
    sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - ga_start_time)))
    print('  '.join(['%4.2f' % ga_opt_acq[i] for i in range(n_inits)]))

    x_rands = torch.cat(tuple([torch.randint(low=0, high=int(n_v), size=(n_inits, 1))
                               for n_v in n_vertices]), dim=1).long()
    sa_args_list = [(x_rands[i], inference_samples, partition_samples, edge_mat_samples,
                     n_vertices, acquisition_func, reference) for i in range(n_inits)]
    sa_start_time = time.time()
    sys.stdout.write('    Sim. Annealing began at %s ' % time.strftime('%H:%M:%S', time.gmtime(sa_start_time)))
    if parallel:
        with mp.Pool(processes=min(n_inits, N_CPU // 2)) as pool:
            sa_result = pool.starmap(simulated_annealing, sa_args_list)
        sa_opt_vrt = [elm[0] for elm in sa_result]
        sa_opt_acq = [elm[1] for elm in sa_result]
    else:
        sa_opt_vrt, sa_opt_acq = zip(*[simulated_annealing(*(sa_args_list[i])) for i in range(n_inits)])
    sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - sa_start_time)))
    print('  '.join(['%4.2f' % sa_opt_acq[i] for i in range(n_inits)]))

    opt_vrt = list(ga_opt_vrt[:]) + list(sa_opt_vrt[:])
    opt_acq = list(ga_opt_acq[:]) + list(sa_opt_acq[:])

    acq_max_inds = np.where(np.max(opt_acq) == np.array(opt_acq))[0]
    acq_max_ind = acq_max_inds[np.random.randint(0, acq_max_inds.size)]
    suggestion = opt_vrt[acq_max_ind]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('(%s) Acquisition function optimization ended %s'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    mean, std, var = prediction_statistic(suggestion, inference_samples, partition_samples, n_vertices)
    return suggestion, mean, std, var
