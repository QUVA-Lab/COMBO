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
from GraphDecompositionBO.acquisition.acquisition_marginalization import suggestion_statistic

MAX_N_ASCENT = float('inf')
N_CPU = os.cpu_count()


def next_evaluation(x_opt, input_data, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                    acquisition_func=expected_improvement, reference=None, parallel=None):
    """
    In case of '[Errno 24] Too many open files', check 'nofile' limit by typing 'ulimit -n' in a terminal
    if it is too small then add lines to '/etc/security/limits.conf'
        *               soft    nofile          [Large Number e.g 65536]
        *               soft    nofile          [Large Number e.g 65536]
    Rebooting may be needed.
    :param x_opt: 1D Tensor
    :param input_data:
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

    start_time = time.time()
    print('Acquisition function optimization initial points selection %s has begun'
          % (time.strftime('%H:%M:%S', time.gmtime(start_time))))

    x_inits, acq_inits = optim_inits(x_opt, inference_samples, partition_samples, edge_mat_samples, n_vertices, acquisition_func, reference)
    n_inits = x_inits.size(0)
    assert n_inits % 2 == 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Acquisition function optimization initial points selection ended %s(%s)'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    start_time = time.time()
    print('Acquisition function optimization with %2d inits %s has begun'
          % (x_inits.size(0), time.strftime('%H:%M:%S', time.gmtime(start_time))))

    if parallel:
        ga_args_list = [(x_inits[i], inference_samples, partition_samples, edge_mat_samples,
                         n_vertices, acquisition_func, MAX_N_ASCENT, reference) for i in range(n_inits)]
        ga_start_time = time.time()
        sys.stdout.write('    Greedy Ascent  began at %s ' % time.strftime('%H:%M:%S', time.gmtime(ga_start_time)))
        with mp.Pool(processes=min(n_inits, N_CPU)) as pool:
            ga_result = pool.starmap(greedy_ascent, ga_args_list)
        ga_opt_vrt = [elm[0] for elm in ga_result]
        ga_opt_acq = [elm[1] for elm in ga_result]
        inits_ind = range(n_inits)
        for i in range(n_inits-1, -1, -1):
            if np.isnan(ga_opt_acq[i]) or (input_data == ga_opt_vrt[i]).all(dim=1).any():
                ga_opt_vrt.pop(i)
                ga_opt_acq.pop(i)
                inits_ind.pop(i)
        sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - ga_start_time)))

        x_inits = x_inits[inits_ind]
        acq_inits = acq_inits[inits_ind]
        n_inits_sa = len(ga_opt_acq)

        sa_args_list = [(ga_opt_vrt[i], inference_samples, partition_samples, edge_mat_samples,
                         n_vertices, acquisition_func, reference) for i in range(n_inits_sa)]
        sa_start_time = time.time()
        sys.stdout.write('    Sim. Annealing began at %s ' % time.strftime('%H:%M:%S', time.gmtime(sa_start_time)))
        with mp.Pool(processes=min(n_inits_sa, N_CPU)) as pool:
            sa_result = pool.starmap(simulated_annealing, sa_args_list)
        sa_opt_vrt = [elm[0] for elm in sa_result]
        sa_opt_acq = [elm[1] for elm in sa_result]
        for i in range(n_inits_sa-1, -1, -1):
            if np.isnan(sa_opt_acq[i]) or (input_data == sa_opt_vrt[i]).all(dim=1).any() or ga_opt_acq[i] > sa_opt_acq[i]:
                sa_opt_vrt[i] = ga_opt_vrt[i]
                sa_opt_acq[i] = ga_opt_acq[i]
        sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - sa_start_time)))
    else:
        ga_opt_vrt = []
        ga_opt_acq = []
        inits_ind = []
        ga_start_time = time.time()
        sys.stdout.write('    Greedy Ascent  began at %s ' % time.strftime('%H:%M:%S', time.gmtime(ga_start_time)))
        for i in range(n_inits):
            max_vrt_ga, max_acq_ga = greedy_ascent(x_inits[i], inference_samples, partition_samples, edge_mat_samples,
                                                   n_vertices, acquisition_func, MAX_N_ASCENT, reference)
            if not np.isnan(max_acq_ga) and not (input_data == max_vrt_ga).all(dim=1).any():
                ga_opt_vrt.append(max_vrt_ga)
                ga_opt_acq.append(max_acq_ga)
                inits_ind.append(i)
        sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - ga_start_time)))

        x_inits = x_inits[inits_ind]
        acq_inits = acq_inits[inits_ind]
        n_inits_sa = len(ga_opt_acq)

        sa_opt_vrt = []
        sa_opt_acq = []
        sa_start_time = time.time()
        sys.stdout.write('    Sim. Annealing began at %s ' % time.strftime('%H:%M:%S', time.gmtime(sa_start_time)))
        for i in range(n_inits_sa):
            max_vrt_sa, max_acq_sa = simulated_annealing(ga_opt_vrt[i], inference_samples, partition_samples,
                                                         edge_mat_samples, n_vertices, acquisition_func, reference)
            if not np.isnan(max_acq_sa) and not (input_data == max_vrt_sa).all(dim=1).any():
                sa_opt_vrt.append(max_vrt_sa if max_acq_sa > ga_opt_acq[i] else ga_opt_vrt[i])
                sa_opt_acq.append(max_acq_sa if max_acq_sa > ga_opt_acq[i] else ga_opt_acq[i])
        sys.stdout.write('and took %s\n' % time.strftime('%H:%M:%S', time.gmtime(time.time() - sa_start_time)))

    acq_max_inds = np.where(np.max(sa_opt_acq) == np.array(sa_opt_acq))[0]
    acq_max_ind = acq_max_inds[np.random.randint(0, acq_max_inds.size)]
    suggestion = sa_opt_vrt[acq_max_ind]
    mean, std, var = suggestion_statistic(suggestion, inference_samples, partition_samples, n_vertices)

    acq_str_list = []
    fmt_str = '\t At an initial point %7.4f (id:%' + str(id_digit) + 'd)' \
              + ' + Greedy Ascent %7.4f (id:%' + str(id_digit) + 'd)' \
              + ' + Simulated Annealing %7.4f (id:%' + str(id_digit) + 'd) '
    for i in range(n_inits_sa):
        acq_str_i = fmt_str % (acq_inits[i], torch.sum(x_inits[i] * id_unit),
                               ga_opt_acq[i], torch.sum(ga_opt_vrt[i] * id_unit),
                               sa_opt_acq[i], torch.sum(sa_opt_vrt[i] * id_unit))
        acq_str_i += ' [' + ('Remain' if torch.all(x_inits[i] == ga_opt_vrt[i]) else 'Update') + ','
        acq_str_i += ('Remain' if torch.all(ga_opt_vrt[i] == sa_opt_vrt[i]) else 'Update') + ']'
        if i == acq_max_ind:
            acq_str_i += '  <=== Chosen Maximum'
        acq_str_list.append(acq_str_i)

    print('\n'.join(acq_str_list))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Acquisition function optimization ended %s(%s)'
          % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    return suggestion, mean, std, var
