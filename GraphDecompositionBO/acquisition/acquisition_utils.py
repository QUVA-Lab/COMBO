import copy
import time
import math
import psutil

import numpy as np

import torch
import torch.multiprocessing as multiprocessing

from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement

N_AVAILABLE_CORE = 8 # When there is this many available cpu cores new optimization is started
N_RANDOM_VERTICES = 20000
N_GREEDY_ASCENT_INIT = 20
N_SPRAY = 10
N_HOPS_SPRAY = 2


def next_evaluation(x_best, adjacency_mat_list, inferences, acquisition_func=expected_improvement, reference=None, verbose=False, parallel=None):
    """
    
    :param x_best: Best evaluation so far
    :param adjacency_mat_list: 
    :param inferences: 
    :param acquisition_func: 
    :param reference: 
    :param verbose: 
    :param pool: 
    :return: 
    """
    if acquisition_func.requires_reference:
        assert reference is not None

    start_time = time.time()
    print('Acqusition function optimization initial points selection (%d random, %d spray) %s has begun' % (N_RANDOM_VERTICES, N_SPRAY, time.strftime('%H:%M:%S', time.gmtime(start_time))))

    x_inits = _greedy_ascent_inits(x_best=x_best, adjacency_mat_list=adjacency_mat_list, inferences=inferences, acquisition_func=acquisition_func, reference=reference)

    end_time = time.time()
    print('Acqusition function optimization initial points selection ended %s(%s)' % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))
    start_time = time.time()
    print('Acqusition function optimization with %2d inits %s has begun' % (N_GREEDY_ASCENT_INIT, time.strftime('%H:%M:%S', time.gmtime(start_time))))

    n_factor = len(adjacency_mat_list)
    local_optima = []
    optima_value = []
    max_ascent = float('inf')
    i = 0
    if parallel:
        while max_ascent > 0:
            pool = multiprocessing.Pool(int(psutil.cpu_count() / 9)) if parallel else None
            results = [pool.apply_async(_greedy_ascent, args=(x_inits[j], adjacency_mat_list, inferences, acquisition_func, max_ascent, reference, verbose)) for j in range(i, i+N_GREEDY_ASCENT_INIT)]
            i += N_GREEDY_ASCENT_INIT
            for local_opt, opt_val in [res.get() for res in results]:
                if not (inferences[0].train_x == local_opt).all(dim=1).any():
                    local_optima.append(local_opt)
                    optima_value.append(opt_val)
            pool.close()
            pool.join()

            if len(local_optima) > 0:
                break

            max_ascent = n_factor * 2 if math.isinf(max_ascent) else max_ascent - 2
    else:
        while max_ascent > 0:
            optimum_loc, optimum_value = _greedy_ascent(x_init=x_inits[i], adjacency_mat_list=adjacency_mat_list, inferences=inferences, acquisition_func=acquisition_func, max_ascent=max_ascent, reference=reference, verbose=verbose)
            i += 1
            if not (inferences[0].train_x == optimum_loc).all(dim=1).any():
                local_optima.append(optimum_loc)
                optima_value.append(optimum_value)

            if i > N_GREEDY_ASCENT_INIT and len(local_optima) > 0:
                break

            if i % N_GREEDY_ASCENT_INIT == 0:
                max_ascent = n_factor * 2 if math.isinf(max_ascent) else max_ascent - 2

    if len(optima_value) == 0:
        for i in range(x_inits.size(0)):
            if not (inferences[0].train_x == x_inits[i]).all(dim=1).any():
                local_optima = [x_inits[i]]
                optima_value = [0]
                break

    end_time = time.time()
    print('Acqusition function optimization ended %s(%s)' % (time.strftime('%H:%M:%S', time.gmtime(end_time)), time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))))

    suggestion = local_optima[np.nanargmax(optima_value)]
    mean, std, var = _mean_std_var(suggestion, inferences)
    return suggestion, mean, std, var


def _greedy_ascent(x_init, adjacency_mat_list, inferences, acquisition_func=expected_improvement, max_ascent=float('inf'), reference=None, verbose=False):
    """
    In order to find local maximum of an acquisition function, at each vertex, it follows the most increasing edge starting from an initial point
    if MAX_ASCENT is infinity, this method tries to find local maximum, otherwise, it may stop at a noncritical vertex (this option is for a computational reason)
    :param inferences:
    :param adjacency_mat_list:
    :param x_init: 1d tensor
    :param acquisition_func:
    :param reference:
    :return:
    """
    if acquisition_func.requires_reference:
        assert reference is not None

    n_ascent = 0
    x = x_init
    max_acquisition = _expected_acquisition(x=x, inferences=inferences, acquisition_func=acquisition_func, reference=reference, verbose=verbose)
    while n_ascent < max_ascent:
        x_nbds = _neighbors(adjacency_mat_list=adjacency_mat_list, x=x)
        nbds_acquisition = _expected_acquisition(x=x_nbds, inferences=inferences, acquisition_func=acquisition_func, reference=reference, verbose=verbose)
        max_nbd_acquisition, max_nbd_ind = torch.max(nbds_acquisition, 0)
        if max_nbd_acquisition > max_acquisition:
            max_acquisition = max_nbd_acquisition
            x = x_nbds[max_nbd_ind.item()]
            n_ascent += 1
        else:
            break
    return x, max_acquisition


def _neighbors(x, adjacency_mat_list):
    """
    For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by adjancency_mat_list
    :param adjacency_mat_list: adjacency matrices for each factor
    :param x: 1d tensor
    :return: 2d tensor in which each row is 1-hamming distance far from x
    """
    n_factors = len(adjacency_mat_list)
    factor_nbd_list = [adjacency_mat_list[i][x[i]].nonzero() for i in range(n_factors)]
    factor_nbd_list = [elm[elm!=x[i]] for i, elm in enumerate(factor_nbd_list)]
    nbd_cnt = [elm.numel() for elm in factor_nbd_list]
    n_nbds = sum(nbd_cnt)
    neighbors = x.repeat(n_nbds, 1)
    type_begin_ind = 0
    for i in range(len(nbd_cnt)):
        neighbors[type_begin_ind:type_begin_ind + nbd_cnt[i], i] = factor_nbd_list[i].squeeze()
        type_begin_ind += nbd_cnt[i]
    return neighbors


def _expected_acquisition(x, inferences, acquisition_func=expected_improvement, reference=None, verbose=False):
    """
    GP hyperparameters are sampled from a posterior. By using posterior samples, the acquisition function is also averaged over posterior samples  
    :param x: 1d or 2d tensor
    :param inferences: inference method for each posterior sample
    :param acquisition_func: 
    :param reference: 
    :param verbose: 
    :return: 
    """
    if acquisition_func.requires_reference:
        assert reference is not None

    if x.dim() == 1:
        x = x.unsqueeze(0)

    acquisition_sample_list = []
    if verbose:
        numerically_stable_list = []
        zero_pred_var_list = []
    for s in range(len(inferences)):
        pred_dist = inferences[s].predict(x, verbose=verbose)
        pred_mean_sample = pred_dist[0].detach()
        pred_var_sample = pred_dist[1].detach()
        acquisition_sample_list.append(acquisition_func(pred_mean_sample[:, 0], pred_var_sample[:, 0], reference=reference))
        if verbose:
            numerically_stable_list.append(pred_dist[2])
            zero_pred_var_list.append(pred_dist[3])

    return torch.stack(acquisition_sample_list, 1).sum(1, keepdim=True)


def _mean_std_var(x, inferences):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    mean_sample_list = []
    std_sample_list = []
    var_sample_list = []
    for s in range(len(inferences)):
        pred_dist = inferences[s].predict(x)
        pred_mean_sample = pred_dist[0]
        pred_var_sample = pred_dist[1]
        pred_std_sample = pred_var_sample ** 0.5
        mean_sample_list.append(pred_mean_sample.data)
        std_sample_list.append(pred_std_sample.data)
        var_sample_list.append(pred_var_sample.data)
    return torch.cat(mean_sample_list, 1).mean(1, keepdim=True),\
           torch.cat(std_sample_list, 1).mean(1, keepdim=True),\
           torch.cat(var_sample_list, 1).mean(1, keepdim=True)


def _greedy_ascent_inits(x_best, adjacency_mat_list, inferences, acquisition_func=expected_improvement, reference=None):
    """
    On (quasi- or not) ramdomly sampled points, expected acquisition function is evaluated 
    and the points with large acqusition values are used as initial points for further greedy ascent on a graph.
    In Spearmint implementation, they pointed out the hack of using 'spray points' is quite critical for good performance.
    So we use the same hack.
    Sobol sequence is not avaiable on graphs, so instead, we use uniform sampling.
    :param x_best: spray points are random selected near the x_best (the best evaluation so far)
    :param adjacency_mat_list: 
    :param inferences: 
    :param acquisition_func: 
    :param reference: 
    :return: 
    """
    if acquisition_func.requires_reference:
        assert reference is not None

    random_vertices = torch.stack([torch.randint(low=0, high=elm.size(0), size=(N_RANDOM_VERTICES,)) for elm in adjacency_mat_list], dim=1).long()
    sprayed_vertices = _random_neighbors(x_center=x_best, adjacency_mat_list=adjacency_mat_list)
    x_init_candidates = torch.cat([sprayed_vertices, random_vertices], dim=0)
    acquisition_values = _expected_acquisition(x=x_init_candidates, inferences=inferences, acquisition_func=acquisition_func, reference=reference, verbose=False)

    nonnan_ind = (acquisition_values == acquisition_values).squeeze(1)
    x_init_candidates = x_init_candidates[nonnan_ind]
    acquisition_values = acquisition_values[nonnan_ind]

    _, acquisition_sort_ind = torch.sort(acquisition_values.squeeze(1), descending=True)
    x_init_candidates = x_init_candidates[acquisition_sort_ind]

    return x_init_candidates


def _random_neighbors(x_center, adjacency_mat_list, n_spray=N_SPRAY):
    """
    For given points, n_hops distant vertices are randomly sampled.
    When x_center is x_best (the best evaluation point so far), the result is a points so called 'spray points' in Spearmint implementation.
    :param x_center: 1d tensor 
    :param adjacency_mat_list: 
    :param n_spray: 
    :param n_hops: 
    :return: 
    """
    n_factors = len(adjacency_mat_list)
    random_neighbor = torch.empty(0, dtype=x_center.dtype, device=x_center.device)
    for i in range(n_spray):
        factor_list = []
        while len(factor_list) < max(1, int(n_factors * 0.1)):
            factor = int(torch.randint(0, n_factors, (1,)))
            if len(factor_list) == 0 or factor != factor_list[-1]:
                factor_list.append(factor)
        x = x_center.clone()
        for f in factor_list:
            adjacent_vertices = adjacency_mat_list[f][x[f]].nonzero()
            x[f] = adjacent_vertices[int(torch.randint(0, adjacent_vertices.numel(), (1,)))]
        random_neighbor = torch.cat([random_neighbor, x.unsqueeze(0)], dim=0)
    return random_neighbor


def deepcopy_inference(inference, gp_param_samples):
    """
    GP hyperparameters are sampled from a posterior. 
    For each posterior sample, we need to have inference class for later purpose  
    :param inference: 
    :param gp_param_samples: 
    :return: 
    """
    inferences = []
    for s in range(gp_param_samples.size(0)):
        model = copy.deepcopy(inference.model)
        deepcopied_inference = inference.__class__((inference.train_x, inference.train_y), model)
        deepcopied_inference.cholesky_update(gp_param_samples[s])
        inferences.append(deepcopied_inference)
    return inferences

