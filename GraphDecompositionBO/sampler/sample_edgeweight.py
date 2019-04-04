import time
import numpy as np

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference


from GraphDecompositionBO.sampler.grouping_utils import sort_partition, compute_unit_in_group, group_input, ungroup_input, strong_product, kronecker


def univariate_slice_sampling(logp, x0):
    '''
    Univariate Slice Sampling using doubling scheme
    :param logp:
    :param x0:
    :return:
    '''
    width = 1.0
    max_steps_out = 10
    upper = width * np.random.rand()
    lower = upper - width
    llh0 = logp(x0)
    slice_h = np.log(np.random.rand()) + llh0
    llh_record = {}

    # Step Out
    steps_out = 0
    logp_lower = logp(lower)
    logp_upper = logp(upper)
    llh_record[float(lower)] = logp_lower
    llh_record[float(upper)] = logp_upper
    while (logp_lower > slice_h or logp_upper > slice_h) and (steps_out < max_steps_out):
        if np.random.rand() < 0.5:
            lower -= (upper - lower)
        else:
            upper += (upper - lower)
        steps_out += 1
        try:
            logp_lower = llh_record[float(lower)]
        except KeyError:
            logp_lower = logp(lower)
            llh_record[float(lower)] = logp_lower
        try:
            logp_upper = llh_record[float(upper)]
        except KeyError:
            logp_upper = logp(lower)
            llh_record[float(upper)] = logp_upper

    # Shrinkage
    start_upper = upper
    start_lower = lower
    n_steps_in = 0
    while lower < upper:
        x1 = (upper - lower) * np.random.rand() + lower
        llh1 = logp(x1)
        if llh1 > slice_h and accept(logp, x0, x1, slice_h, width, start_lower, start_upper, llh_record):
            return x1 + torch.zeros_like(x0)
        else:
            if x1 < x0:
                lower = x1
            else:
                upper = x1
        n_steps_in += 1
    raise RuntimeError('Shrinkage collapsed to a degenerated interval(point)')


def accept(logp, x0, x1, slice_h, width, lower, upper, llh_record):
    acceptance = False
    while upper - lower > 1.1 * width:
        mid = (lower + upper) / 2.0
        if (x0 < mid and x1 >= mid) or (x0 >= mid and x1 < mid):
            acceptance = True
        if x1 < mid:
            upper = mid
        else:
            lower = mid
        try:
            logp_lower = llh_record[float(lower)]
        except KeyError:
            logp_lower = logp(lower)
            llh_record[float(lower)] = logp_lower
        try:
            logp_upper = llh_record[float(upper)]
        except KeyError:
            logp_upper = logp(lower)
            llh_record[float(upper)] = logp_upper
        if acceptance and slice_h >= logp_lower and slice_h >= logp_upper:
            return False
    return True


def edgeweight_log_prior(log_beta_ind):
    '''

    :param log_beta_ind: scalar ind-th element of log_beta
    :return:
    '''
    # TODO : define a prior prior for (scalar) log_beta
    if np.exp(log_beta_ind) > 2.0:
        return -float('inf')
    else:
        return np.log(1.0 / 2.0)


def slice_edgeweight(grouped_input_data, output_data, list_of_adjacency, mean, log_amp, log_beta, log_noise_var,
                     sorted_partition, fourier_freq_list, fourier_basis_list, ind):
    '''
    Slice sampling edgeweight
    :param grouped_input_data:
    :param output_data:
    :param categories:
    :param list_of_adjacency:
    :param mean:
    :param log_amp:
    :param log_beta:
    :param log_noise_var:
    :param sorted_partition:
    :param fourier_freq_list:
    :param fourier_basis_list:
    :param ind:
    :param freq_basis:
    :return:
    '''
    updated_subset_ind = [(ind in subset) for subset in sorted_partition].index(True)
    updated_subset = sorted_partition[updated_subset_ind]
    n_pre = updated_subset.index(ind)
    n_suf = len(updated_subset) - n_pre - 1
    if n_pre > 0:
        if n_pre > 1:
            prefix = strong_product(list_of_adjacency=list_of_adjacency, beta=torch.exp(log_beta), subset=updated_subset[:n_pre])
        else:
            prefix = list_of_adjacency[updated_subset[0]]
        prefix_id_added = prefix + torch.diag(prefix.new_ones(prefix.size(0)))
    else:
        prefix_id_added = None
    if n_suf > 0:
        if n_suf > 1:
            suffix = strong_product(list_of_adjacency=list_of_adjacency, beta=torch.exp(log_beta), subset=updated_subset[-n_suf:])
        else:
            suffix = list_of_adjacency[updated_subset[-1]]
        suffix_id_added = suffix + torch.diag(suffix.new_ones(suffix.size(0)))
    else:
        suffix_id_added = None
    id_for_updated_adj_mat = torch.diag(list_of_adjacency[ind].new_ones(list_of_adjacency[ind].size(0)))

    kernel = DiffusionKernel(fourier_freq_list=fourier_freq_list, fourier_basis_list=fourier_basis_list)
    kernel.log_amp.data = log_amp
    model = GPRegression(kernel=kernel)
    model.mean.const_mean.data = mean
    model.likelihood.log_noise_var.data = log_noise_var

    inference = Inference(train_data=(grouped_input_data, output_data), model=model)

    # numerical_buffer is added for numerical stability in eigendecomposition and subtracted later
    numerical_buffer = 1.0
    def logp(log_beta_ind):
        log_prior = edgeweight_log_prior(log_beta_ind)
        if np.isinf(log_prior):
            return log_prior
        adj_id_added = list_of_adjacency[ind] * np.exp(log_beta_ind) + id_for_updated_adj_mat
        if prefix_id_added is not None:
            adj_id_added = kronecker(prefix_id_added, adj_id_added)
        if suffix_id_added is not None:
            adj_id_added = kronecker(adj_id_added, suffix_id_added)
        # D(id_added) - A(id_added) = D(original) - A(original)
        laplacian = torch.diag(torch.sum(adj_id_added, dim=0) + numerical_buffer) - adj_id_added
        fourier_freq_buffer, fourier_basis = torch.symeig(laplacian, eigenvectors=True)
        kernel.fourier_freq_list[updated_subset_ind] = (fourier_freq_buffer - numerical_buffer).clamp(min=0)
        kernel.fourier_basis_list[updated_subset_ind] = fourier_basis
        return edgeweight_log_prior(log_beta_ind) - inference.negative_log_likelihood(hyper=model.param_to_vec())

    x0 = log_beta[ind]
    x1 = univariate_slice_sampling(logp, x0)
    log_beta[ind] = x1
    return log_beta, kernel.fourier_freq_list, kernel.fourier_basis_list


if __name__ == '__main__':
    import progressbar
    n_vars = 50
    n_data = 60
    categories = np.random.randint(2, 3, n_vars)
    list_of_adjacency = []
    for d in range(n_vars):
        adjacency = torch.ones(categories[d], categories[d])
        adjacency[range(categories[d]), range(categories[d])] = 0
        list_of_adjacency.append(adjacency)
    input_data = torch.zeros(n_data, n_vars).long()
    output_data = torch.randn(n_data, 1)
    for a in range(n_vars):
        input_data[:, a] = torch.randint(0, categories[a], (n_data,))
    inds = range(n_vars)
    np.random.shuffle(inds)
    b = 0
    random_partition = []
    while b < n_vars:
        subset_size = np.random.poisson(2) + 1
        random_partition.append(inds[b:b + subset_size])
        b += subset_size
    sorted_partition = sort_partition(random_partition)
    unit_in_group = compute_unit_in_group(sorted_partition, categories)
    grouped_input_data = group_input(input_data, sorted_partition, unit_in_group)
    input_data_re = ungroup_input(grouped_input_data, sorted_partition, unit_in_group)
    mean = torch.mean(output_data, dim=0)
    amp = torch.std(output_data, dim=0)
    log_amp = torch.log(amp)
    log_noise_var = torch.log(amp / 1000.)
    log_beta = torch.zeros(n_vars)

    start_time = time.time()
    fourier_freq_list = []
    fourier_basis_list = []
    for subset in sorted_partition:
        adj_mat = strong_product(list_of_adjacency=list_of_adjacency, beta=torch.exp(log_beta), subset=subset)
        deg_mat = torch.diag(torch.sum(adj_mat, dim=0))
        laplacian = deg_mat - adj_mat
        fourier_freq, fourier_basis = torch.symeig(laplacian, eigenvectors=True)
        fourier_freq_list.append(fourier_freq)
        fourier_basis_list.append(fourier_basis)
    print('init elapsed time', time.time() - start_time)

    start_time = time.time()
    print('%d variables' % n_vars)
    print(len(sorted_partition))
    print(sorted([len(elm) for elm in sorted_partition]))
    bar = progressbar.ProgressBar(max_value=n_vars)
    for e in range(n_vars):
        bar.update(e)
        log_beta, fourier_freq_list, fourier_basis_list = slice_edgeweight(grouped_input_data, output_data, list_of_adjacency, mean, log_amp, log_beta, log_noise_var, sorted_partition, fourier_freq_list, fourier_basis_list, ind=e)
    print(time.time() - start_time)
    print(sorted([len(elm) for elm in sorted_partition]))
