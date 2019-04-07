import time
import numpy as np


from GraphDecompositionBO.graphGP.sampler.tool_partition import sort_partition, compute_unit_in_group, group_input, ungroup_input, strong_product


def univariate_slice_sampling(logp, x0):
    '''
    Univariate Slice Sampling using doubling scheme
    :param logp: numeric(float) -> numeric(float), a log density function
    :param x0: numeric(float)
    :return: numeric(float), sampled x1
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
            return x1
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


if __name__ == '__main__':
    import progressbar
    import torch
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
