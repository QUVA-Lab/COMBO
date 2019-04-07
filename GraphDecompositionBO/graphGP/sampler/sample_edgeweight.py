import numpy as np

import torch

from GraphDecompositionBO.graphGP.inference.inference import Inference
from GraphDecompositionBO.graphGP.sampler.tool_partition import strong_product, kronecker
from GraphDecompositionBO.graphGP.sampler.tool_slice_sampling import univariate_slice_sampling
from GraphDecompositionBO.graphGP.sampler.priors import log_prior_edgeweight
from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_unit_in_group, group_input


def slice_edgeweight(model, input_data, output_data, list_of_adjacency, log_beta,
                     sorted_partition, fourier_freq_list, fourier_basis_list, ind):
    '''
    Slice sampling the edgeweight(exp('log_beta')) at 'ind' in 'log_beta' vector
    Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
    :param model:
    :param input_data:
    :param output_data:
    :param categories:
    :param list_of_adjacency:
    :param log_beta:
    :param sorted_partition:
    :param fourier_freq_list:
    :param fourier_basis_list:
    :param ind:
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

    model.kernel.fourier_freq_list = fourier_freq_list
    model.kernel.fourier_basis_list = fourier_basis_list
    unit_in_group = compute_unit_in_group(sorted_partition=sorted_partition, categories=categories)
    grouped_input_data = group_input(input_data=input_data, sorted_partition=sorted_partition, unit_in_group=unit_in_group)
    inference = Inference(train_data=(grouped_input_data, output_data), model=model)

    # numerical_buffer is added for numerical stability in eigendecomposition and subtracted later
    numerical_buffer = 1.0
    def logp(log_beta_ind):
        '''
        Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
        :param log_beta_ind: numeric(float)
        :return: numeric(float)
        '''
        log_prior = log_prior_edgeweight(log_beta_ind)
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
        model.kernel.fourier_freq_list[updated_subset_ind] = (fourier_freq_buffer - numerical_buffer).clamp(min=0)
        model.kernel.fourier_basis_list[updated_subset_ind] = fourier_basis
        return log_prior - float(inference.negative_log_likelihood(hyper=model.param_to_vec()))

    x0 = float(log_beta[ind])
    x1 = univariate_slice_sampling(logp, x0)
    log_beta[ind] = x1
    return log_beta, model.kernel.fourier_freq_list, model.kernel.fourier_basis_list


if __name__ == '__main__':
    pass
    import progressbar
    import time
    from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
    from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
    from GraphDecompositionBO.sampler.tool_partition import sort_partition, compute_unit_in_group, group_input, ungroup_input
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
    amp = torch.std(output_data, dim=0)
    log_beta = torch.randn(n_vars)
    model = GPRegression(kernel=DiffusionKernel(fourier_freq_list=[], fourier_basis_list=[]))
    model.kernel.log_amp.data = torch.log(amp)
    model.mean.const_mean.data = torch.mean(output_data, dim=0)
    model.likelihood.log_noise_var.data = torch.log(amp / 1000.)

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
    print(log_beta)
    bar = progressbar.ProgressBar(max_value=n_vars)
    for e in range(n_vars):
        bar.update(e)
        log_beta, fourier_freq_list, fourier_basis_list = slice_edgeweight(model, grouped_input_data, output_data, list_of_adjacency, log_beta, sorted_partition, fourier_freq_list, fourier_basis_list, ind=e)
    print(time.time() - start_time)
    print(log_beta)
