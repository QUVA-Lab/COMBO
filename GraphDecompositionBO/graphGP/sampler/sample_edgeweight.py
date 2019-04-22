import numpy as np

import torch

from GraphDecompositionBO.graphGP.inference.inference import Inference
from GraphDecompositionBO.graphGP.sampler.tool_partition import strong_product, kronecker
from GraphDecompositionBO.graphGP.sampler.tool_slice_sampling import univariate_slice_sampling
from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_unit_in_group, group_input
from GraphDecompositionBO.graphGP.sampler.priors import log_prior_edgeweight


def slice_edgeweight(model, input_data, output_data, categories, list_of_adjacency, log_beta,
                     sorted_partition, fourier_freq_list, fourier_basis_list, ind):
    '''
    Slice sampling the edgeweight(exp('log_beta')) at 'ind' in 'log_beta' vector
    Note that model.kernel members (fourier_freq_list, fourier_basis_list) are updated.
    :param model:
    :param input_data:
    :param output_data:
    :param categories: 1d np.array
    :param list_of_adjacency: list of 2D torch.Tensor of adjacency matrix of base subgraphs
    :param log_beta:
    :param sorted_partition: Partition of {0, ..., K-1}, list of subsets(list)
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
    from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel as DiffusionKernel_
    from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression as GPRegression_
    from GraphDecompositionBO.graphGP.sampler.tool_partition import sort_partition as sort_partition_
    from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_unit_in_group as compute_unit_in_group_
    from GraphDecompositionBO.graphGP.sampler.tool_partition import group_input as group_input_
    from GraphDecompositionBO.graphGP.sampler.tool_partition import ungroup_input as ungroup_input_
    from GraphDecompositionBO.graphGP.sampler.priors import log_prior_partition as log_prior_partition_
    n_vars_ = 100
    n_data_ = 60
    categories_ = np.random.randint(5, 6, n_vars_)
    list_of_adjacency_ = []
    for d_ in range(n_vars_):
        adjacency_ = torch.ones(categories_[d_], categories_[d_])
        adjacency_[range(categories_[d_]), range(categories_[d_])] = 0
        list_of_adjacency_.append(adjacency_)
    input_data_ = torch.zeros(n_data_, n_vars_).long()
    output_data_ = torch.randn(n_data_, 1)
    for a_ in range(n_vars_):
        input_data_[:, a_] = torch.randint(0, categories_[a_], (n_data_,))
    inds_ = range(n_vars_)
    np.random.shuffle(inds_)
    while True:
        random_partition_ = []
        b_ = 0
        while b_ < n_vars_:
            subset_size_ = np.random.randint(1, 5)
            random_partition_.append(inds_[b_:b_ + subset_size_])
            b_ += subset_size_
        sorted_partition_ = sort_partition_(random_partition_)
        print(sorted_partition_)
        if np.isinf(log_prior_partition_(sorted_partition_, categories_)):
            print('Infeasible partition')
        else:
            print('Feasible partition')
            break
    unit_in_group_ = compute_unit_in_group_(sorted_partition_, categories_)
    grouped_input_data_ = group_input_(input_data_, sorted_partition_, unit_in_group_)
    input_data_re_ = ungroup_input_(grouped_input_data_, sorted_partition_, unit_in_group_)
    amp_ = torch.std(output_data_, dim=0)
    log_beta_ = torch.randn(n_vars_)
    model_ = GPRegression_(kernel=DiffusionKernel_(fourier_freq_list=[], fourier_basis_list=[]))
    model_.kernel.log_amp = torch.log(amp_)
    model_.mean.const_mean = torch.mean(output_data_, dim=0)
    model_.likelihood.log_noise_var = torch.log(amp_ / 1000.)

    start_time_ = time.time()
    fourier_freq_list_ = []
    fourier_basis_list_ = []
    for subset_ in sorted_partition_:
        adj_mat_ = strong_product(list_of_adjacency=list_of_adjacency_, beta=torch.exp(log_beta_), subset=subset_)
        deg_mat_ = torch.diag(torch.sum(adj_mat_, dim=0))
        laplacian_ = deg_mat_ - adj_mat_
        fourier_freq_, fourier_basis_ = torch.symeig(laplacian_, eigenvectors=True)
        fourier_freq_list_.append(fourier_freq_)
        fourier_basis_list_.append(fourier_basis_)
    print('\ninit elapsed time : %f' % (time.time() - start_time_))

    start_time_ = time.time()
    print('%d variables' % n_vars_)
    bar_ = progressbar.ProgressBar(max_value=n_vars_)
    for e_ in range(n_vars_):
        bar_.update(e_)
        log_beta_, fourier_freq_list_, fourier_basis_list_ = slice_edgeweight(model_, input_data_, output_data_, categories_, list_of_adjacency_, log_beta_, sorted_partition_, fourier_freq_list_, fourier_basis_list_, ind=e_)
    print('\n%f' % (time.time() - start_time_))
