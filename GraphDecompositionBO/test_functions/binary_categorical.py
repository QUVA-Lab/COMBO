import time
import itertools
import numpy as np

import torch
from GraphDecompositionBO.test_functions.aero_struct.aerostruct_lin_model import kl_decoupled_models
from GraphDecompositionBO.test_functions.experiment_configuration import ISING_GRID_H, ISING_GRID_W
from GraphDecompositionBO.test_functions.experiment_configuration import ISING_N_EDGES, CONTAMINATION_N_STAGES, AEROSTRUCTURAL_N_COUPLINGS
from GraphDecompositionBO.test_functions.experiment_configuration import sample_init_points, generate_ising_interaction, generate_contamination_dynamics


AERO_STRUCT_EVAL_UPPER_LIMIT = 1000


def spin_covariance(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    density = np.zeros(spin_cfgs.shape[0])
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        density[i] = np.exp(log_interaction_energy)
    interaction_partition = np.sum(density)
    density = density / interaction_partition

    covariance = spin_cfgs.T.dot(spin_cfgs * density.reshape((-1, 1)))
    return covariance, interaction_partition


def partition(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    interaction_partition = 0
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        interaction_partition += np.exp(log_interaction_energy)

    return interaction_partition


def ising_dense(interaction_original, interaction_sparsified, covariance, partition_original, partition_sparsified, grid_h):
    diff_horizontal = interaction_original[0] - interaction_sparsified[0]
    diff_vertical = interaction_original[1] - interaction_sparsified[1]

    kld = 0
    n_spin = covariance.shape[0]
    for i in range(n_spin):
        i_h, i_v = int(i / grid_h), int(i % grid_h)
        for j in range(i, n_spin):
            j_h, j_v = int(j / grid_h), int(j % grid_h)
            if i_h == j_h and abs(i_v - j_v) == 1:
                kld += diff_horizontal[i_h, min(i_v, j_v)] * covariance[i, j]
            elif abs(i_h - j_h) == 1 and i_v == j_v:
                kld += diff_vertical[min(i_h, j_h), i_v] * covariance[i, j]

    return kld * 2 + np.log(partition_sparsified / partition_original)


def _bocs_consistency_mapping(x):
    '''
    This is for the comparison with BOCS implementation
    :param x:
    :return:
    '''
    horizontal_ind = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]
    vertical_ind = sorted([elm for elm in range(24) if elm not in horizontal_ind])
    return x[horizontal_ind].reshape((ISING_GRID_H, ISING_GRID_W - 1)), x[vertical_ind].reshape((ISING_GRID_H - 1, ISING_GRID_W))


class Ising1(object):
    """
    Ising Sparsification Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = [2] * ISING_N_EDGES
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points([2] * ISING_N_EDGES, 20 - self.suggested_init.size(0), random_seed_pair[1]).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)
        interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, random_seed_pair[0])
        self.interaction = interaction[0].numpy(), interaction[1].numpy()
        self.covariance, self.partition_original = spin_covariance(self.interaction, (ISING_GRID_H, ISING_GRID_W))

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        x_h, x_v = _bocs_consistency_mapping(x.numpy())
        interaction_sparsified = x_h * self.interaction[0], x_v * self.interaction[1]
        partition_sparsified = partition(interaction_sparsified, (ISING_GRID_H, ISING_GRID_W))
        evaluation = ising_dense(interaction_sparsified=interaction_sparsified, interaction_original=self.interaction,
                                  covariance=self.covariance, partition_sparsified=partition_sparsified, partition_original=self.partition_original)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


class Ising2(object):
    """
    Ising Sparsification Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = [2] * ISING_N_EDGES
        self.suggested_init = torch.empty(0).long()
        binary_random_init = sample_init_points([2] * ISING_N_EDGES, 20 - self.suggested_init.size(0), random_seed_pair[1])
        random_init = torch.empty(20 - self.suggested_init.size(0), len(self.n_vertices)).long()
        for i in range(random_init.size(0)):
            random_init[i] = self._from_binary(binary_random_init[i])
        self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)
        interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, random_seed_pair[0])
        self.interaction = interaction[0].numpy(), interaction[1].numpy()
        self.covariance, self.partition_original = spin_covariance(self.interaction, (ISING_GRID_H, ISING_GRID_W))

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        x_h, x_v = _bocs_consistency_mapping(x.numpy())
        interaction_sparsified = x_h * self.interaction[0], x_v * self.interaction[1]
        partition_sparsified = partition(interaction_sparsified, (ISING_GRID_H, ISING_GRID_W))
        evaluation = ising_dense(interaction_sparsified=interaction_sparsified, interaction_original=self.interaction,
                                  covariance=self.covariance, partition_sparsified=partition_sparsified, partition_original=self.partition_original)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()

    @staticmethod
    def _from_binary(x):
        assert x.dim() == 1
        x = x.reshape(-1, 3).t().clone()
        x[0] *= 4
        x[1] *= 2
        return torch.sum(x, dim=0)

    @staticmethod
    def _to_binary(x):
        assert x.dim() == 1
        assert (x < 8).all()
        return torch.stack([x / 4, (x % 4) / 2, (x % 4) % 2]).t().reshape(-1)


def _contamination(x, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == CONTAMINATION_N_STAGES

    rho = 1.0
    n_simulations = 100

    Z = np.zeros((x.size, n_simulations))
    Z[0] = lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    for i in range(1, CONTAMINATION_N_STAGES):
        Z[i] = lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1]) + (1.0 - gammas[i] * x[i]) * Z[i - 1]

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)


class Contamination1(object):
    """
    Contamination Control Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = [2] * CONTAMINATION_N_STAGES
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1])], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(random_seed_pair[0])

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _contamination(x=(x.cpu() if x.is_cuda else x).numpy(), cost=np.ones(x.numel()), init_Z=self.init_Z, lambdas=self.lambdas, gammas=self.gammas, U=0.1, epsilon=0.05)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


class AeroStruct1(object):
    """
    Aero Structural Multi-Component Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed=None):
        self.lamda = lamda
        self.n_vertices = [2] * AEROSTRUCTURAL_N_COUPLINGS
        self.suggested_init = torch.empty(0).long()
        binary_random_init = sample_init_points([2] * AEROSTRUCTURAL_N_COUPLINGS, 20 - self.suggested_init.size(0), random_seed)
        random_init = binary_random_init
        self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4) if random_seed is not None else 'None'
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = min(kl_decoupled_models((x.cpu() if x.is_cuda else x).numpy()), AERO_STRUCT_EVAL_UPPER_LIMIT)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


class AeroStruct2(object):
    """
    Aero Structural Multi-Component Problem with a arbitrary grouped graph
    """
    def __init__(self, lamda, random_seed=None):
        self.lamda = lamda
        self.n_vertices = [4] * 10 + [2]
        self.suggested_init = torch.empty(0).long()
        binary_random_init = sample_init_points([2] * AEROSTRUCTURAL_N_COUPLINGS, 20 - self.suggested_init.size(0), random_seed)
        random_init = torch.empty(20 - self.suggested_init.size(0), len(self.n_vertices)).long()
        for i in range(random_init.size(0)):
            random_init[i] = self._from_binary(binary_random_init[i])
        self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4) if random_seed is not None else 'None'
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        assert (x[:-1] < 4).all() and x[-1] < 2
        x = self._to_binary(x)
        evaluation = min(kl_decoupled_models((x.cpu() if x.is_cuda else x).numpy()), AERO_STRUCT_EVAL_UPPER_LIMIT)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()

    @staticmethod
    def _from_binary(x):
        assert x.dim() == 1
        x = torch.cat([x, x.new_zeros(1)]).reshape(-1, 2).t()
        x[1] *= 2
        return torch.sum(x, dim=0)

    @staticmethod
    def _to_binary(x):
        assert x.dim() == 1
        binarized_x = torch.stack([x % 2, x / 2]).t().reshape(-1)
        assert binarized_x[-1] == 0
        x = binarized_x[:-1]
        return x


class AeroStruct3(object):
    """
    Aero Structural Multi-Component Problem with a arbitrary grouped graph
    """
    def __init__(self, lamda, random_seed=None):
        self.lamda = lamda
        self.n_vertices = [8] * 7
        self.suggested_init = torch.empty(0).long()
        binary_random_init = sample_init_points([2] * AEROSTRUCTURAL_N_COUPLINGS, 20 - self.suggested_init.size(0), random_seed)
        random_init = torch.empty(20 - self.suggested_init.size(0), len(self.n_vertices)).long()
        for i in range(random_init.size(0)):
            random_init[i] = self._from_binary(binary_random_init[i])
        self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
        self.adjacency_mat = []
        self.fourier_coef = []
        self.fourier_basis = []
        self.random_seed_info = str(random_seed).zfill(4) if random_seed is not None else 'None'
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_coef.append(eigval)
            self.fourier_basis.append(eigvec)

    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        assert (x < 8).all()
        x = self._to_binary(x)
        evaluation = min(kl_decoupled_models((x.cpu() if x.is_cuda else x).numpy()), AERO_STRUCT_EVAL_UPPER_LIMIT)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()

    @staticmethod
    def _from_binary(x):
        assert x.dim() == 1
        x = x.reshape(-1, 3).t().clone()
        x[0] *= 4
        x[1] *= 2
        return torch.sum(x, dim=0)

    @staticmethod
    def _to_binary(x):
        assert x.dim() == 1
        assert (x < 8).all()
        return torch.stack([x / 4, (x % 4) / 2, (x % 4) % 2]).t().reshape(-1)


if __name__ == '__main__':
    pass
    # from GraphDecompositionBO.test_functions.experiment_configuration import generate_random_seed_aerostruct, generate_random_seed_pair_ising, generate_random_seed_pair_contamination, generate_ising_interaction

    # random_seed_pairs = generate_random_seed_pair_ising()
    # lamda = 0.00
    # for case_seed in sorted(random_seed_pairs.keys()):
    #     init_seed_list = sorted(random_seed_pairs[case_seed])
    #     for init_seed in init_seed_list:
    #         print('Ising' + ('-' * 20) + str(case_seed).zfill(4) + ',' + str(init_seed).zfill(4) + ('-' * 20))
    #         evaluator = Ising1(lamda=lamda, random_seed_pair=(case_seed, init_seed))
    #         evaluations = []
    #         for i in range(evaluator.suggested_init.size(0)):
    #             evaluations.append(evaluator.evaluate(evaluator.suggested_init[i]).item())
    #         print(['%8.4f' % elm for elm in evaluations])

    # random_seed_pairs = generate_random_seed_pair_contamination()
    # lamda = 0.00
    # for case_seed in sorted(random_seed_pairs.keys()):
    #     init_seed_list = sorted(random_seed_pairs[case_seed])
    #     for init_seed in init_seed_list:
    #         print('Contamination  '  + str(case_seed).zfill(4) + ',' + str(init_seed).zfill(4))
    #         evaluator = Contamination1(lamda=lamda, random_seed_pair=(case_seed, init_seed))
    #         evaluations = []
    #         for i in range(evaluator.suggested_init.size(0)):
    #             evaluations.append(evaluator.evaluate(evaluator.suggested_init[i]).item())
    #         print(['%8.4f' % elm for elm in evaluations])

    # random_seeds = generate_random_seed_aerostruct()
    # print(random_seeds)
    # lamda = 0.00
    # eval_types = [AeroStruct1]
    # for random_seed in sorted(random_seeds):
    #     print(('-' * 20) + str(random_seed).zfill(4) + ('-' * 20))
    #     for i in range(len(eval_types)):
    #         evaluator = eval_types[i](lamda=lamda, random_seed=random_seed)
    #         evaluations = []
    #         for i in range(evaluator.suggested_init.size(0)):
    #             evaluations.append(evaluator.evaluate(evaluator.suggested_init[i]).item())
    #         print(['%8.4f' % elm for elm in evaluations])
    #         print(min(evaluations))