import itertools
import numpy as np

import torch
from COMBO.experiments.test_functions.experiment_configuration import ISING_GRID_H, ISING_GRID_W, \
    ISING_N_EDGES, CONTAMINATION_N_STAGES
from COMBO.experiments.test_functions.experiment_configuration import sample_init_points, \
    generate_ising_interaction, generate_contamination_dynamics


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


def log_partition(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    log_interaction_energy_list = []
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        log_interaction_energy_list.append(log_interaction_energy)

    log_interaction_energy_list = np.array(log_interaction_energy_list)
    max_log_interaction_energy = np.max(log_interaction_energy_list)
    interaction_partition = np.sum(np.exp(log_interaction_energy_list - max_log_interaction_energy))

    return np.log(interaction_partition) + max_log_interaction_energy


def ising_dense(interaction_original, interaction_sparsified, covariance, log_partition_original, log_partition_sparsified):
    diff_horizontal = interaction_original[0] - interaction_sparsified[0]
    diff_vertical = interaction_original[1] - interaction_sparsified[1]

    kld = 0
    n_spin = covariance.shape[0]
    for i in range(n_spin):
        i_h, i_v = int(i / ISING_GRID_H), int(i % ISING_GRID_H)
        for j in range(i, n_spin):
            j_h, j_v = int(j / ISING_GRID_H), int(j % ISING_GRID_H)
            if i_h == j_h and abs(i_v - j_v) == 1:
                kld += diff_horizontal[i_h, min(i_v, j_v)] * covariance[i, j]
            elif abs(i_h - j_h) == 1 and i_v == j_v:
                kld += diff_vertical[min(i_h, j_h), i_v] * covariance[i, j]

    return kld * 2 + log_partition_sparsified - log_partition_original


def _bocs_consistency_mapping(x):
    """
    This is for the comparison with BOCS implementation
    :param x:
    :return:
    """
    horizontal_ind = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]
    vertical_ind = sorted([elm for elm in range(24) if elm not in horizontal_ind])
    return x[horizontal_ind].reshape((ISING_GRID_H, ISING_GRID_W - 1)), x[vertical_ind].reshape((ISING_GRID_H - 1, ISING_GRID_W))


class Ising(object):
    """
    Ising Sparsification Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = np.array([2] * ISING_N_EDGES)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1]).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, random_seed_pair[0])
        self.interaction = interaction[0].numpy(), interaction[1].numpy()
        self.covariance, self.partition_original = spin_covariance(self.interaction, (ISING_GRID_H, ISING_GRID_W))

    def evaluate(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)

    def _evaluate_single(self, x):
        assert x.dim() == 1
        x_h, x_v = _bocs_consistency_mapping(x.numpy())
        interaction_sparsified = x_h * self.interaction[0], x_v * self.interaction[1]
        log_partition_sparsified = log_partition(interaction_sparsified, (ISING_GRID_H, ISING_GRID_W))
        evaluation = ising_dense(interaction_sparsified=interaction_sparsified, interaction_original=self.interaction,
                                 covariance=self.covariance, log_partition_sparsified=log_partition_sparsified,
                                 log_partition_original=np.log(self.partition_original))
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


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


class Contamination(object):
    """
    Contamination Control Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = np.array([2] * CONTAMINATION_N_STAGES)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1])], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(random_seed_pair[0])

    def evaluate(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)

    def _evaluate_single(self, x):
        assert x.dim() == 1
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _contamination(x=(x.cpu() if x.is_cuda else x).numpy(), cost=np.ones(x.numel()), init_Z=self.init_Z, lambdas=self.lambdas, gammas=self.gammas, U=0.1, epsilon=0.05)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()


if __name__ == '__main__':
    interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, np.random.randint(0, 10000))
    interaction = interaction[0].numpy(), interaction[1].numpy()
    grid_shape = (ISING_GRID_H, ISING_GRID_W)
    print(np.log(partition(interaction, grid_shape)))
    print(log_partition(interaction, grid_shape))
