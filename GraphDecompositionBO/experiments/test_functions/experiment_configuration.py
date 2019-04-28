import os
import numpy as np
import scipy.io as sio

import torch


ISING_GRID_H = 4
ISING_GRID_W = 4
ISING_N_EDGES = 24

CONTAMINATION_N_STAGES = 25

AEROSTRUCTURAL_N_COUPLINGS = 21


SEED_STR_LIST = ['2019ICML_ISING', '2019ICML_CONTAMINATION', '2019ICML_AEROSTRUCTURAL', '2019ICML_TRAVELPLAN', '2019ICML_PESTCONTROL', '2019ICML_CENTROID']


def generate_random_seed_pair_ising():
    return _generate_random_seed_pair('2019ICML_ISING', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_pair_contamination():
    return _generate_random_seed_pair('2019ICML_CONTAMINATION', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_aerostruct():
    return _generate_random_seed('2019ICML_AEROSTRUCTURAL', n_init_point_seed=10)


def generate_random_seed_pair_travelplan():
    return _generate_random_seed_pair('2019ICML_TRAVELPLAN', n_test_case_seed=5, n_init_point_seed=5)


def generate_random_seed_pestcontrol():
    return _generate_random_seed('2019ICML_PESTCONTROL', n_init_point_seed=25)


def generate_random_seed_pair_centroid():
    return _generate_random_seed_pair('2019ICML_CENTROID', n_test_case_seed=5, n_init_point_seed=5)


def _generate_random_seed(seed_str, n_init_point_seed=10):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))


def _generate_random_seed_pair(seed_str, n_test_case_seed=5, n_init_point_seed=5):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_test_case_seed):
        result[rng_state.randint(0, 10000)] = list(rng_state.randint(0, 10000, (n_init_point_seed, )))
    return result


def _convert_random_data_to_matfile():
    MAT_SAVE_DIR = '/home/coh1/git_repositories/BOCS/random_data'

    seed_str = '2019ICML_ISING'
    seed_pair = _generate_random_seed_pair(seed_str)
    for test_case_seed, init_point_seed_list in seed_pair.iteritems():
        for init_point_seed in init_point_seed_list:
            file_tag = str(test_case_seed).zfill(4) + '_' + str(init_point_seed).zfill(4)
            horizontal_interaction, vertical_interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, test_case_seed)
            interaction_array = interaction_dense2sparse(horizontal_interaction, vertical_interaction).numpy()
            init_points_array = sample_init_points(n_vertices=[2] * ISING_N_EDGES, n_points=20, random_seed=init_point_seed).numpy()
            filename = os.path.join(MAT_SAVE_DIR, '_'.join(['ising', file_tag]) + '.mat')
            sio.savemat(file_name=filename, mdict={'Theta': interaction_array, 'x_vals': init_points_array})

    seed_str = '2019ICML_CONTAMINATION'
    seed_pair = _generate_random_seed_pair(seed_str)
    for test_case_seed, init_point_seed_list in seed_pair.iteritems():
        for init_point_seed in init_point_seed_list:
            file_tag = str(test_case_seed).zfill(4) + '_' + str(init_point_seed).zfill(4)
            initialX, lamda, gamma = generate_contamination_dynamics(test_case_seed)
            init_points_array = sample_init_points(n_vertices=[2] * CONTAMINATION_N_STAGES, n_points=20, random_seed=init_point_seed).numpy()
            filename = os.path.join(MAT_SAVE_DIR, '_'.join(['contamination', file_tag]) + '.mat')
            sio.savemat(file_name=filename, mdict={'initialX': initialX, 'Lambda': lamda, 'Gamma': gamma, 'x_vals': init_points_array})

    seed_str = '2019ICML_AEROSTRUCTURAL'
    seed_list = _generate_random_seed(seed_str)
    for init_seed in seed_list:
        file_tag = str(init_seed).zfill(4)
        init_points_array = sample_init_points(n_vertices=[2] * AEROSTRUCTURAL_N_COUPLINGS, n_points=20, random_seed=init_seed).numpy()
        filename = os.path.join(MAT_SAVE_DIR, '_'.join(['aerostruct', file_tag]) + '.mat')
        sio.savemat(file_name=filename, mdict={'x_vals': init_points_array})


def sample_init_points(n_vertices, n_points, random_seed=None):
    """

    :param n_vertices: 1D array
    :param n_points:
    :param random_seed:
    :return:
    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat([init_points, torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1)], dim=0)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points


def generate_ising_interaction(grid_h, grid_w, random_seed=None):
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    horizontal_interaction = ((torch.randint(0, 2, (grid_h * (grid_w - 1), )) * 2 - 1).float() * (torch.rand(grid_h * (grid_w - 1)) * (5 - 0.05) + 0.05)).view(grid_h, grid_w-1)
    vertical_interaction = ((torch.randint(0, 2, ((grid_h - 1) * grid_w, )) * 2 - 1).float() * (torch.rand((grid_h - 1) * grid_w) * (5 - 0.05) + 0.05)).view(grid_h-1, grid_w)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return horizontal_interaction, vertical_interaction


def generate_contamination_dynamics(random_seed=None):
    n_stages = CONTAMINATION_N_STAGES
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(init_alpha, init_beta, size=(n_simulations,))
    lambdas = np.random.RandomState(random_seed).beta(contam_alpha, contam_beta, size=(n_stages, n_simulations))
    gammas = np.random.RandomState(random_seed).beta(restore_alpha, restore_beta, size=(n_stages, n_simulations))

    return init_Z, lambdas, gammas


def interaction_sparse2dense(bocs_representation):
    assert bocs_representation.size(0) == bocs_representation.size(1)
    grid_size = int(bocs_representation.size(0) ** 0.5)
    horizontal_interaction = torch.zeros(grid_size, grid_size-1)
    vertical_interaction = torch.zeros(grid_size-1, grid_size)
    for i in range(bocs_representation.size(0)):
        r_i = i // grid_size
        c_i = i % grid_size
        for j in range(i + 1, bocs_representation.size(1)):
            r_j = j // grid_size
            c_j = j % grid_size
            if abs(r_i - r_j) + abs(c_i - c_j) > 1:
                assert bocs_representation[i, j] == 0
            elif abs(r_i - r_j) == 1:
                vertical_interaction[min(r_i, r_j), c_i] = bocs_representation[i, j]
            else:
                horizontal_interaction[r_i, min(c_i, c_j)] = bocs_representation[i, j]
    return horizontal_interaction, vertical_interaction


def interaction_dense2sparse(horizontal_interaction, vertical_interaction):
    grid_size = horizontal_interaction.size(0)
    bocs_representation = torch.zeros(grid_size ** 2, grid_size ** 2)
    for i in range(bocs_representation.size(0)):
        r_i = i // grid_size
        c_i = i % grid_size
        for j in range(i + 1, bocs_representation.size(1)):
            r_j = j // grid_size
            c_j = j % grid_size
            if abs(r_i - r_j) + abs(c_i - c_j) > 1:
                assert bocs_representation[i, j] == 0
            elif abs(r_i - r_j) == 1:
                bocs_representation[i, j] = vertical_interaction[min(r_i, r_j), c_i]
            else:
                bocs_representation[i, j] = horizontal_interaction[r_i, min(c_i, c_j)]
    return bocs_representation + bocs_representation.t()


if __name__ == '__main__':
    _convert_random_data_to_matfile()