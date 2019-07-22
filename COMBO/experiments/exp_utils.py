import torch


def sample_init_points(n_vertices, n_points, random_seed=None):
    """

    :param n_vertices: 1D np.array
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


if __name__ == '__main__':
    print(sample_init_points([2] * 10, 5, 3))

