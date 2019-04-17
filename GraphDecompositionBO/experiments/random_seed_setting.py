

import numpy as np


SEED_STR_LIST = ['2019NeurIPS_SYNTHETICBINARY', '2019NeurIPS_MAXSAT', ]


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