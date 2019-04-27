import sys

import numpy as np

from simanneal import Annealer

from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement
from GraphDecompositionBO.acquisition.acquisition_marginalization import acquisition_expectation
from GraphDecompositionBO.acquisition.acquisition_optimizers.graph_utils import neighbors


class GraphSimulatedAnnealing(Annealer):

    def __init__(self, initial_state, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                 acquisition_func=expected_improvement, reference=None):
        """

        :param initial_state: 1D Tensor
        :param inference_samples:
        :param partition_samples:
        :param edge_mat_samples:
        :param n_vertices:
        :param acquisition_func:
        :param reference:
        """
        super(GraphSimulatedAnnealing, self).__init__(initial_state)
        self.inference_samples = inference_samples
        self.partition_samples = partition_samples
        self.edge_mat_samples = edge_mat_samples
        self.n_vertices = n_vertices
        self.acquisition_func = acquisition_func
        self.reference = reference
        self.state_history = []
        self.eval_history = []

    def move(self):
        nbds = neighbors(self.state, self.partition_samples, self.edge_mat_samples, self.n_vertices, uniquely=False)
        self.state = nbds[np.random.randint(0, nbds.size(0))]

    def energy(self):
        # anneal() minimize
        evaluation = -acquisition_expectation(self.state, self.inference_samples, self.acquisition_func, self.reference).item()
        self.state_history.append(self.state.clone())
        self.eval_history.append(evaluation)
        return evaluation

    # To overwrite unnecessary printing
    # def update(self, *args, **kwargs):
    #     pass


def simulated_annealing(x_init, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                        acquisition_func, reference=None):
    """
    Note that Annealer.anneal() MINIMIZES an objective.
    :param x_init:
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices:
    :param acquisition_func:
    :param reference:
    :return: 1D Tensor, numeric(float)
    """
    sa_runner = GraphSimulatedAnnealing(x_init, inference_samples, partition_samples, edge_mat_samples, n_vertices,
                                        acquisition_func, reference)
    sa_param = sa_runner.auto(minutes=15.0/60.0, steps=10)
    sa_runner.steps = sa_param['steps']
    sa_runner.Tmax = sa_param['tmax']
    sa_runner.Tmin = sa_param['tmin']
    opt_state, opt_eval = sa_runner.anneal()
    sys.stdout.flush()

    # Annealer.anneal() MINinimzes an objective but acqusition functions should be MAXimized.
    return opt_state, -opt_eval
