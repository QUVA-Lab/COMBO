import numpy as np

from simanneal import Annealer

from GraphDecompositionBO.acquisition.acquisition_functions import expected_improvement
from GraphDecompositionBO.acquisition.acquisition_utils import acquisition_expectation
from GraphDecompositionBO.acquisition.acquisition_optimizers.graph_utils import neighbors


N_EVAL = 100


class GraphSimulatedAnnealing(Annealer):

    def __init__(self, initial_state, inference_samples, partition_samples, edge_mat_samples, n_vertex,
                 acquisition_func=expected_improvement, reference=None):
        """

        :param initial_state: 1D Tensor
        :param inference_samples:
        :param partition_samples:
        :param edge_mat_samples:
        :param n_vertex:
        :param acquisition_func:
        :param reference:
        """
        super(GraphSimulatedAnnealing, self).__init__(initial_state)
        self.inference_samples = inference_samples
        self.partition_samples = partition_samples
        self.edge_mat_samples = edge_mat_samples
        self.n_vertex = n_vertex
        self.acquisition_func = acquisition_func
        self.reference = reference
        self.state_history = []
        self.eval_history = []

    def move(self):
        nbds = neighbors(self.state, self.partition_samples, self.edge_mat_samples, self.n_vertex, uniquely=False)
        self.state = nbds[np.random.randint(0, nbds.size(0))]

    def energy(self):
        evaluation = acquisition_expectation(self.state, self.inference_samples, self.acquisition_func, self.reference).item()
        self.state_history.append(self.state.clone())
        self.eval_history.append(evaluation)
        return evaluation


def simulated_annealing(x_init, inference_samples, partition_samples, edge_mat_samples, n_vertex,
                        acquisition_func, reference=None):
    """

    :param x_init:
    :param inference_samples:
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertex:
    :param acquisition_func:
    :param reference:
    :return:
    """
    sa_runner = GraphSimulatedAnnealing(x_init, inference_samples, partition_samples, edge_mat_samples, n_vertex,
                                        acquisition_func, reference)
    # configuration equivalent to that of BOCS' SA implementation
    sa_runner.Tmax = 1.0
    sa_runner.Tmin = 0.8 ** N_EVAL
    sa_runner.steps = N_EVAL
    sa_runner.anneal()

    max_ind = np.argmax(sa_runner.eval_history)
    return sa_runner.state_history[max_ind].clone(), sa_runner.eval_history[max_ind]
