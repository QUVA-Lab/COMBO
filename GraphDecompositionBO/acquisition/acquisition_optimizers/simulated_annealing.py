import math
import random
import time

import numpy as np

from simanneal import Annealer
from simanneal.anneal import round_figures

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
        evaluation = -acquisition_expectation(self.state, self.inference_samples, self.partition_samples,
                                              self.n_vertices, self.acquisition_func, self.reference).item()
        self.state_history.append(self.state.clone())
        self.eval_history.append(evaluation)
        return evaluation

    # To overwrite unnecessary printing
    def update(self, *args, **kwargs):
        pass

    # To overwrite the original auto
    def auto(self, minutes, steps=2000):
        """Explores the annealing landscape and
        estimates optimal temperature settings.

        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                self.move()
                E = self.energy()
                dE = E - prevEnergy
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.energy() - E)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration, 'updates': self.updates}


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
    sa_schedule = sa_runner.auto(minutes=15.0/60.0, steps=10)
    sa_runner.set_schedule(sa_schedule)
    opt_state, opt_eval = sa_runner.anneal()

    # Annealer.anneal() MINinimzes an objective but acqusition functions should be MAXimized.
    return opt_state, -opt_eval
