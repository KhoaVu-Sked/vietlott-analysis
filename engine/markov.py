"""Markov term of the engine."""

import numpy as np


class MarkovTransitionModel:
    """Markov transition model. $M_{ij} = (n_{ij} + \\epsilon) / \\sum_j (n_{ij} + \\epsilon)$ with $n_{ij}$ the
    times number $j$ was drawn right after a draw holding $i$. Given the last draw $D$, $q_j =
    \\frac{1}{|D|}\\sum_{i \\in D} M_{ij}$ and a ticket's term is $\\sum_{j \\in T} \\log(N q_j)$, 0 before any
    transition is known."""

    def __init__(self, n_balls, k, eps=1e-6):
        self.n, self.k, self.eps = n_balls, k, eps
        self.counts = np.zeros((n_balls, n_balls))
        self.last = None

    def update(self, draw):
        d = np.asarray(draw) - 1
        if self.last is not None:
            self.counts[np.ix_(self.last, d)] += 1.0
        self.last = d

    def matrix(self):
        m = self.counts + self.eps
        return m / m.sum(axis=1, keepdims=True)

    def log_terms(self):
        if self.last is None:
            return np.zeros(self.n)
        return np.log(self.n * self.matrix()[self.last].mean(axis=0))
