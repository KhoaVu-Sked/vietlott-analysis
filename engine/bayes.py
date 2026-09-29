"""Bayes term of the engine: decayed Dirichlet counts."""

import math

import numpy as np


class BayesianDirichletModel:
    """Bayesian Dirichlet model with exponential forgetting. Pseudo-counts $a_i = \\alpha_0 + \\sum_t
    e^{-\\lambda (T - t)} \\, \\mathbb{1}[i \\in D_t]$ give the posterior mean $p_i = a_i / \\sum_j a_j$; a
    ticket's term is $\\sum_{i \\in T} \\log(N p_i)$, 0 under a fair posterior."""

    def __init__(self, n_balls, k, alpha0=1.0, decay=0.005):
        self.n, self.k, self.alpha0 = n_balls, k, alpha0
        self.fade = math.exp(-decay)
        self.counts = np.zeros(n_balls)

    def update(self, draw):
        self.counts *= self.fade
        self.counts[np.asarray(draw) - 1] += 1.0

    def probabilities(self):
        a = self.alpha0 + self.counts
        return a / a.sum() if a.sum() > 0 else np.full(self.n, 1 / self.n)

    def log_terms(self):
        return np.log(self.n * self.probabilities())
