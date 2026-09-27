"""Partial pooling (Bayes Rules! ch. 15-16, conjugacy from ch. 3-4): each number's chance is its own record pulled toward
the fair 6/N. How hard it is pulled is chosen by the data themselves, through the Beta-Binomial marginal likelihood."""

import model_tools as mt

follow = mt.Follower()


def predict(past, n_balls):
    tr = follow(past, n_balls)
    kappa = follow.cached("kappa", 26, lambda: mt.prior_strength(
        [(tr.cum[tr.t][i], tr.t) for i in range(1, n_balls + 1)], tr.p))
    return [(tr.p * kappa + tr.cum[tr.t][i]) / (kappa + tr.t) for i in range(1, n_balls + 1)]
