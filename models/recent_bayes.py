"""Hot numbers done the Bayesian way (Gamma-Poisson / Beta-Binomial conjugacy, Bayes Rules! ch. 3-5): each number's rate
over the last 30 draws, shrunk toward the fair rate by a prior whose strength the data choose."""

import model_tools as mt

WINDOW = 30
follow = mt.Follower()


def predict(past, n_balls):
    tr = follow(past, n_balls)
    w = min(WINDOW, tr.t)
    counts = [tr.recent(i, WINDOW) for i in range(1, n_balls + 1)]
    kappa = follow.cached("kappa", 26, lambda: mt.prior_strength([(c, w) for c in counts], tr.p))
    return [(tr.p * kappa + c) / (kappa + w) for c in counts]
