"""Memorylessness test (Bernoulli trials: the wait until a number returns is geometric, so its chance should not depend
on how long it has been missing). Learns the chance of returning after each gap length from all past draws, pooled
toward the fair chance with a Beta prior whose strength the data choose, and favours the gaps that paid off most."""

import model_tools as mt

follow = mt.Follower()


def predict(past, n_balls):
    tr = follow(past, n_balls)
    kappa = follow.cached("kappa", 26, lambda: mt.prior_strength(list(zip(tr.gap_hit, tr.gap_risk)), tr.p))
    hazard = [(tr.p * kappa + h) / (kappa + r) for h, r in zip(tr.gap_hit, tr.gap_risk)]
    return [hazard[min(tr.gap(tr.t, i), mt.MAX_GAP)] for i in range(1, n_balls + 1)]
