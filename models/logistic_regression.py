"""Logistic regression (Bayes Rules! ch. 13, with a Normal prior on the slopes as an L2 penalty, ch. 9): the chance of a
number from its short, medium and long-run frequency, how long it has been missing and whether it came up last draw.
Refitted every 26 draws on the latest 400 draws, starting from the previous fit."""

import math

import model_tools as mt

follow = mt.Follower()
last_fit = {}


def fit(tr):
    rows, ys = follow.training()
    std = mt.Standardiser(rows)
    x = [[1.0] + std(r) for r in rows]
    start = last_fit.get(follow.generation) or [math.log(tr.p / (1 - tr.p))] + [0.0] * (len(x[0]) - 1)
    beta = mt.fit_logistic(x, ys, start)
    last_fit.clear()
    last_fit[follow.generation] = beta
    return std, beta


def predict(past, n_balls):
    tr = follow(past, n_balls)
    if tr.t < 60:
        return [1.0] * n_balls
    std, beta = follow.cached("fit", 26, lambda: fit(tr))
    out = []
    for i in range(1, n_balls + 1):
        eta = sum(b * v for b, v in zip(beta, [1.0] + std(tr.features(tr.t, i))))
        out.append(1 / (1 + math.exp(-eta)))
    return out
