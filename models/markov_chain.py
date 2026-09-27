"""Markov chain (a first-order chain over draws): the chance of each number given the numbers in the previous draw,
learned from every past pair of consecutive draws and smoothed toward the fair chance."""

import model_tools as mt

ALPHA = 10.0
follow = mt.Follower()


def predict(past, n_balls):
    tr = follow(past, n_balls)
    if not tr.draws:
        return [1.0] * n_balls
    prev = tr.draws[-1]
    return [sum((tr.trans[j][i] + ALPHA * tr.p) / (tr.nprev[j] + ALPHA) for j in prev) / mt.K
            for i in range(1, n_balls + 1)]
