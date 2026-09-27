"""Exponentially weighted hot numbers: every past appearance counts, fading by 3% per draw."""

import model_tools as mt

follow = mt.Follower()


def predict(past, n_balls):
    tr = follow(past, n_balls)
    return [tr.ewma[i] + 0.05 for i in range(1, n_balls + 1)]
