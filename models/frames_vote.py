"""Consensus of the eight lenses: every lens picks its 6 numbers and each number scores one vote per lens that picked it."""

import model_tools as mt

state = {}


def predict(past, n_balls):
    fs = mt.follow_frames(state, past, n_balls)
    votes = [1.0] * n_balls
    for f, ch in enumerate(fs.frames()):
        for i in fs.picks(ch, f):
            votes[i - 1] += 1
    return votes
