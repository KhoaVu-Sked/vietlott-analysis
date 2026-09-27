"""Bayesian model averaging with forgetting (dynamic model averaging): like bayes_average, but each past draw's evidence
fades by 5% per draw, so the weights can move to whichever of the eight lenses has been working lately."""

import math

import model_tools as mt

FADE = 0.95
state = {}


def predict(past, n_balls):
    fs = mt.follow_frames(state, past, n_balls)
    frames = fs.frames()
    logw = [0.0] * len(frames)
    for scores in fs.scores:
        logw = [FADE * w + s for w, s in zip(logw, scores)]
    top = max(logw)
    weights = [math.exp(w - top) for w in logw]
    total = sum(weights)
    return [sum(w / total * ch[i] for w, ch in zip(weights, frames)) for i in range(n_balls)]
