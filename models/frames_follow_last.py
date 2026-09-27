"""Frame switching: look at the draws through eight lenses (fair, frequency, hot 30, fading hot, overdue, Markov, gap
hazard, repeat last) and use whichever lens caught the most numbers in the previous draw."""

import model_tools as mt

state = {}


def predict(past, n_balls):
    fs = mt.follow_frames(state, past, n_balls)
    frames = fs.frames()
    if not fs.hits:
        return frames[0]
    totals = mt.frame_totals(fs, 1)
    return frames[max(range(len(frames)), key=lambda f: totals[f])]
