"""Frame switching the other way round: use the lens that caught the fewest numbers over the last 10 draws, on the idea
that a lens that has been missing is due."""

import model_tools as mt

state = {}


def predict(past, n_balls):
    fs = mt.follow_frames(state, past, n_balls)
    frames = fs.frames()
    if not fs.hits:
        return frames[0]
    totals = mt.frame_totals(fs, 10)
    return frames[min(range(len(frames)), key=lambda f: totals[f])]
