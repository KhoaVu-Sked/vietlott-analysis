"""Frame switching: use whichever of the eight lenses caught the most numbers over the last 10 draws, on the idea that
the lens that fits the machine changes over time."""

import model_tools as mt

state = {}


def predict(past, n_balls):
    fs = mt.follow_frames(state, past, n_balls)
    frames = fs.frames()
    if not fs.hits:
        return frames[0]
    totals = mt.frame_totals(fs, 10)
    return frames[max(range(len(frames)), key=lambda f: totals[f])]
