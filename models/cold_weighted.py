"""Cold numbers with a learned weight. Numbers that have come up least since the first draw are treated as due, and a
weight from 0 to 100 says how much to trust that. After every draw the model checks its list of the 12 coldest numbers:
more of them drawn than chance gives raises the weight, fewer lowers it. The ticket leans on the cold numbers as
strongly as the weight says, with the last draw's numbers pushed back and in a typical shape (see model_tools)."""

import model_tools as mt

state = {}


def learner_for(past, n_balls, **params):
    ln = state.get("learner")
    if (ln is None or ln.tr.n != n_balls or ln.tr.t > len(past) or params != state.get("params")
            or (ln.tr.t and (past[0] != ln.tr.draws[0] or past[ln.tr.t - 1] != ln.tr.draws[ln.tr.t - 1]))):
        ln = state["learner"] = mt.ColdLearner(n_balls, len(past[0]) if past else 6, **params)
        state["params"] = params
    for d in past[ln.tr.t:]:
        ln.add(d)
    return ln


def predict(past, n_balls):
    if len(past) < 30:
        return [1.0] * n_balls
    picks = set(learner_for(past, n_balls).tickets(past)[0])
    return [1.0 + (1e-9 if i in picks else 0.0) for i in range(1, n_balls + 1)]
