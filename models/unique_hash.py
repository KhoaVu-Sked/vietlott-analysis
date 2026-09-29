"""Every universe has its own hash: its draws never repeat a combination. The next draw is then equally likely to be
any combination not drawn yet, so a number's chance is the share of those unused combinations that hold it. A number
that was in fewer past draws sits in more unused combinations and gains, by a few parts in a million."""

import math

state = {}


def predict(past, n_balls):
    k = len(past[0]) if past else 6
    t = state.get("t", 0)
    if state.get("n") != n_balls or t > len(past) or (t and (past[0] != state["first"] or past[t - 1] != state["last"])):
        state.clear()
        state.update(n=n_balls, t=0, seen=set(), inside=[0] * (n_balls + 1))
    for d in past[state["t"]:]:
        combo = tuple(sorted(d))
        if combo not in state["seen"]:
            state["seen"].add(combo)
            for i in combo:
                state["inside"][i] += 1
    if past:
        state.update(t=len(past), first=past[0], last=past[-1])
    unused = math.comb(n_balls, k) - len(state["seen"])
    holding = math.comb(n_balls - 1, k - 1)
    return [(holding - state["inside"][i]) / unused for i in range(1, n_balls + 1)]
