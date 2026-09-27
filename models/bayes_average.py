"""Bayesian model averaging (Bayes' rule over models, Bayes Rules! ch. 2 and 10-11): starts trusting seven models
equally, then after every draw reweights each by how likely it said that draw was, and predicts with the weighted mix.
If one model really had an edge, its weight would grow toward 1."""

import math

import model_tools as mt

NAMES = ("fair", "hierarchical", "recent", "ewma", "overdue", "markov", "gap")


class Averager:
    def __init__(self, n_balls):
        self.tr = mt.Tracker(n_balls)
        self.logw = [0.0] * len(NAMES)
        self.kappas = {}

    def strength(self, name, pairs):
        key = (name, self.tr.t // 26)
        if key not in self.kappas:
            self.kappas[key] = mt.prior_strength(pairs, self.tr.p)
        return self.kappas[key]

    def components(self):
        tr, n, p = self.tr, self.tr.n, self.tr.p
        nums = range(1, n + 1)
        out = [[1.0] * n]
        k_all = self.strength("all", [(tr.cum[tr.t][i], tr.t) for i in nums])
        out.append([(p * k_all + tr.cum[tr.t][i]) / (k_all + tr.t) for i in nums])
        w = min(30, tr.t)
        recent = [tr.recent(i, 30) for i in nums]
        k_rec = self.strength("recent", [(c, w) for c in recent])
        out.append([(p * k_rec + c) / (k_rec + w) for c in recent])
        out.append([tr.ewma[i] + 0.05 for i in nums])
        out.append([float(tr.gap(tr.t, i)) for i in nums])
        if tr.draws:
            out.append([sum((tr.trans[j][i] + 10 * p) / (tr.nprev[j] + 10) for j in tr.draws[-1]) / mt.K for i in nums])
        else:
            out.append([1.0] * n)
        k_gap = self.strength("gap", list(zip(tr.gap_hit, tr.gap_risk)))
        hazard = [(p * k_gap + h) / (k_gap + r) for h, r in zip(tr.gap_hit, tr.gap_risk)]
        out.append([hazard[min(tr.gap(tr.t, i), mt.MAX_GAP)] for i in nums])
        return [mt.chances(c) for c in out]

    def add(self, draw):
        drawn = set(draw)
        for m, ch in enumerate(self.components()):
            self.logw[m] += mt.log_score(ch, drawn)
        self.tr.add(draw)

    def weights(self):
        top = max(self.logw)
        raw = [math.exp(v - top) for v in self.logw]
        total = sum(raw)
        return [v / total for v in raw]


state = {"avg": None}


def predict(past, n_balls):
    avg = state["avg"]
    if (avg is None or avg.tr.n != n_balls or avg.tr.t > len(past)
            or (avg.tr.t and (past[0] != avg.tr.draws[0] or past[avg.tr.t - 1] != avg.tr.draws[avg.tr.t - 1]))):
        avg = state["avg"] = Averager(n_balls)
    for draw in past[avg.tr.t:]:
        avg.add(draw)
    comps, w = avg.components(), avg.weights()
    return [sum(wm * c[i] for wm, c in zip(w, comps)) for i in range(n_balls)]
