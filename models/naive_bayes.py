"""Naive Bayes classification (Bayes Rules! ch. 14): the same signals as the logistic regression, each assumed Normal and
independent given whether the number is drawn, combined with Bayes' rule. Refitted every 26 draws on the latest 400."""

import math

import model_tools as mt

follow = mt.Follower()


def fit(tr):
    rows, ys = follow.training()
    stats = {}
    for cls in (0, 1):
        sub = [r for r, y in zip(rows, ys) if y == cls]
        mus = [sum(r[a] for r in sub) / len(sub) for a in range(6)]
        sds = [math.sqrt(sum((r[a] - mus[a]) ** 2 for r in sub) / len(sub)) or 1e-3 for a in range(6)]
        stats[cls] = (mus, sds, sum(r[5] for r in sub) / len(sub))
    return sum(ys) / len(ys), stats


def predict(past, n_balls):
    tr = follow(past, n_balls)
    if tr.t < 60:
        return [1.0] * n_balls
    prior1, stats = follow.cached("fit", 26, lambda: fit(tr))
    out = []
    for i in range(1, n_balls + 1):
        x = tr.features(tr.t, i)
        logp = {}
        for cls, prior in ((1, prior1), (0, 1 - prior1)):
            mus, sds, last_rate = stats[cls]
            lp = math.log(prior) + sum(-0.5 * ((x[a] - mus[a]) / sds[a]) ** 2 - math.log(sds[a]) for a in range(5))
            logp[cls] = lp + math.log(min(max(last_rate if x[5] else 1 - last_rate, 1e-9), 1 - 1e-9))
        out.append(1 / (1 + math.exp(max(-700, min(700, logp[0] - logp[1])))))
    return out
