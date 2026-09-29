"""Overlap term. For a ticket $T$ and the last draw $D$, $o = |T \\cap D|$ is hypergeometric under a fair machine,
$P(o) = \\binom{k}{o}\\binom{N-k}{k-o} / \\binom{N}{k}$; the term is $\\log P(o)$, and 0 when there is no last draw."""

import math

import numpy as np


def log_probabilities(n_balls, k):
    total = math.comb(n_balls, k)
    return np.log(np.array([math.comb(k, o) * math.comb(n_balls - k, k - o) / total for o in range(k + 1)]))


def overlap_counts(cands, last_draw):
    return np.isin(cands, np.asarray(last_draw)).sum(axis=1)


def term(cands, last_draw, table):
    if last_draw is None:
        return np.zeros(len(cands))
    return table[overlap_counts(cands, last_draw)]
