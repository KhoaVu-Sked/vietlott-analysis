"""Entropy term of the engine."""

import math

import numpy as np


class InformationEntropyScorer:
    """Entropy term. Four Shannon entropies $H = -\\sum_b p_b \\log p_b$ of the ticket's numbers: over bins of
    ten ($\\lceil N/10 \\rceil$ bins), residues mod 3, residues mod 5 and last digits, each divided by its
    maximum $\\log \\min(k, \\text{bins})$ and averaged into a score in $[0, 1]$. The term is $\\log
    \\max(\\text{score}, 10^{-9})$."""

    def __init__(self, n_balls, k):
        self.n, self.k = n_balls, k
        self.decades = math.ceil(n_balls / 10)

    def _entropy(self, labels, bins):
        p = np.stack([(labels == b).sum(axis=1) for b in range(bins)], axis=1) / self.k
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        return h / math.log(min(self.k, bins))

    def score(self, cands):
        c = cands.astype(np.int64)
        parts = [self._entropy((c - 1) // 10, self.decades), self._entropy(c % 3, 3), self._entropy(c % 5, 5),
                 self._entropy(c % 10, 10)]
        return np.mean(parts, axis=0)

    def term(self, cands):
        return np.log(np.maximum(self.score(cands), 1e-9))
