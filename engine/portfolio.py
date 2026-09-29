"""Greedy portfolio picker of the engine."""

import numpy as np


class LotteryWheelingOptimizer:
    """Greedy wheeling. Candidates are visited in score order; the best is taken, then each next candidate whose
    overlap $|T_a \\cap T_b|$ with every chosen ticket is at most the cap (default $k - 2$), until $m$ tickets
    are chosen or the candidates run out. A duplicate of a chosen ticket overlaps it in $k$ numbers and is
    skipped by the same rule."""

    def __init__(self, k, max_overlap=None):
        self.k = k
        self.max_overlap = k - 2 if max_overlap is None else max_overlap

    def select(self, cands, scores, m):
        chosen = []
        for idx in np.argsort(-np.asarray(scores), kind="stable"):
            row = cands[idx]
            if all(int(np.isin(row, c).sum()) <= self.max_overlap for c in chosen):
                chosen.append(row)
                if len(chosen) == m:
                    break
        return [sorted(int(x) for x in row) for row in chosen]
