"""Spatial gap term of the engine."""

import numpy as np


class SpatialGapAnalyzer:
    """Spatial gap term. A sorted ticket $x_1 < \\dots < x_k$ has $k+1$ gaps $g_0 = x_1 - 1$, $g_i = x_{i+1} -
    x_i - 1$, $g_k = N - x_k$ summing to $G = N - k$. With $q = g / G$ the term is $-\\mathrm{KL}(q \\| U_{k+1})
    = -\\sum_i q_i \\log((k+1) q_i)$: 0 for an even spread, negative when the numbers cluster. Gap variance and
    the min/max ratio are diagnostics only."""

    def __init__(self, n_balls, k):
        self.n, self.k = n_balls, k

    def gaps(self, cands):
        c = cands.astype(np.int64)
        return np.concatenate([c[:, :1] - 1, c[:, 1:] - c[:, :-1] - 1, self.n - c[:, -1:]], axis=1)

    def term(self, cands):
        q = self.gaps(cands) / (self.n - self.k)
        with np.errstate(divide="ignore", invalid="ignore"):
            contribution = np.where(q > 0, q * np.log(q * (self.k + 1)), 0.0)
        return -contribution.sum(axis=1)

    def diagnostics(self, cands):
        g = self.gaps(cands)
        return {"variance": g.var(axis=1), "min_max_ratio": g.min(axis=1) / np.maximum(g.max(axis=1), 1)}
