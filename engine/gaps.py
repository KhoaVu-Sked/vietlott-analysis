"""Gap-combination term of the engine."""

import math

import numpy as np

KIND_EDGES = (1, 4, 9)
KIND_RANGES = ((0, 0), (1, 3), (4, 8), (9, None))


def fair_shares(n_balls, k):
    top = n_balls - k
    polys = []
    for lo, hi in KIND_RANGES:
        p = np.zeros(top + 1, dtype=np.int64)
        hi = top if hi is None else min(hi, top)
        if lo <= hi:
            p[lo:hi + 1] = 1
        polys.append(p)
    ends = np.arange(top + 1, 0, -1, dtype=np.int64)
    shares = np.zeros(k ** 4)
    total, inner = math.comb(n_balls, k), k - 1
    for n0 in range(inner + 1):
        for n1 in range(inner + 1 - n0):
            for n2 in range(inner + 1 - n0 - n1):
                n3 = inner - n0 - n1 - n2
                ways = np.ones(1, dtype=np.int64)
                for poly, times in zip(polys, (n0, n1, n2, n3)):
                    for _ in range(times):
                        ways = np.convolve(ways, poly)[:top + 1]
                orders = math.factorial(inner) // (math.factorial(n0) * math.factorial(n1) * math.factorial(n2)
                                                   * math.factorial(n3))
                shares[((n0 * k + n1) * k + n2) * k + n3] = orders * int(ways @ ends[:len(ways)]) / total
    return shares


class SpatialGapAnalyzer:
    """Gap-combination term. The $k-1$ in-between gaps $g_i = x_{i+1} - x_i - 1$ of a sorted ticket are sorted
    into kinds: next to ($g = 0$), close ($1$-$3$), medium ($4$-$8$) and wide ($\\ge 9$); the counts $(n_0, n_1,
    n_2, n_3)$ per kind are the ticket's combination $c$. Its exact share under a fair machine is $f(c) =
    \\binom{k-1}{n_0, n_1, n_2, n_3} \\sum_S [x^S] \\prod_j P_j(x)^{n_j} \\, (N - k - S + 1) / \\binom{N}{k}$,
    with $P_j$ the generating polynomial of the gap values of kind $j$ and $N - k - S + 1$ the ways to place the
    two outer gaps. After $t$ draws, $n_t(c)$ of them of combination $c$, the term is $\\log \\frac{n_t(c) +
    \\kappa}{t f(c) + \\kappa}$ with $\\kappa = 20$, the Gamma-Poisson estimate of how much more often $c$ comes
    up than a fair machine gives: 0 while history matches, and a combination needs about $\\kappa$ expected
    occurrences before its own history counts for half, so a rare one seen once barely moves it. The old evenness
    score $\\mathrm{KL}(q \\| U_{k+1})$ of all $k+1$ gaps, $q = g / (N - k)$, stays a diagnostic with the gap
    variance and the min/max ratio."""

    def __init__(self, n_balls, k, strength=20.0):
        self.n, self.k, self.strength = n_balls, k, strength
        self.radix = np.array([k ** 3, k ** 2, k, 1], dtype=np.int64)
        self.fair = fair_shares(n_balls, k)
        self.seen = np.zeros(len(self.fair))
        self.t = 0

    def code(self, next_to, close, medium, wide):
        return int(np.dot([next_to, close, medium, wide], self.radix))

    def kinds(self, cands):
        c = cands.astype(np.int64)
        kind = np.digitize(c[:, 1:] - c[:, :-1] - 1, KIND_EDGES)
        return np.stack([(kind == j).sum(axis=1) for j in range(4)], axis=1) @ self.radix

    def update(self, draw):
        self.seen[self.kinds(np.array([sorted(draw)]))[0]] += 1
        self.t += 1

    def log_terms(self):
        return np.log((self.seen + self.strength) / (self.t * self.fair + self.strength))

    def gaps(self, cands):
        c = cands.astype(np.int64)
        return np.concatenate([c[:, :1] - 1, c[:, 1:] - c[:, :-1] - 1, self.n - c[:, -1:]], axis=1)

    def diagnostics(self, cands):
        g = self.gaps(cands)
        q = g / (self.n - self.k)
        with np.errstate(divide="ignore", invalid="ignore"):
            unevenness = np.where(q > 0, q * np.log(q * (self.k + 1)), 0.0).sum(axis=1)
        return {"variance": g.var(axis=1), "min_max_ratio": g.min(axis=1) / np.maximum(g.max(axis=1), 1),
                "unevenness": unevenness}
