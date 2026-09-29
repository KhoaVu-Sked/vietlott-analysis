"""Empirical validity checks on a draw history. Chi-square goodness of fit of the number counts against the discrete
uniform $U(1, N)$, $\\chi^2 = \\frac{N-1}{N-k} \\sum_i (c_i - e)^2 / e$ on $N - 1$ degrees of freedom: each draw
holds k distinct numbers, which shrinks the raw sum by $(N-k)/(N-1)$. The Wald-Wolfowitz runs test on the draw sums
split at their median, $z = (R - \\mu) / \\sigma$ with $\\mu = 2 n_1 n_0 / n + 1$ and $\\sigma^2 = 2 n_1 n_0 (2 n_1
n_0 - n) / (n^2 (n - 1))$. A Kolmogorov-Smirnov test of the draw sums against the exact sum distribution of a uniform
k-subset, $D = \\max_s |F_n(s) - F(s)|$, $p = Q_{KS}((\\sqrt{n} + 0.12 + 0.11/\\sqrt{n}) D)$ with $Q_{KS}(\\lambda) =
2 \\sum_{j \\ge 1} (-1)^{j-1} e^{-2 j^2 \\lambda^2}$. The tails are computed here so the engine needs nothing outside
numpy."""

import math

import numpy as np


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def _upper_gamma(a, x):
    if x <= 0:
        return 1.0
    if x < a + 1:
        total = term = 1 / a
        n = 1
        while abs(term) > abs(total) * 1e-16 and n < 100_000:
            term *= x / (a + n)
            total += term
            n += 1
        return 1 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    return _upper_gamma(df / 2, x / 2)


def kolmogorov_sf(lam):
    if lam <= 0:
        return 1.0
    if lam < 1:
        below = math.sqrt(2 * math.pi) / lam * sum(
            math.exp(-(2 * j - 1) ** 2 * math.pi ** 2 / (8 * lam * lam)) for j in range(1, 20))
        return min(1.0, max(0.0, 1 - below))
    return min(1.0, max(0.0, 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 100))))


def chi_square_uniformity_test(history, n_balls):
    draws = np.asarray(history)
    k = draws.shape[1]
    counts = np.bincount(draws.ravel(), minlength=n_balls + 1)[1:]
    expected = counts.sum() / n_balls
    stat = float(((counts - expected) ** 2 / expected).sum()) * (n_balls - 1) / (n_balls - k)
    return stat, chi2_sf(stat, n_balls - 1)


def runs_test_autocorrelation(history):
    sums = np.asarray(history).sum(axis=1)
    above = sums > np.median(sums)
    n1, n0 = int(above.sum()), int((~above).sum())
    n = n1 + n0
    runs = 1 + int((above[1:] != above[:-1]).sum())
    mean = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
    z = (runs - mean) / math.sqrt(var) if var > 0 else 0.0
    return z, 2 * norm_sf(abs(z))


def sum_distribution(n_balls, k):
    max_sum = sum(range(n_balls - k + 1, n_balls + 1))
    ways = np.zeros((k + 1, max_sum + 1))
    ways[0, 0] = 1.0
    for v in range(1, n_balls + 1):
        for j in range(k, 0, -1):
            ways[j, v:] += ways[j - 1, :max_sum + 1 - v]
    return ways[k] / ways[k].sum()


def kolmogorov_smirnov_sum_test(history, n_balls, k):
    pmf = sum_distribution(n_balls, k)
    sums = np.asarray(history).sum(axis=1)
    empirical = np.cumsum(np.bincount(sums, minlength=len(pmf))[:len(pmf)]) / len(sums)
    d = float(np.abs(empirical - np.cumsum(pmf)).max())
    root = math.sqrt(len(sums))
    return d, kolmogorov_sf((root + 0.12 + 0.11 / root) * d)
