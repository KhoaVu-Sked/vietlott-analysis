# Lottery Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A numpy scoring-and-portfolio engine (`engine/`) for Mega 6/45, Power 6/55 and Lotto 5/35 whose five-term ensemble weights are fitted from the draw history, plugged into the backtest harness as a model and into the app as a fourth ticket goal.

**Architecture:** Nine small modules under `engine/`, each one term or one job, composed by an `Engine` class that keeps incremental state and per-draw snapshots so the walk-forward fit is a single pass. A thin adapter in `models/` makes it a harness model; a study script and an app goal expose the fitted weights with error bars.

**Tech Stack:** Python 3.9, numpy 2.0 (engine only), unittest. Everything else in the repo stays standard library.

**Spec:** `docs/superpowers/specs/2026-09-28-lottery-engine-design.md`

## Global Constraints

- Python 3.9.6: no `match`, no `X | Y` type hints, no `zip(strict=)`.
- numpy only inside `engine/`, `models/engine_ensemble.py`, `lottery/engine_study.py`, `tests/test_engine.py` and the engine helper in `prediction.py`; no scipy anywhere; no other new imports.
- `engine/` never imports from `lottery/`.
- No code comments. One docstring per module and per class carrying the formula in LaTeX; no docstring that restates a signature; no function docstrings.
- Everything seeded: `numpy.random.default_rng((seed, t))` for sampling, `random.Random(seed)` for fake histories.
- Run every command from the repo root `/Users/khoa.vu/Desktop/KhoaVu_AI/vietlott-analysis`.
- Full three-game `python3 lottery/backtest_models.py --no-update` must finish under 5 minutes.
- Commits: Khoa runs them. Stage nothing; hand over one `git add … && git commit` command at the end.
- Test command for every task: `python3 -m unittest tests.test_engine -v` (add `-k ClassName` to run one class).

## Review Focus

- A game whose N is not a multiple of 10 (Lotto 5/35): the decade bins must be 4, not 5, and entropy must still land in [0, 1]. Test in Task 2.
- A draw with a repeated number, a number outside 1..N, or the wrong count: `Engine.add` must raise `ValueError`, never corrupt the counts. Test in Task 7.
- A history too short to fit (10 draws or fewer): `Engine.fit` must raise `ValueError` with a message, not divide by zero. Test in Task 7.
- The same ticket twice in a candidate pool: the portfolio must never return duplicates. Test in Task 6.
- numpy missing: the harness must skip `engine_ensemble` with a message and still test the other models. Test in Task 8 (a broken model file in a temporary models directory).

---

### Task 1: Candidate matrices

**Files:**
- Create: `engine/candidates.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_engine.py`

**Interfaces:**
- Produces: `candidates.sample(n_balls, k, m, rng) -> np.ndarray (m, k) int16`, sorted rows, uniform over all C(N, k) tickets, duplicates allowed. `candidates.all_tickets(n_balls, k) -> np.ndarray (C(N,k), k) int16` in lexicographic order, `ValueError` when C(N, k) > `candidates.MAX_ALL` (10,000,000).

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` empty. Create `tests/test_engine.py`:

```python
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lottery"))

from engine import candidates


def fair_history(n, k, t, seed):
    r = random.Random(seed)
    return [sorted(r.sample(range(1, n + 1), k)) for _ in range(t)]


def rigged_history(n, k, t, seed, hot=10, boost=3.0):
    r = random.Random(seed)
    out = []
    for _ in range(t):
        pool = list(range(1, n + 1))
        weights = [boost if x <= hot else 1.0 for x in pool]
        draw = []
        while len(draw) < k:
            i = pool.index(r.choices(pool, weights)[0])
            draw.append(pool.pop(i))
            weights.pop(i)
        out.append(sorted(draw))
    return out


class TestCandidates(unittest.TestCase):
    def test_sample_rows_are_sorted_distinct_and_in_range(self):
        rows = candidates.sample(45, 6, 5000, np.random.default_rng(1))
        self.assertEqual(rows.shape, (5000, 6))
        self.assertEqual(rows.dtype, np.int16)
        self.assertTrue((np.diff(rows, axis=1) > 0).all())
        self.assertGreaterEqual(int(rows.min()), 1)
        self.assertLessEqual(int(rows.max()), 45)

    def test_sample_is_uniform_over_numbers(self):
        rows = candidates.sample(45, 6, 100_000, np.random.default_rng(2))
        counts = np.bincount(rows.ravel(), minlength=46)[1:]
        expected = 100_000 * 6 / 45
        self.assertLess(float(np.abs(counts - expected).max()) / math.sqrt(expected), 5)

    def test_all_tickets_small(self):
        rows = candidates.all_tickets(10, 3)
        self.assertEqual(len(rows), 120)
        self.assertEqual(len(np.unique(rows, axis=0)), 120)
        self.assertTrue((np.diff(rows, axis=1) > 0).all())
        self.assertEqual(rows[0].tolist(), [1, 2, 3])
        self.assertEqual(rows[-1].tolist(), [8, 9, 10])

    def test_all_tickets_refuses_655(self):
        with self.assertRaises(ValueError):
            candidates.all_tickets(55, 6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ERROR importing `engine` (no module named `engine`).

- [ ] **Step 3: Write the implementation**

Create `engine/candidates.py`:

```python
"""Candidate ticket matrices: M sorted rows of k distinct numbers from 1..N as int16, either sampled uniformly from
all $\\binom{N}{k}$ tickets (the k smallest of N random keys is a uniform k-subset) or every ticket in lexicographic
order, built level by level."""

import math

import numpy as np

MAX_ALL = 10_000_000


def sample(n_balls, k, m, rng):
    rows = np.empty((m, k), dtype=np.int16)
    block = 1 << 16
    for start in range(0, m, block):
        keys = rng.random((min(m, start + block) - start, n_balls))
        rows[start:start + len(keys)] = np.sort(np.argpartition(keys, k - 1, axis=1)[:, :k], axis=1) + 1
    return rows


def all_tickets(n_balls, k):
    total = math.comb(n_balls, k)
    if total > MAX_ALL:
        raise ValueError(f"C({n_balls}, {k}) = {total:,} tickets is more than the {MAX_ALL:,} limit; use sample()")
    rows = np.arange(1, n_balls + 1, dtype=np.int16).reshape(-1, 1)
    for _ in range(1, k):
        last = rows[:, -1].astype(np.int64)
        counts = n_balls - last
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        step = np.arange(int(counts.sum())) - offsets
        rows = np.column_stack([np.repeat(rows, counts, axis=0), (np.repeat(last, counts) + 1 + step).astype(np.int16)])
    return rows
```

Also create an empty `engine/__init__.py` for now (Task 7 fills it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 4 tests OK.

---

### Task 2: Static terms: overlap, gaps, entropy

**Files:**
- Create: `engine/overlap.py`, `engine/gaps.py`, `engine/entropy.py`
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Produces: `overlap.log_probabilities(n_balls, k) -> np.ndarray (k+1,)` of log hypergeometric probabilities; `overlap.term(cands, last_draw, table) -> np.ndarray (M,)`, zeros when `last_draw is None`.
- `gaps.SpatialGapAnalyzer(n_balls, k)` with `.gaps(cands) -> (M, k+1) int64`, `.term(cands) -> (M,)`, `.diagnostics(cands) -> dict(variance, min_max_ratio)`.
- `entropy.InformationEntropyScorer(n_balls, k)` with `.decades` (int), `.score(cands) -> (M,) in [0, 1]`, `.term(cands) -> (M,)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py` (add `from engine import entropy, gaps, overlap` next to the candidates import):

```python
class TestStaticTerms(unittest.TestCase):
    def test_overlap_probabilities_sum_to_one(self):
        table = overlap.log_probabilities(45, 6)
        self.assertEqual(len(table), 7)
        self.assertAlmostEqual(float(np.exp(table).sum()), 1.0, places=12)

    def test_overlap_term_counts_shared_numbers(self):
        cands = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 40, 41, 42], [7, 8, 9, 10, 11, 12]], dtype=np.int16)
        table = overlap.log_probabilities(45, 6)
        self.assertEqual(overlap.term(cands, [1, 2, 3, 4, 5, 6], table).tolist(), [table[6], table[3], table[0]])
        self.assertEqual(overlap.term(cands, None, table).tolist(), [0.0, 0.0, 0.0])

    def test_gap_term_is_zero_for_even_spread_and_lower_for_a_run(self):
        g = gaps.SpatialGapAnalyzer(13, 6)
        cands = np.array([[2, 4, 6, 8, 10, 12], [1, 2, 3, 4, 5, 6]], dtype=np.int16)
        term = g.term(cands)
        self.assertAlmostEqual(float(term[0]), 0.0, places=12)
        self.assertLess(float(term[1]), float(term[0]))
        self.assertEqual(g.gaps(cands[:1]).tolist(), [[1, 1, 1, 1, 1, 1, 1]])
        diag = g.diagnostics(cands)
        self.assertEqual(float(diag["variance"][0]), 0.0)
        self.assertEqual(float(diag["min_max_ratio"][1]), 0.0)

    def test_entropy_score_in_unit_range_and_ranks_spread_above_run(self):
        e = entropy.InformationEntropyScorer(45, 6)
        score = e.score(np.array([[3, 11, 22, 30, 37, 44], [1, 2, 3, 4, 5, 6]], dtype=np.int16))
        self.assertTrue(((score >= 0) & (score <= 1)).all())
        self.assertGreater(float(score[0]), float(score[1]))
        self.assertEqual(e.decades, 5)
        self.assertTrue((e.term(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int16)) < 0).all())

    def test_entropy_handles_lotto_535_bins(self):
        e = entropy.InformationEntropyScorer(35, 5)
        self.assertEqual(e.decades, 4)
        score = e.score(np.array([[1, 12, 23, 34, 35], [31, 32, 33, 34, 35]], dtype=np.int16))
        self.assertTrue(((score >= 0) & (score <= 1)).all())
        self.assertGreater(float(score[0]), float(score[1]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError for `entropy, gaps, overlap`.

- [ ] **Step 3: Write the implementation**

Create `engine/overlap.py`:

```python
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
```

Create `engine/gaps.py`:

```python
"""Spatial gap term. A sorted ticket $x_1 < \\dots < x_k$ has $k+1$ gaps $g_0 = x_1 - 1$, $g_i = x_{i+1} - x_i - 1$,
$g_k = N - x_k$ summing to $G = N - k$. With $q = g / G$ the term is $-\\mathrm{KL}(q \\| U_{k+1}) =
-\\sum_i q_i \\log((k+1) q_i)$: 0 for an even spread, negative when the numbers cluster. Gap variance and the
min/max ratio are diagnostics only."""

import numpy as np


class SpatialGapAnalyzer:
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
```

Create `engine/entropy.py`:

```python
"""Entropy term. Four Shannon entropies $H = -\\sum_b p_b \\log p_b$ of the ticket's numbers: over bins of ten
($\\lceil N/10 \\rceil$ bins), residues mod 3, residues mod 5 and last digits, each divided by its maximum
$\\log \\min(k, \\text{bins})$ and averaged into a score in $[0, 1]$. The term is $\\log \\max(\\text{score}, 10^{-9})$."""

import math

import numpy as np


class InformationEntropyScorer:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 9 tests OK.

---

### Task 3: Dynamic terms: Bayes and Markov

**Files:**
- Create: `engine/bayes.py`, `engine/markov.py`
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Produces: `bayes.BayesianDirichletModel(n_balls, k, alpha0=1.0, decay=0.005)` with `.counts (N,)`, `.update(draw)`, `.probabilities() -> (N,)`, `.log_terms() -> (N,)` = `log(N p_i)`.
- `markov.MarkovTransitionModel(n_balls, k, eps=1e-6)` with `.counts (N, N)`, `.last` (0-based index array or None), `.update(draw)`, `.matrix() -> (N, N)`, `.log_terms() -> (N,)`, zeros before any draw.
- Both take `draw` as a sequence of 1-based ints.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py` (add `bayes, markov` to the engine import):

```python
class TestDynamicTerms(unittest.TestCase):
    def test_bayes_without_decay_or_prior_equals_raw_counts(self):
        b = bayes.BayesianDirichletModel(10, 3, alpha0=0.0, decay=0.0)
        for d in ([1, 2, 3], [1, 2, 4], [1, 5, 6]):
            b.update(d)
        self.assertEqual(b.counts.tolist(), [3, 2, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(float(b.probabilities().sum()), 1.0, places=12)
        self.assertAlmostEqual(float(b.probabilities()[0]), 3 / 9, places=12)

    def test_bayes_decay_weights_the_newest_draw_most(self):
        b = bayes.BayesianDirichletModel(10, 3, alpha0=1.0, decay=0.5)
        b.update([1, 2, 3])
        b.update([4, 5, 6])
        self.assertAlmostEqual(float(b.counts[0]), math.exp(-0.5), places=12)
        self.assertAlmostEqual(float(b.counts[3]), 1.0, places=12)

    def test_bayes_is_uniform_before_any_draw(self):
        self.assertTrue(np.allclose(bayes.BayesianDirichletModel(45, 6).log_terms(), 0.0))
        self.assertTrue(np.allclose(bayes.BayesianDirichletModel(45, 6, alpha0=0.0).log_terms(), 0.0))

    def test_markov_rows_sum_to_one_and_start_uniform(self):
        m = markov.MarkovTransitionModel(10, 3)
        self.assertTrue(np.allclose(m.log_terms(), 0.0))
        m.update([1, 2, 3])
        self.assertTrue(np.allclose(m.log_terms(), 0.0))
        m.update([4, 5, 6])
        m.update([4, 7, 8])
        self.assertTrue(np.allclose(m.matrix().sum(axis=1), 1.0))
        self.assertEqual(float(m.counts[3, 3]), 1.0)
        self.assertGreater(float(m.log_terms()[6]), 0.0)
        self.assertLess(float(m.log_terms()[0]), 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError for `bayes, markov`.

- [ ] **Step 3: Write the implementation**

Create `engine/bayes.py`:

```python
"""Bayesian Dirichlet model with exponential forgetting. Pseudo-counts
$a_i = \\alpha_0 + \\sum_t e^{-\\lambda (T - t)} \\, \\mathbb{1}[i \\in D_t]$ give the posterior mean
$p_i = a_i / \\sum_j a_j$; a ticket's term is $\\sum_{i \\in T} \\log(N p_i)$, 0 under a fair posterior."""

import math

import numpy as np


class BayesianDirichletModel:
    def __init__(self, n_balls, k, alpha0=1.0, decay=0.005):
        self.n, self.k, self.alpha0 = n_balls, k, alpha0
        self.fade = math.exp(-decay)
        self.counts = np.zeros(n_balls)

    def update(self, draw):
        self.counts *= self.fade
        self.counts[np.asarray(draw) - 1] += 1.0

    def probabilities(self):
        a = self.alpha0 + self.counts
        return a / a.sum() if a.sum() > 0 else np.full(self.n, 1 / self.n)

    def log_terms(self):
        return np.log(self.n * self.probabilities())
```

Create `engine/markov.py`:

```python
"""Markov transition model. $M_{ij} = (n_{ij} + \\epsilon) / \\sum_j (n_{ij} + \\epsilon)$ with $n_{ij}$ the times
number $j$ was drawn right after a draw holding $i$. Given the last draw $D$, $q_j = \\frac{1}{|D|}\\sum_{i \\in D} M_{ij}$
and a ticket's term is $\\sum_{j \\in T} \\log(N q_j)$, 0 before any transition is known."""

import numpy as np


class MarkovTransitionModel:
    def __init__(self, n_balls, k, eps=1e-6):
        self.n, self.k, self.eps = n_balls, k, eps
        self.counts = np.zeros((n_balls, n_balls))
        self.last = None

    def update(self, draw):
        d = np.asarray(draw) - 1
        if self.last is not None:
            self.counts[np.ix_(self.last, d)] += 1.0
        self.last = d

    def matrix(self):
        m = self.counts + self.eps
        return m / m.sum(axis=1, keepdims=True)

    def log_terms(self):
        if self.last is None:
            return np.zeros(self.n)
        return np.log(self.n * self.matrix()[self.last].mean(axis=0))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 13 tests OK.

---

### Task 4: Diagnostics

**Files:**
- Create: `engine/diagnostics.py`
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Produces: `diagnostics.norm_sf(z)`, `diagnostics.chi2_sf(x, df)`, `diagnostics.kolmogorov_sf(lam)`, `diagnostics.sum_distribution(n_balls, k) -> np.ndarray` (pmf indexed by sum, length max_sum + 1), `diagnostics.chi_square_uniformity_test(history, n_balls) -> (stat, p)`, `diagnostics.runs_test_autocorrelation(history) -> (z, p)`, `diagnostics.kolmogorov_smirnov_sum_test(history, n_balls, k) -> (d, p)`. `history` is a list of sorted draws.

- [ ] **Step 1: Write the failing tests**

Append (add `diagnostics` to the engine import):

```python
class TestDiagnostics(unittest.TestCase):
    def test_tails_match_known_values(self):
        import power645_study as independent
        self.assertAlmostEqual(diagnostics.chi2_sf(3.841, 1), 0.05, places=3)
        self.assertAlmostEqual(diagnostics.chi2_sf(0.0, 5), 1.0, places=12)
        for x, df in ((60.0, 44), (10.0, 5), (200.0, 54), (0.5, 3)):
            self.assertAlmostEqual(diagnostics.chi2_sf(x, df), independent.chi2_sf(x, df), places=6)
        self.assertAlmostEqual(diagnostics.norm_sf(1.959964), 0.025, places=5)
        self.assertAlmostEqual(diagnostics.kolmogorov_sf(1.36), 0.05, places=2)
        self.assertAlmostEqual(diagnostics.kolmogorov_sf(0.999), diagnostics.kolmogorov_sf(1.001), places=3)

    def test_sum_distribution_is_exact(self):
        pmf = diagnostics.sum_distribution(10, 3)
        self.assertEqual(len(pmf), 28)
        self.assertAlmostEqual(float(pmf.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(pmf[6]), 1 / 120, places=12)
        self.assertAlmostEqual(float(pmf[27]), 1 / 120, places=12)

    def test_fair_history_passes_all_three(self):
        h = fair_history(45, 6, 1500, 1)
        for stat, p in (diagnostics.chi_square_uniformity_test(h, 45), diagnostics.runs_test_autocorrelation(h),
                        diagnostics.kolmogorov_smirnov_sum_test(h, 45, 6)):
            self.assertTrue(0.001 <= p <= 0.999, (stat, p))

    def test_rigged_history_fails_chi_square(self):
        _, p = diagnostics.chi_square_uniformity_test(rigged_history(45, 6, 1500, 1), 45)
        self.assertLess(p, 1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError for `diagnostics`.

- [ ] **Step 3: Write the implementation**

Create `engine/diagnostics.py`:

```python
"""Empirical validity checks on a draw history. Chi-square goodness of fit of the number counts against the discrete
uniform $U(1, N)$, $\\chi^2 = \\sum_i (c_i - e)^2 / e$ on $N - 1$ degrees of freedom. The Wald-Wolfowitz runs test on
the draw sums split at their median, $z = (R - \\mu) / \\sigma$ with $\\mu = 2 n_1 n_0 / n + 1$ and
$\\sigma^2 = 2 n_1 n_0 (2 n_1 n_0 - n) / (n^2 (n - 1))$. A Kolmogorov-Smirnov test of the draw sums against the exact sum
distribution of a uniform k-subset, $D = \\max_s |F_n(s) - F(s)|$, $p = Q_{KS}((\\sqrt{n} + 0.12 + 0.11/\\sqrt{n}) D)$
with $Q_{KS}(\\lambda) = 2 \\sum_{j \\ge 1} (-1)^{j-1} e^{-2 j^2 \\lambda^2}$. The tails are computed here so the engine
needs nothing outside numpy."""

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
    counts = np.bincount(np.asarray(history).ravel(), minlength=n_balls + 1)[1:]
    expected = counts.sum() / n_balls
    stat = float(((counts - expected) ** 2 / expected).sum())
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 17 tests OK. If `test_fair_history_passes_all_three` fails on a p-value just outside the band, print the three (stat, p) pairs: a value like p = 0.0004 with a correct statistic means the seed produced a rare history and the test seed should change to 2; a statistic in the thousands means a bug in the counts.

---

### Task 5: Scorer and weight fit

**Files:**
- Create: `engine/scorer.py`
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Produces: `scorer.TERMS = ("bayes", "markov", "gap", "entropy", "overlap")`; `scorer.Fit` with `.w (5,)`, `.se (5,)`, `.z` property, `.draws_used`, `.iterations`, `.converged`, `.as_dict() -> {term: {"w", "se", "z"}}`; `scorer.JointEnsembleScorer(weights=None)` with `.w` and `.score(features (M, 5)) -> (M,)`; `scorer.fit_weights(chosen (T, 5), sampled (T, S, 5), prior_sd=1.0, max_iter=50, tol=1e-8) -> Fit`.

- [ ] **Step 1: Write the failing tests**

Append (add `scorer` to the engine import):

```python
class TestScorer(unittest.TestCase):
    def test_score_is_the_weighted_sum(self):
        s = scorer.JointEnsembleScorer([1.0, 2.0, 0.0, 0.0, -1.0])
        self.assertEqual(s.score(np.array([[1.0, 1.0, 5.0, 5.0, 1.0]])).tolist(), [2.0])
        self.assertEqual(scorer.JointEnsembleScorer().w.tolist(), [0.0] * 5)

    def test_fit_finds_zero_on_noise_and_the_signal_when_there_is_one(self):
        rng = np.random.default_rng(3)
        draws, alternatives = 400, 200
        sampled = rng.standard_normal((draws, alternatives, 5))
        noise = scorer.fit_weights(rng.standard_normal((draws, 5)), sampled)
        self.assertTrue(noise.converged)
        self.assertTrue((np.abs(noise.z) < 3).all(), noise.as_dict())
        self.assertEqual(noise.draws_used, draws)
        chosen = np.empty((draws, 5))
        for t in range(draws):
            u = sampled[t, :, 0]
            p = np.exp(u - u.max())
            chosen[t] = sampled[t][rng.choice(alternatives, p=p / p.sum())]
        signal = scorer.fit_weights(chosen, sampled)
        self.assertGreater(float(signal.z[0]), 5, signal.as_dict())
        self.assertTrue((np.abs(signal.z[1:]) < 3).all(), signal.as_dict())
        self.assertAlmostEqual(float(signal.w[0]), 1.0, delta=0.3)
        self.assertEqual(sorted(signal.as_dict()), sorted(scorer.TERMS))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError for `scorer`.

- [ ] **Step 3: Write the implementation**

Create `engine/scorer.py`:

```python
"""Joint ensemble scorer. A ticket's composite is $w \\cdot f(T)$ with $f = (\\text{bayes}, \\text{markov},
\\text{gap}, \\text{entropy}, \\text{overlap})$. The weights are fitted by conditional logit: each past draw $D_t$ is the
alternative chosen among itself and $S$ sampled tickets, $\\log P(D_t \\mid w) = w \\cdot f_{t0} - \\log \\sum_s e^{w \\cdot
f_{ts}}$, maximised with a $\\mathcal{N}(0, \\sigma^2)$ prior on each weight by Newton's method with step halving.
Standard errors are $\\sqrt{\\mathrm{diag}((-H)^{-1})}$ at the optimum."""

import numpy as np

TERMS = ("bayes", "markov", "gap", "entropy", "overlap")


class Fit:
    def __init__(self, w, se, draws_used, iterations, converged):
        self.w, self.se = np.asarray(w, dtype=float), np.asarray(se, dtype=float)
        self.draws_used, self.iterations, self.converged = draws_used, iterations, converged

    @property
    def z(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.se) & (self.se > 0), self.w / self.se, 0.0)

    def as_dict(self):
        return {name: {"w": float(w), "se": float(se), "z": float(z)}
                for name, w, se, z in zip(TERMS, self.w, self.se, self.z)}


class JointEnsembleScorer:
    def __init__(self, weights=None):
        self.w = np.zeros(len(TERMS)) if weights is None else np.asarray(weights, dtype=float)

    def score(self, features):
        return features @ self.w


def fit_weights(chosen, sampled, prior_sd=1.0, max_iter=50, tol=1e-8):
    features = np.concatenate([np.asarray(chosen)[:, None, :], np.asarray(sampled)], axis=1)
    draws, dim = features.shape[0], features.shape[2]
    ridge = 1 / prior_sd ** 2
    w = np.zeros(dim)

    def objective(w):
        u = features @ w
        top = u.max(axis=1)
        return float((u[:, 0] - top - np.log(np.exp(u - top[:, None]).sum(axis=1))).sum() - 0.5 * ridge * (w @ w))

    def gradient_hessian(w):
        u = features @ w
        p = np.exp(u - u.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean = np.einsum("ts,tsf->tf", p, features)
        g = (features[:, 0, :] - mean).sum(axis=0) - ridge * w
        h = -(np.einsum("ts,tsf,tsg->fg", p, features, features) - mean.T @ mean) - ridge * np.eye(dim)
        return g, h

    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        g, h = gradient_hessian(w)
        try:
            step = np.linalg.solve(h, -g)
        except np.linalg.LinAlgError:
            return Fit(np.zeros(dim), np.full(dim, np.inf), draws, iterations, False)
        base, scale = objective(w), 1.0
        while scale > 1e-6 and objective(w + scale * step) < base:
            scale /= 2
        w = w + scale * step
        if float(np.abs(scale * step).max()) < tol:
            converged = True
            break
    _, h = gradient_hessian(w)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(-h)))
    except np.linalg.LinAlgError:
        se = np.full(dim, np.inf)
    return Fit(w, se, draws, iterations, converged)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 19 tests OK.

---

### Task 6: Portfolio picker

**Files:**
- Create: `engine/portfolio.py`
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Produces: `portfolio.LotteryWheelingOptimizer(k, max_overlap=None)` (default `k - 2`) with `.select(cands, scores, m) -> list of sorted int lists`, in score order, at most `m`.

- [ ] **Step 1: Write the failing tests**

Append (add `portfolio` to the engine import):

```python
class TestPortfolio(unittest.TestCase):
    def test_respects_overlap_cap_and_skips_duplicates(self):
        cands = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 8, 9],
                          [10, 11, 12, 13, 14, 15], [1, 2, 3, 20, 21, 22]], dtype=np.int16)
        scores = np.array([9.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        chosen = portfolio.LotteryWheelingOptimizer(6).select(cands, scores, 3)
        self.assertEqual(chosen, [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 8, 9], [10, 11, 12, 13, 14, 15]])
        strict = portfolio.LotteryWheelingOptimizer(6, max_overlap=0).select(cands, scores, 5)
        self.assertEqual(strict, [[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]])

    def test_takes_the_highest_scores_first(self):
        cands = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]], dtype=np.int16)
        chosen = portfolio.LotteryWheelingOptimizer(6).select(cands, np.array([1.0, 3.0, 2.0]), 2)
        self.assertEqual(chosen, [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError for `portfolio`.

- [ ] **Step 3: Write the implementation**

Create `engine/portfolio.py`:

```python
"""Greedy wheeling. Candidates are visited in score order; the best is taken, then each next candidate whose overlap
$|T_a \\cap T_b|$ with every chosen ticket is at most the cap (default $k - 2$), until $m$ tickets are chosen or the
candidates run out. A duplicate of a chosen ticket overlaps it in $k$ numbers and is skipped by the same rule."""

import numpy as np


class LotteryWheelingOptimizer:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 21 tests OK.

---

### Task 7: The Engine class

**Files:**
- Modify: `engine/__init__.py` (replace the empty file)
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `engine.Engine(n_balls, k, alpha0=1.0, decay=0.005, eps=1e-6, seed=2026)` with `.n`, `.k`, `.seed`, `.draws` (list of sorted lists), `.t` (property, draws seen), `.bayes`, `.markov`, `.gaps`, `.entropy`, `.overlap_table`, `.snap_bayes` / `.snap_markov` (lists of (N,) arrays: the log terms as they stood before each draw), `.scorer` (JointEnsembleScorer), `.fit_result` (Fit or None); methods `.add(draw)`, `.extend(draws)` (adds only `draws[self.t:]`), `.static_terms(cands) -> (M, 2)` (gap, entropy), `.features(cands, static=None, at=None) -> (M, 5)` in `scorer.TERMS` order (`at=t` uses the snapshots before draw index t), `.fit(samples=1000, start=10, prior_sd=1.0) -> Fit` (raises `ValueError` when `self.t <= start`), `.score(cands, block=1_000_000) -> (M,)`, `.best(cands) -> sorted int list`, `.portfolio(m, cands, max_overlap=None) -> list of sorted int lists`.
- Re-exports: `from engine import Engine, candidates, diagnostics` must work.

- [ ] **Step 1: Write the failing tests**

Append (change the import line to `from engine import Engine, bayes, candidates, diagnostics, entropy, gaps, markov, overlap, portfolio, scorer`):

```python
class TestEngine(unittest.TestCase):
    def test_rejects_bad_draws(self):
        eng = Engine(45, 6)
        for bad in ([1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 46]):
            with self.assertRaises(ValueError):
                eng.add(bad)
        self.assertEqual(eng.t, 0)
        self.assertEqual(float(eng.bayes.counts.sum()), 0.0)

    def test_features_match_plain_python(self):
        h = fair_history(45, 6, 200, 4)
        eng = Engine(45, 6)
        eng.extend(h)
        cands = candidates.sample(45, 6, 50, np.random.default_rng(5))
        feats = eng.features(cands)
        self.assertEqual(feats.shape, (50, 5))
        pb, pm = eng.bayes.log_terms(), eng.markov.log_terms()
        last, table = set(h[-1]), overlap.log_probabilities(45, 6)
        for row, f in zip(cands.tolist(), feats):
            self.assertAlmostEqual(f[0], sum(pb[x - 1] for x in row), places=10)
            self.assertAlmostEqual(f[1], sum(pm[x - 1] for x in row), places=10)
            g = [row[0] - 1] + [b - a - 1 for a, b in zip(row, row[1:])] + [45 - row[-1]]
            self.assertAlmostEqual(f[2], -sum(v / 39 * math.log(v / 39 * 7) for v in g if v > 0), places=10)
            self.assertAlmostEqual(f[4], table[len(last & set(row))], places=10)
        self.assertTrue(np.allclose(feats[:, 2:4], eng.static_terms(cands)))

    def test_extend_is_incremental_and_snapshots_are_pre_draw(self):
        h = fair_history(45, 6, 60, 6)
        eng = Engine(45, 6)
        eng.extend(h[:30])
        eng.extend(h)
        self.assertEqual(eng.t, 60)
        self.assertEqual(len(eng.snap_bayes), 60)
        self.assertTrue(np.allclose(eng.snap_bayes[0], 0.0))
        fresh = Engine(45, 6)
        fresh.extend(h[:25])
        self.assertTrue(np.allclose(eng.snap_bayes[25], fresh.bayes.log_terms()))
        self.assertTrue(np.allclose(eng.snap_markov[25], fresh.markov.log_terms()))
        self.assertTrue(np.allclose(eng.features(np.array([h[30]], dtype=np.int16), at=25)[0, :2],
                                    fresh.features(np.array([h[30]], dtype=np.int16))[0, :2]))

    def test_fit_on_fair_is_within_noise_and_rigged_shows_bayes(self):
        eng = Engine(45, 6)
        eng.extend(fair_history(45, 6, 1500, 7))
        fit = eng.fit()
        self.assertTrue(fit.converged)
        self.assertEqual(fit.draws_used, 1490)
        self.assertTrue((np.abs(fit.z) < 3).all(), fit.as_dict())
        self.assertIs(eng.fit_result, fit)
        self.assertTrue(np.array_equal(eng.scorer.w, fit.w))
        rig = Engine(45, 6)
        rig.extend(rigged_history(45, 6, 1500, 7))
        self.assertGreater(float(rig.fit().z[0]), 5, rig.fit_result.as_dict())

    def test_fit_needs_history(self):
        eng = Engine(45, 6)
        eng.extend(fair_history(45, 6, 10, 8))
        with self.assertRaises(ValueError):
            eng.fit()

    def test_best_and_portfolio_are_deterministic(self):
        h = fair_history(35, 5, 300, 9)
        a, b = Engine(35, 5), Engine(35, 5)
        a.extend(h)
        b.extend(h)
        a.fit()
        b.fit()
        cands = candidates.sample(35, 5, 20_000, np.random.default_rng(10))
        self.assertEqual(a.best(cands), b.best(cands))
        self.assertEqual(len(a.best(cands)), 5)
        port = a.portfolio(10, cands)
        self.assertEqual(len(port), 10)
        self.assertTrue(all(len(set(x) & set(y)) <= 3 for x in port for y in port if x != y))
        self.assertTrue(np.allclose(a.score(cands), a.score(cands, block=7)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -v`
Expected: ImportError: cannot import name `Engine`.

- [ ] **Step 3: Write the implementation**

Replace `engine/__init__.py` with:

```python
"""Engine(n_balls, k): the five-term ensemble over a draw history. State is incremental: each draw updates the Bayes
and Markov terms and stores the per-number log terms as they stood before it, so a walk-forward fit is one pass. A
ticket's features are $(\\text{bayes}, \\text{markov}, \\text{gap}, \\text{entropy}, \\text{overlap})$ and its score is
$w \\cdot f$ with $w$ from the conditional-logit fit in scorer.py."""

import numpy as np

from . import candidates, diagnostics, overlap
from .bayes import BayesianDirichletModel
from .entropy import InformationEntropyScorer
from .gaps import SpatialGapAnalyzer
from .markov import MarkovTransitionModel
from .portfolio import LotteryWheelingOptimizer
from .scorer import TERMS, Fit, JointEnsembleScorer, fit_weights

__all__ = ["Engine", "Fit", "TERMS", "candidates", "diagnostics"]


class Engine:
    def __init__(self, n_balls, k, alpha0=1.0, decay=0.005, eps=1e-6, seed=2026):
        self.n, self.k, self.seed = n_balls, k, seed
        self.bayes = BayesianDirichletModel(n_balls, k, alpha0, decay)
        self.markov = MarkovTransitionModel(n_balls, k, eps)
        self.gaps = SpatialGapAnalyzer(n_balls, k)
        self.entropy = InformationEntropyScorer(n_balls, k)
        self.overlap_table = overlap.log_probabilities(n_balls, k)
        self.draws, self.snap_bayes, self.snap_markov = [], [], []
        self.scorer = JointEnsembleScorer()
        self.fit_result = None

    @property
    def t(self):
        return len(self.draws)

    def add(self, draw):
        d = sorted(int(x) for x in draw)
        if len(d) != self.k or len(set(d)) != self.k or d[0] < 1 or d[-1] > self.n:
            raise ValueError(f"draw {list(draw)!r} is not {self.k} distinct numbers in 1..{self.n}")
        self.snap_bayes.append(self.bayes.log_terms())
        self.snap_markov.append(self.markov.log_terms())
        self.bayes.update(d)
        self.markov.update(d)
        self.draws.append(d)

    def extend(self, draws):
        for d in draws[self.t:]:
            self.add(d)

    def static_terms(self, cands):
        return np.stack([self.gaps.term(cands), self.entropy.term(cands)], axis=1)

    def features(self, cands, static=None, at=None):
        if at is None:
            b, m, last = self.bayes.log_terms(), self.markov.log_terms(), (self.draws[-1] if self.draws else None)
        else:
            b, m, last = self.snap_bayes[at], self.snap_markov[at], (self.draws[at - 1] if at > 0 else None)
        idx = cands.astype(np.int64) - 1
        sta = self.static_terms(cands) if static is None else static
        return np.column_stack([b[idx].sum(axis=1), m[idx].sum(axis=1), sta,
                                overlap.term(cands, last, self.overlap_table)])

    def fit(self, samples=1000, start=10, prior_sd=1.0):
        if self.t <= start:
            raise ValueError(f"fitting needs more than {start} draws, have {self.t}")
        chosen, sampled = [], []
        for t in range(start, self.t):
            pool = candidates.sample(self.n, self.k, samples, np.random.default_rng((self.seed, t)))
            chosen.append(self.features(np.array([self.draws[t]], dtype=np.int16), at=t)[0])
            sampled.append(self.features(pool, at=t))
        self.fit_result = fit_weights(np.array(chosen), np.array(sampled), prior_sd)
        self.scorer = JointEnsembleScorer(self.fit_result.w)
        return self.fit_result

    def score(self, cands, block=1_000_000):
        out = np.empty(len(cands))
        for s in range(0, len(cands), block):
            out[s:s + block] = self.scorer.score(self.features(cands[s:s + block]))
        return out

    def best(self, cands):
        return sorted(int(x) for x in cands[int(np.argmax(self.score(cands)))])

    def portfolio(self, m, cands, max_overlap=None):
        return LotteryWheelingOptimizer(self.k, max_overlap).select(cands, self.score(cands), m)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 27 tests OK, under 90 seconds. If `test_fit_on_fair_is_within_noise_and_rigged_shows_bayes` fails only on the `|z| < 3` line with one z between 3 and 4, that is the 1-in-300 chance per term of the seed; change the fair history seed to 12 and rerun. Any z above 10 on a fair history is a bug in the features.

---

### Task 8: Harness adapter and the numpy-missing path

**Files:**
- Create: `models/engine_ensemble.py`
- Modify: `lottery/backtest_models.py` (`load_models`, lines 64-72)
- Modify: `tests/test_engine.py` (append)

**Interfaces:**
- Consumes: `engine.Engine`, `engine.candidates.sample`.
- Produces: `predict(past, n_balls) -> list of n_balls floats` (the harness protocol); module constants `START = 50`, `REFIT = 100`, `POOL = 200_000`, `SEED = 2026`; module dict `state`.
- `backtest_models.load_models(names)` now imports each model once and skips any that raises `ImportError`, printing `"{name} skipped: {message}"`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
class TestAdapter(unittest.TestCase):
    def load_adapter(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine_ensemble", ROOT / "models" / "engine_ensemble.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_predict_shape_and_determinism(self):
        mod = self.load_adapter()
        h = fair_history(45, 6, 160, 11)
        self.assertEqual(mod.predict(h[:20], 45), [1.0] * 45)
        first = mod.predict(h[:150], 45)
        self.assertEqual(len(first), 45)
        self.assertEqual(sum(v > 1.0 for v in first), 6)
        again = mod.predict(h[:151], 45)
        self.assertEqual(sum(v > 1.0 for v in again), 6)
        mod.state.clear()
        self.assertEqual(mod.predict(h[:150], 45), first)

    def test_harness_skips_a_model_that_cannot_import(self):
        import tempfile
        import backtest_models as bm
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.py").write_text("raise ImportError('needs a missing package')\n")
            (Path(tmp) / "fine.py").write_text("def predict(past, n_balls):\n    return [1.0] * n_balls\n")
            keep = bm.MODELS_DIR
            bm.MODELS_DIR = Path(tmp)
            try:
                loaded = bm.load_models(None)
            finally:
                bm.MODELS_DIR = keep
        self.assertEqual(sorted(loaded), ["fine"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest tests.test_engine -k TestAdapter -v`
Expected: FileNotFoundError for `models/engine_ensemble.py`; the skip test fails because `load_models` raises ImportError.

- [ ] **Step 3: Write the adapter**

Create `models/engine_ensemble.py`:

```python
"""The numpy engine as a harness model: the five-term ensemble (Bayes, Markov, gap, entropy, overlap) with weights
fitted by conditional logit on the draws so far, refitted every 100 draws, scoring 200,000 sampled candidates and
returning the best one. See engine/ and docs/superpowers/specs/2026-09-28-lottery-engine-design.md."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from engine import Engine, candidates
except ImportError as exc:
    raise ImportError("engine_ensemble needs numpy: pip3 install numpy") from exc

START, REFIT, POOL, SEED = 50, 100, 200_000, 2026
state = {}


def engine_for(past, n_balls):
    eng = state.get("engine")
    if (eng is None or eng.n != n_balls or eng.t > len(past)
            or (eng.t and (past[0] != eng.draws[0] or past[eng.t - 1] != eng.draws[eng.t - 1]))):
        state.clear()
        eng = state["engine"] = Engine(n_balls, len(past[0]), seed=SEED)
    eng.extend(past)
    return eng


def predict(past, n_balls):
    if len(past) < START:
        return [1.0] * n_balls
    eng = engine_for(past, n_balls)
    if state.get("fitted_at") is None or (eng.t - START) % REFIT == 0 and state["fitted_at"] != eng.t:
        eng.fit()
        state["cands"] = candidates.sample(n_balls, eng.k, POOL, np.random.default_rng((SEED, eng.t)))
        state["static"] = eng.static_terms(state["cands"])
        state["fitted_at"] = eng.t
    scores = eng.scorer.score(eng.features(state["cands"], static=state["static"]))
    picks = set(int(x) for x in state["cands"][int(np.argmax(scores))])
    return [1.0 + (1e-9 if i in picks else 0.0) for i in range(1, n_balls + 1)]
```

- [ ] **Step 4: Make the harness skip models that cannot import**

In `lottery/backtest_models.py`, replace `load_models`:

```python
def load_models(names):
    out = {}
    for path in sorted(MODELS_DIR.glob("*.py")):
        if path.name.startswith("_") or (names and path.stem not in names):
            continue
        try:
            load_module(path)
        except ImportError as exc:
            print(f"{path.stem} skipped: {exc}")
            continue
        out[path.stem] = (path, hashlib.sha256(path.read_bytes()).hexdigest()[:12])
    if not out:
        raise SystemExit(f"no models found in {MODELS_DIR}")
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest tests.test_engine -v`
Expected: 29 tests OK.

- [ ] **Step 6: Run the real harness and time it**

Run: `time python3 lottery/backtest_models.py --no-update`
Expected: finishes under 5 minutes; `results/result.json` has 21 models for each of 645, 655, 535 including `engine_ensemble`; its verdict is within luck on all three (any z above 3 on a game is worth a look at the adapter, not a celebration). If wall-clock is over 5 minutes, set `POOL = 100_000` in the adapter and re-measure; record the time in the final report. Then `python3 -c "import json; r = json.load(open('results/result.json')); print({g: (len(v['models']), round(v['models']['engine_ensemble']['z_vs_fair'], 2)) for g, v in r['games'].items()})"`.

---

### Task 9: Study script and update_all step

**Files:**
- Create: `lottery/engine_study.py`
- Modify: `update_all.py` (FIRST_WAVE list, lines 24-37)

**Interfaces:**
- Consumes: `engine.Engine`, `engine.candidates`, `engine.diagnostics`, `backtest_models.GAMES`, `backtest_models.load_game`, `prediction.simulated_any(tickets, n_balls, seed, size)`, `prediction.SIMS`, `coverage.design`.
- Produces: `results/engine_study.json` with `{"generated_at", "games": {code: {"draws", "diagnostics": {...}, "fit": {...}, "fakes": [{...}], "best_ticket", "portfolio", "p_any_portfolio", "p_any_design"}}}`, and a printed report. CLI: `--fakes 2 --seed 2026 --no-update`.

- [ ] **Step 1: Write the script**

Create `lottery/engine_study.py`:

```python
"""Run the numpy engine on each game: the three randomness diagnostics, the conditional-logit fit of the five ensemble
weights with error bars on the real history and on fake fair lotteries, the best-scored ticket and a 10-ticket greedy
portfolio compared with the covering design. Writes results/engine_study.json.

Usage: python3 lottery/engine_study.py [--fakes 2] [--seed 2026] [--no-update]
"""

import argparse
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import backtest_models as bm
import coverage
import prediction as app
from engine import Engine, TERMS, candidates, diagnostics

OUT = ROOT / "results" / "engine_study.json"
POOL = 1_000_000
PORTFOLIO = 10


def fitted(draws, n, k, seed):
    eng = Engine(n, k, seed=seed)
    eng.extend(draws)
    eng.fit()
    return eng


def weights_line(fit):
    parts = [f"{name} {v['w']:+.3f}±{v['se']:.3f} (z {v['z']:+.1f})" for name, v in fit.as_dict().items()]
    flagged = [name for name, v in fit.as_dict().items() if abs(v["z"]) > 3]
    verdict = "all within error bars: nothing beyond a fair lottery" if not flagged else "beyond 3 errors: " + ", ".join(flagged)
    return " · ".join(parts), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        for code in bm.GAMES:
            update.refresh(code)
    report = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "games": {}}
    for code, game in bm.GAMES.items():
        n, k = game["balls"], game["k"]
        draws, _ = bm.load_game(code)
        chi = diagnostics.chi_square_uniformity_test(draws, n)
        runs = diagnostics.runs_test_autocorrelation(draws)
        ks = diagnostics.kolmogorov_smirnov_sum_test(draws, n, k)
        eng = fitted(draws, n, k, args.seed)
        if math.comb(n, k) <= candidates.MAX_ALL:
            pool, pool_note = candidates.all_tickets(n, k), f"all {math.comb(n, k):,} tickets"
        else:
            pool, pool_note = candidates.sample(n, k, POOL, np.random.default_rng((args.seed, 1))), f"{POOL:,} sampled tickets"
        best = eng.best(pool)
        port = eng.portfolio(PORTFOLIO, pool)
        design = coverage.design(PORTFOLIO, n, args.seed, size=k)
        p_port = app.simulated_any(port, n, args.seed, size=k)
        p_design = app.simulated_any(design, n, args.seed, size=k)
        fakes = []
        for f in range(args.fakes):
            r = random.Random(args.seed + 500 + f)
            fake = [sorted(r.sample(range(1, n + 1), k)) for _ in draws]
            fakes.append(fitted(fake, n, k, args.seed).fit_result.as_dict())
        line, verdict = weights_line(eng.fit_result)
        print(f"{game['name']}: {len(draws):,} draws")
        print(f"  chi-square uniformity {chi[0]:.1f} (p {chi[1]:.3f}) · runs test z {runs[0]:+.2f} (p {runs[1]:.3f})"
              f" · KS on sums D {ks[0]:.4f} (p {ks[1]:.3f})")
        print(f"  fitted weights on {eng.fit_result.draws_used:,} draws: {line}")
        print(f"  verdict: {verdict}")
        for i, fake in enumerate(fakes, 1):
            print(f"  fake lottery {i}: " + " · ".join(f"{name} z {v['z']:+.1f}" for name, v in fake.items()))
        print(f"  best ticket over {pool_note}: {' '.join(f'{x:02d}' for x in best)}")
        print(f"  {PORTFOLIO}-ticket portfolio (overlap ≤ {k - 2}): any prize {p_port * 100:.2f}% vs covering design"
              f" {p_design * 100:.2f}% ({app.SIMS:,} simulated draws)")
        for t in port:
            print("      " + " ".join(f"{x:02d}" for x in t))
        print()
        report["games"][code] = {
            "draws": len(draws),
            "diagnostics": {"chi_square": {"stat": chi[0], "p": chi[1]}, "runs": {"z": runs[0], "p": runs[1]},
                            "ks_sums": {"d": ks[0], "p": ks[1]}},
            "fit": eng.fit_result.as_dict(), "fit_draws": eng.fit_result.draws_used, "verdict": verdict,
            "fakes": fakes, "candidates": pool_note, "best_ticket": best, "portfolio": port,
            "p_any_portfolio": p_port, "p_any_design": p_design}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the update_all step**

In `update_all.py`, append to `FIRST_WAVE` after the "Lotto 5/35 study" entry:

```python
    ("Engine study: diagnostics and fitted weights", ["engine_study.py", "--no-update"], "engine_study.txt", True),
```

- [ ] **Step 3: Run it twice and compare**

Run: `python3 lottery/engine_study.py --no-update > /tmp/engine1.txt && python3 lottery/engine_study.py --no-update > /tmp/engine2.txt && diff <(grep -v generated_at /tmp/engine1.txt) <(grep -v generated_at /tmp/engine2.txt) && echo identical`
Expected: `identical`; both runs under 3 minutes; on every fake lottery every |z| < 3; `results/engine_study.json` exists. Then `python3 update_all.py --help` still works (no import error).

---

### Task 10: The app's Engine goal

**Files:**
- Modify: `prediction.py` (`GOALS`, `GAME_GOALS`, `HOW` at lines 33-38; `mega645` lines 103-139; `power655` lines 141-172; `cold_model` neighbourhood line 174; `lotto535` lines 185-225)

**Interfaces:**
- Consumes: `engine.Engine`, `engine.candidates`, `coverage.design`, existing `simulated_any`, `EXACT_LIMIT`, `SIMS`.
- Produces: `engine_model(results, n_balls, size, k, seed, status) -> (tickets, lines)` where `size` is numbers per ticket, `k` is ticket count; each game function handles `goal == "engine"`.

- [ ] **Step 1: Add the goal tables**

Replace lines 33-38 of `prediction.py`:

```python
GOALS = {"win": ("Most chance to win", "every number used, least overlap"),
         "value": ("Best value", "bigger jackpot share, avoids popular numbers"),
         "cold": ("Cold-number model", "least-drawn numbers first, typical shapes"),
         "engine": ("Engine", "five-term scorer with fitted weights, greedy portfolio")}
GAME_GOALS = {"645": ("win", "value", "cold", "engine"), "655": ("win", "value", "cold", "engine"),
              "535": ("win", "cold", "engine")}
HOW = {"win": "most chance to win, least overlap", "value": "best value, avoiding popular numbers",
       "cold": "cold-number model, typical shapes", "engine": "engine: fitted five-term scorer"}
```

- [ ] **Step 2: Add the helper next to `cold_model`**

Insert after `cold_model` (before `def lotto535`):

```python
def engine_model(results, n_balls, size, k, seed, status):
    try:
        import numpy as np
        from engine import Engine, candidates
    except ImportError:
        return None, ["the Engine goal needs numpy: pip3 install numpy"]
    status(f"Fitting the engine on {len(results):,} draws")
    eng = Engine(n_balls, size, seed=seed)
    eng.extend(results)
    fit = eng.fit()
    if math.comb(n_balls, size) <= candidates.MAX_ALL:
        status(f"Scoring all {math.comb(n_balls, size):,} tickets")
        pool = candidates.all_tickets(n_balls, size)
    else:
        status("Scoring 1,000,000 sampled tickets")
        pool = candidates.sample(n_balls, size, 1_000_000, np.random.default_rng((seed, 1)))
    tickets = eng.portfolio(k, pool)
    terms = fit.as_dict()
    flagged = [name for name, v in terms.items() if abs(v["z"]) > 3]
    lines = ["engine weights ± error: " + " · ".join(f"{name} {v['w']:+.2f}±{v['se']:.2f}" for name, v in terms.items()),
             "all within error bars: nothing beyond a fair lottery" if not flagged
             else "beyond 3 errors: " + ", ".join(flagged) + " (check the study before trusting it)"]
    return tickets, lines
```

- [ ] **Step 3: Wire `mega645`**

In `mega645`, replace the block from `cold_line = None` to the `else:` branch with:

```python
    extra = []
    if goal == "cold":
        tickets, cold_line = cold_model([d["result"] for d in draws], s.N_BALLS, 6, k, seed)
        extra = [cold_line]
    elif goal == "engine":
        tickets, extra = engine_model([d["result"] for d in draws], s.N_BALLS, 6, k, seed, status)
        if tickets is None:
            tickets = coverage.design(k, s.N_BALLS, seed)
    elif goal == "win":
        tickets = coverage.design(k, s.N_BALLS, seed)
    else:
        tickets = spread_design(o.optimise, scorer, k, s.N_BALLS, seed)
```

Replace the `p_any = ...` statement with a helper and the comparison:

```python
    def any_prize(ts):
        if k <= EXACT_LIMIT:
            return float(v.summarise(v.enumerate_all(ts), k)["any"])
        return simulated_any(ts, s.N_BALLS, seed)

    p_any = any_prize(tickets)
    if goal == "engine" and len(extra) == 2:
        extra.append(f"any prize: this set {p_any * 100:.2f}%, most-chance design"
                     f" {any_prize(coverage.design(k, s.N_BALLS, seed)) * 100:.2f}%")
```

And change the end of the `"forecast"` entry from `+ ([cold_line] if cold_line else [])` to `+ extra`.

- [ ] **Step 4: Wire `power655`**

Same shape inside `power655`: replace from `cold_line = None` through the `else:` branch with

```python
        extra = []
        if goal == "cold":
            tickets, cold_line = cold_model([d["result"] for d in draws], sp.N, 6, k, seed)
            extra = [cold_line]
        elif goal == "engine":
            tickets, extra = engine_model([d["result"] for d in draws], sp.N, 6, k, seed, status)
            if tickets is None:
                tickets = coverage.design(k, sp.N, seed)
        elif goal == "win":
            tickets = coverage.design(k, sp.N, seed)
        else:
            tickets = spread_design(o.optimise, scorer, k, sp.N, seed)
```

then replace the `exact = ...` statement with

```python
        def any_prize(ts):
            if k <= EXACT_LIMIT:
                return sp.exact_summary(ts)[0]
            return {"any": simulated_any(ts, sp.N, seed), **sp.jackpot_chances(ts)}

        exact = any_prize(tickets)
        if goal == "engine" and len(extra) == 2:
            extra.append(f"any prize: this set {exact['any'] * 100:.2f}%, most-chance design"
                         f" {any_prize(coverage.design(k, sp.N, seed))['any'] * 100:.2f}%")
```

and change `+ ([cold_line] if cold_line else [])` to `+ extra`.

- [ ] **Step 5: Wire `lotto535`**

Replace from `cold_line = None` through `tickets = ls.with_specials(mains)` with:

```python
    extra = []
    rows, _ = ls.load_draws()
    if goal == "cold":
        mains, cold_line = cold_model([r["result"] for r in rows], ls.N, ls.K, k, seed)
        extra = [cold_line]
    elif goal == "engine":
        mains, extra = engine_model([r["result"] for r in rows], ls.N, ls.K, k, seed, status)
        if mains is None:
            mains = coverage.design(k, ls.N, seed, size=ls.K)
    else:
        mains = coverage.design(k, ls.N, seed, size=ls.K)
    tickets = ls.with_specials(mains)
```

Replace the `if k <= 2 * EXACT_LIMIT:` block with a helper:

```python
    def any_prize(ts):
        if k <= 2 * EXACT_LIMIT:
            money, p = ls.exact_money(ts, dict(ls.BASE, jackpot=f["jackpot"]))
            return money, p
        return None, simulated_any([m for m, _ in ts], ls.N, seed, size=ls.K)

    money, p_real = any_prize(tickets)
    if money is not None:
        total = sum(money.values())
        p_refund = sum(w for v, w in money.items() if v > 0) / total
    else:
        p_refund = 1 - (1 - p_real) * (1 - min(k, ls.S) / ls.S)
    if goal == "engine" and len(extra) == 2:
        p_design = any_prize(ls.with_specials(coverage.design(k, ls.N, seed, size=ls.K)))[1]
        extra.append(f"any prize: this set {p_real * 100:.2f}%, most-chance design {p_design * 100:.2f}%")
```

and replace `if cold_line:\n        lines.append(cold_line)` with `lines += extra`.

- [ ] **Step 6: Check every path**

Run each and read the output:

```bash
python3 prediction.py --game 645 --tickets 10 --goal engine --no-update
python3 prediction.py --game 655 --tickets 5 --goal engine --no-update
python3 prediction.py --game 535 --tickets 10 --goal engine --no-update
python3 prediction.py --game 645 --tickets 5 --goal cold --no-update
python3 prediction.py --game 535 --tickets 3 --goal win --no-update
python3 -m py_compile prediction.py
```

Expected: the engine runs print tickets, the "engine weights ± error" line, the verdict line and the "any prize" comparison; the cold and win runs are unchanged. Then the menu path: drive `python3 prediction.py` through the scratchpad pty test (`/private/tmp/claude-502/-Users-khoa-vu-Desktop-KhoaVu-AI/3982054c-d275-48ed-b1d5-f0cd7ab260a4/scratchpad/fetch_paths_test.py`) and confirm the goal menu shows four entries for 645 and three for 535. Finally `python3 -m unittest tests.test_engine` again.

---

### Task 11: README and hand-over

**Files:**
- Modify: `README.md` (line 10 dependency sentence; the model table after the `cold_weighted.py` row, about line 135; the study table after `lotto535_study.py`; a new section "The engine" before "Deeper studies")

- [ ] **Step 1: Edit the README**

Change line 10 to: `Python 3.9 or newer. Standard library only, except the engine (` `engine/`, the ` `engine_ensemble` ` model, ` `engine_study.py` ` and the app's Engine goal), which needs numpy: ` `pip3 install numpy` `. Run every command from the repo root.`

Add to the model table after the `cold_weighted.py` row:

```
| `engine_ensemble.py` | the numpy engine: Bayes (decayed Dirichlet counts), Markov transitions, gap spread, four-way entropy and overlap with the last draw, combined with weights fitted by conditional logit on the draws so far and refitted every 100 draws; scores 200,000 sampled tickets and plays the best | the seven-module engine specification, with the weights fitted instead of hand-set |
```

Add to the study table after `lotto535_study.py`:

```
| `engine_study.py` | the engine on each game: chi-square, runs and KS diagnostics, the fitted weights with error bars on the real draws and on fake lotteries, the best ticket and a 10-ticket portfolio against the covering design |
```

Add a section before "## Deeper studies":

```
## The engine

`engine/` is a vectorized scorer. Every ticket gets five terms: how much the decayed counts favour its numbers
(Bayes), how often its numbers followed the last draw (Markov), how evenly it is spread (gap), how varied its
decades, residues and last digits are (entropy), and how many numbers it shares with the last draw (overlap). The
composite is a weighted sum. The weights are not chosen by hand: each past draw is treated as the one ticket that
came up against 1,000 random tickets that did not, and the weights that best tell them apart are fitted with an error
bar. On a fair lottery every weight sits at 0 ± its error bar, which is what the fake lotteries in `engine_study.py`
show and what the real draws show too. The app's fourth ticket goal, "Engine", prints the weights, a verdict, and the
set's any-prize chance next to the covering design's.

python3 -m unittest tests.test_engine        # the engine's own tests
python3 lottery/engine_study.py              # diagnostics, weights, portfolio for each game
python3 prediction.py --game 645 --tickets 10 --goal engine
```

- [ ] **Step 2: Final verification**

Run in order and record the outputs:

```bash
python3 -m unittest tests.test_engine
time python3 lottery/backtest_models.py --no-update
python3 lottery/engine_study.py --no-update
python3 /private/tmp/claude-502/-Users-khoa-vu-Desktop-KhoaVu-AI/3982054c-d275-48ed-b1d5-f0cd7ab260a4/scratchpad/fetch_paths_test.py
python3 /private/tmp/claude-502/-Users-khoa-vu-Desktop-KhoaVu-AI/3982054c-d275-48ed-b1d5-f0cd7ab260a4/scratchpad/shadow_check.py
grep -rn "#" engine models/engine_ensemble.py lottery/engine_study.py tests/test_engine.py | grep -v "^.*#!/" || echo "no comments"
```

Expected: tests pass; backtest under 5 minutes with 21 models per game; study deterministic; fetch paths pass; the shadow check reports 0 real bugs; no `#` comments in the new code.

- [ ] **Step 3: Hand over the commit**

Give Khoa:

```bash
cd ~/Desktop/KhoaVu_AI/vietlott-analysis && git add -A && git commit -m "Add the numpy scoring engine with fitted weights, its study and app goal" && git push
```

with the measured backtest time, the engine's z per game, the fitted weights on the real draws, and the fake-lottery z ranges.
