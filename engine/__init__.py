"""The lottery engine: Engine and the modules it composes."""

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
    """The five-term ensemble over a draw history. State is incremental: each draw updates
    the Bayes and Markov terms and stores the per-number log terms as they stood before it, so a walk-forward fit
    is one pass. A ticket's features are $(\\text{bayes}, \\text{markov}, \\text{gap}, \\text{entropy},
    \\text{overlap})$ and its score is $w \\cdot f$ with $w$ from the conditional-logit fit in scorer.py."""

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
