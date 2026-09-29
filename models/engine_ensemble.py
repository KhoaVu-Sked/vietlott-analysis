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
