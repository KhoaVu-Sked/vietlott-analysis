# Lottery analysis engine: design

Date: 2026-09-28. Status: approved in conversation, awaiting review of this document.

## Goal

A vectorized, numpy-based scoring and portfolio engine for Vietlott Mega 6/45, Power 6/55 and Lotto 5/35, built from
the seven-module specification Khoa supplied, with one change: the ensemble weights are fitted from the draw history
instead of set by hand. The engine plugs into the existing walk-forward backtest as a model and into the app as a
fourth ticket goal, so it is judged exactly like the other twenty models.

## Non-goals

- No change to any existing model, script or ticket goal.
- No scipy, no Python above 3.9, no dependency outside numpy for the engine and none at all elsewhere.
- No modelling of the Power 6/55 bonus ball or the Lotto 5/35 special number; the app fills specials as today.
- No multi-ticket scoring in the backtest harness; it scores one ticket per draw as now.
- No new command-line interface: the existing app and scripts are the interface.

## Constraints

- Python 3.9.6 on Khoa's Mac: no `match`, no `X | Y` type unions; numpy 2.0.2 is installed, scipy is not.
- The repo stays standard-library only outside `engine/`; a missing numpy must degrade with a message, never a traceback.
- Repo rules: no code comments; one docstring per module and class carrying the formula in LaTeX (the spec asks for
  them); no docstring that restates a signature.
- Everything seeded and deterministic: two runs on the same data give the same tickets, weights and JSON.
- Full three-game backtest (`backtest_models.py`) under 5 minutes wall-clock on the same machine.

## Layout

```
engine/
  __init__.py        Engine(n_balls, k, alpha0=1.0, decay=0.005, eps=1e-6): the public entry point
  diagnostics.py     chi_square_uniformity_test, runs_test_autocorrelation, kolmogorov_smirnov_sum_test
                     (runs test on the draw sums above/below their median; KS against the exact sum distribution
                     of a uniform k-subset, built by dynamic programming over 1..N)
  bayes.py           BayesianDirichletModel
  gaps.py            SpatialGapAnalyzer
  entropy.py         InformationEntropyScorer
  markov.py          MarkovTransitionModel
  overlap.py         hypergeometric overlap term
  scorer.py          JointEnsembleScorer and the conditional-logit weight fit
  portfolio.py       LotteryWheelingOptimizer
  candidates.py      sample(n_balls, k, m, rng) and all_tickets(n_balls, k)
models/engine_ensemble.py   harness adapter
lottery/engine_study.py     diagnostics and fitted weights per game -> results/engine_study.json
tests/test_engine.py        unittest
docs/superpowers/specs/2026-09-28-lottery-engine-design.md   this document
```

`prediction.py`, `update_all.py` and `README.md` are edited; nothing else existing changes.

## Data flow

Draws arrive as today: a list of sorted lists of k main numbers, oldest first. `Engine.add(draw)` validates (k distinct
integers in 1..N, else `ValueError`) and updates, per draw: the 0/1 history matrix, the decayed Dirichlet counts, the
Markov transition counts, and two snapshot rows of per-number log-probabilities as they stood *before* that draw
(`LB[t]`, `LM[t]`). The snapshots make the walk-forward fit a single pass. `Engine.extend(draws)` adds only the draws
not yet seen, so the harness never rebuilds from scratch.

Candidates are an (M × k) int16 matrix of sorted rows. `Engine.features(cands)` returns an (M × 5) float64 matrix in
the fixed column order bayes, markov, gap, entropy, overlap. The gap and entropy columns depend only on the ticket, so
callers cache them per candidate matrix; the other three are a gather over the current per-number vectors and the last
draw. `Engine.score(cands, w)` is the dot product with the weights; `Engine.best(cands)` is its argmax;
`Engine.portfolio(m, cands, max_overlap)` is the greedy picker.

## The five terms

Each term is centred so that 0 is the value a fair lottery gives; the weights then measure departure from fair.

- Bayes (`bayes.py`): counts a_i = alpha0 + sum_t exp(-decay (T - t)) 1[i in D_t]; p_i = a_i / sum a.
  Term = sum over i in T of log(N p_i).
- Markov (`markov.py`): M_ij = (count of j at s given i at s-1, plus eps) / row total. Given the last draw D,
  q_j = mean over i in D of M_ij. Term = sum over j in T of log(N q_j). With no last draw the term is 0.
- Gap (`gaps.py`), changed 2026-09-29 at Khoa's request: real draws are rarely evenly spaced, so the term compares
  the ticket's gap combination with history instead of rewarding even spacing. The k-1 in-between gaps are sorted
  into next to (0), close (1-3), medium (4-8) and wide (9+); the counts per kind are the combination c. f(c) is its
  exact share under a fair machine, counted with generating polynomials; after t draws with n_t(c) of combination c,
  the term is log((n_t(c) + 20) / (t f(c) + 20)), a Gamma-Poisson estimate that shrinks every combination by the same
  20 pseudo-counts, so a rare combination seen once barely moves it (a first version shrank by 50 f(c), which let
  one-off rare combinations reach +4 and made the engine pick runs like 38 39 40 41 42); 0 while history matches a
  fair machine. It is snapshotted before each draw like the Bayes and Markov terms. The old evenness score
  KL(q || uniform over k+1), the gap variance and the min/max ratio are diagnostics only.
- Entropy (`entropy.py`): Shannon entropy over bins of ten (ceil(N/10) bins), residues mod 3, residues mod 5 and last
  digit, each divided by log(min(k, bins)) and averaged into a score in [0, 1]. Term = log(max(score, 1e-9)).
- Overlap (`overlap.py`): o = |T ∩ D_last|. Term = log(C(k, o) C(N-k, k-o) / C(N, k)); 0 with no last draw.

## Fitting the weights

Conditional logit. For each past draw t >= 10 (the harness only asks from draw 50, so the first fit sees 40 draws),
the drawn ticket D_t is the chosen alternative against S = 1,000 tickets
sampled uniformly from all C(N, k) with a generator seeded by (seed, t). Features for draw t use `LB[t]`, `LM[t]` and
D_{t-1} only. Log-posterior = sum_t [w·f_t0 - log sum_s exp(w·f_ts)] - |w|² / (2 sigma²), sigma = 1. Newton's method
with step halving on the objective, at most 50 iterations, stop when the step is below 1e-8. Standard errors are the
square roots of the diagonal of the inverse of the negative Hessian at the optimum. `Fit` carries `w`, `se`,
`z = w / se`, `draws_used`, `iterations`, `converged`. A singular Hessian returns w = 0, se = inf, `converged = False`.

On a fair lottery every z sits at 0 ± 1 up to noise. That is the test the engine exists to run.

## Candidates

- `candidates.sample(n_balls, k, m, rng)`: m sorted rows drawn uniformly from all C(N, k); duplicates allowed for
  scoring, removed (`numpy.unique` on rows) before the portfolio.
- `candidates.all_tickets(n_balls, k)`: every combination, generated in numpy blocks, refused with `ValueError` when
  C(N, k) > 10,000,000. Used for 6/45 (8,145,060 rows, about 50 MB as int8).
- Backtest: 200,000 sampled candidates per refit period. App: all tickets for 6/45, 1,000,000 sampled for 6/55 and 5/35.

## Portfolio

`LotteryWheelingOptimizer.select(cands, scores, m, max_overlap=k-2)`: sort by score, take the best, then repeatedly the
best remaining candidate whose overlap with every chosen ticket is at most `max_overlap`; stop at m tickets or when the
candidates run out. Returns the tickets in score order. The app reports the set's exact any-prize chance next to the
`coverage.py` design's, because k-2 allows 4 shared numbers on 6/45 while the covering design allows about 1.

## Harness adapter (`models/engine_ensemble.py`)

Same shape as `models/cold_weighted.py`: a module-level engine kept between calls and reset when the first or last draw
changes, the game changes or the history shrinks. `predict(past, n_balls)` returns `[1.0] * n_balls` while
`len(past) < 50`; otherwise it extends the engine, refits and resamples 200,000 candidates when `len(past) % 100 == 50`,
scores, and returns 1.0 for every number and 1.0 + 1e-9 for the best ticket's numbers. If numpy cannot be imported the
module raises `ImportError("engine_ensemble needs numpy: pip3 install numpy")`; `backtest_models.load_models` catches
it: `load_models` imports each model file once in the parent process, prints the message for any that fails, and
leaves that model out.

## App (`prediction.py`)

Fourth goal `engine` ("Engine", "fitted five-term scorer, greedy portfolio") in `GOALS`, `HOW` and every entry of
`GAME_GOALS`. A helper `engine_model(results, n_balls, size, k, seed)` (size = numbers per ticket, k = tickets, the
same convention as `cold_model`) builds the engine on the full history, fits once,
builds the candidate matrix (all tickets for N = 45, else 1,000,000 sampled), and returns the k-ticket portfolio and the
forecast lines:

- `engine weights ± error: bayes +0.02±0.05 · markov −0.01±0.04 · gap … · entropy … · overlap …`
- a verdict: `all within error bars: nothing beyond a fair lottery`, or the names of any term with |z| > 3
- `any prize: this set 22.1%, most-chance design 23.0%`

Each game function calls it when `goal == "engine"`; 5/35 passes the mains through `with_specials` as for the cold
goal. When numpy is missing the goal falls back to the most-chance tickets and shows the ImportError message as its
first forecast line.

## Study (`lottery/engine_study.py`)

`python3 lottery/engine_study.py [--fakes 2] [--no-update] [--seed 2026]`. Per game: the three diagnostics with
p-values (`diagnostics.py` carries its own chi-square, normal and Kolmogorov tails, so `engine/` never imports from
`lottery/`); the fit on the real history and on each fake with
weights, errors and z; the best single ticket; a 10-ticket portfolio with its any-prize chance from the app's 200,000-draw
simulation next to the covering design's. Writes
`results/engine_study.json` and prints a short report. Added to `update_all.py`'s first wave as a study step that
`--quick` skips.

## README

A section "The engine" after "The models": what the five terms are, how the weights are fitted, how to read the error
bars, the four commands. A `engine_ensemble.py` row in the model table, an `engine_study.py` row in the study table,
and the dependency line changed to: numpy for the engine only, standard library everywhere else.

## Tests (`tests/test_engine.py`, unittest)

1. Diagnostics: a seeded fair simulated history gives all three p-values in [0.001, 0.999]; a rigged history
   (numbers 1..10 drawn three times as often) gives a chi-square p below 1e-6.
2. Bayes: decay 0 and alpha0 0 reproduce the raw counts; probabilities sum to 1.
3. Markov: every row of M sums to 1; with no history q is uniform.
4. Gap: an evenly spread ticket scores 0 and 1-2-3-4-5-6 scores below it.
5. Entropy: score in [0, 1]; a spread ticket scores above 1-2-3-4-5-6.
6. Overlap: the k+1 probabilities sum to 1.
7. Scorer: the vectorized features equal a plain-Python computation on 50 tickets to 1e-12; the fit on a fake lottery
   of 1,500 draws gives every |z| < 3; the fit on the rigged history gives Bayes z > 5.
8. Portfolio: every pair respects the overlap cap; exactly m tickets when enough candidates exist; deterministic.
9. Adapter: returns N scores; the same history twice gives the same ticket; a history under 50 draws is uniform.

Then, by hand: the unit tests, `engine_study.py`, the full `backtest_models.py`, the app's Engine goal in menu and plain
mode for all three games, and the existing fetch-path test.

## Acceptance

- `python3 -m unittest tests.test_engine` passes.
- `python3 lottery/backtest_models.py --no-update` finishes under 5 minutes with `engine_ensemble` present for all
  three games in `results/result.json`.
- `results/engine_study.json` shows every |z| < 3 on the fake histories.
- `python3 prediction.py --game 535 --tickets 10 --goal engine --no-update` prints tickets with specials and the three
  engine lines; the menu path shows the same.
- Two consecutive study runs produce identical JSON apart from the timestamp.
