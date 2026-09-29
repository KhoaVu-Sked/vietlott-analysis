import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lottery"))

from engine import Engine, bayes, candidates, diagnostics, entropy, gaps, markov, overlap, portfolio, scorer


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

    def test_gap_kinds_classify_the_in_between_gaps(self):
        g = gaps.SpatialGapAnalyzer(45, 6)
        cands = np.array([[1, 2, 3, 4, 5, 6], [1, 5, 9, 20, 30, 40], [6, 12, 18, 25, 34, 41]], dtype=np.int16)
        self.assertEqual(g.kinds(cands).tolist(), [g.code(5, 0, 0, 0), g.code(0, 2, 0, 3), g.code(0, 0, 5, 0)])

    def test_fair_gap_shares_match_counting_every_ticket(self):
        import itertools
        for n, k in ((20, 5), (16, 6), (12, 4)):
            g = gaps.SpatialGapAnalyzer(n, k)
            rows = np.array(list(itertools.combinations(range(1, n + 1), k)), dtype=np.int16)
            counted = np.bincount(g.kinds(rows), minlength=len(g.fair)) / len(rows)
            self.assertTrue(np.allclose(g.fair, counted, rtol=0, atol=1e-15), (n, k))
        for n, k in ((45, 6), (55, 6), (35, 5)):
            self.assertAlmostEqual(float(gaps.SpatialGapAnalyzer(n, k).fair.sum()), 1.0, places=12)

    def test_gap_term_starts_fair_and_learns_from_draws(self):
        g = gaps.SpatialGapAnalyzer(45, 6)
        self.assertTrue(np.allclose(g.log_terms(), 0.0))
        g.update([1, 5, 9, 20, 30, 40])
        seen, other = g.code(0, 2, 0, 3), g.code(0, 2, 2, 1)
        f, fo = float(g.fair[seen]), float(g.fair[other])
        self.assertAlmostEqual(float(g.log_terms()[seen]), math.log((1 + 20) / (f + 20)), places=12)
        self.assertAlmostEqual(float(g.log_terms()[other]), math.log(20 / (fo + 20)), places=12)

    def test_rare_gap_combinations_cannot_swing_the_term(self):
        g = gaps.SpatialGapAnalyzer(45, 6)
        for d in fair_history(45, 6, 1500, 15):
            g.update(d)
        self.assertLess(float(np.abs(g.log_terms()).max()), 0.5)
        rare = int(np.argmin(np.where(g.fair > 0, g.fair, 1.0)))
        g.seen[rare] += 1
        self.assertLess(float(g.log_terms()[rare]), math.log(1 + 2 / 20))

    def test_gap_diagnostics_keep_the_old_evenness_score(self):
        g = gaps.SpatialGapAnalyzer(13, 6)
        cands = np.array([[2, 4, 6, 8, 10, 12], [1, 2, 3, 4, 5, 6]], dtype=np.int16)
        self.assertEqual(g.gaps(cands[:1]).tolist(), [[1, 1, 1, 1, 1, 1, 1]])
        diag = g.diagnostics(cands)
        self.assertAlmostEqual(float(diag["unevenness"][0]), 0.0, places=12)
        self.assertGreater(float(diag["unevenness"][1]), 0.0)
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


class TestDiagnostics(unittest.TestCase):
    def test_tails_match_known_values(self):
        import power645_study as independent
        self.assertAlmostEqual(diagnostics.chi2_sf(3.841, 1), 0.05, places=3)
        self.assertAlmostEqual(diagnostics.chi2_sf(0.0, 5), 1.0, places=12)
        for x, df in ((60.0, 44), (10.0, 5), (200.0, 54), (0.5, 3)):
            self.assertAlmostEqual(diagnostics.chi2_sf(x, df), independent.chi2_sf(x, df), places=6)
        self.assertAlmostEqual(diagnostics.norm_sf(1.959964), 0.025, places=5)
        self.assertAlmostEqual(diagnostics.kolmogorov_sf(1.36), 0.05, places=2)
        self.assertAlmostEqual(diagnostics.kolmogorov_sf(1.0), 0.2700, places=3)
        self.assertAlmostEqual(diagnostics.kolmogorov_sf(0.9999), diagnostics.kolmogorov_sf(1.0001), places=3)

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

    def test_chi_square_is_calibrated_for_draws_without_replacement(self):
        stats = [diagnostics.chi_square_uniformity_test(fair_history(45, 6, 200, 1000 + s), 45)[0] for s in range(300)]
        self.assertLess(abs(float(np.mean(stats)) - 44), 2.5, float(np.mean(stats)))

    def test_rigged_history_fails_chi_square(self):
        _, p = diagnostics.chi_square_uniformity_test(rigged_history(45, 6, 1500, 1), 45)
        self.assertLess(p, 1e-6)


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
        fair = gaps.SpatialGapAnalyzer(45, 6).fair

        def kind(row):
            n = [0, 0, 0, 0]
            for a, b in zip(row, row[1:]):
                v = b - a - 1
                n[0 if v == 0 else 1 if v <= 3 else 2 if v <= 8 else 3] += 1
            return n[0] * 216 + n[1] * 36 + n[2] * 6 + n[3]

        seen = [kind(d) for d in h]
        for row, f in zip(cands.tolist(), feats):
            self.assertAlmostEqual(f[0], sum(pb[x - 1] for x in row), places=10)
            self.assertAlmostEqual(f[1], sum(pm[x - 1] for x in row), places=10)
            c = kind(row)
            ratio = (seen.count(c) + 20) / (len(h) * fair[c] + 20)
            self.assertAlmostEqual(f[2], math.log(ratio), places=10)
            self.assertAlmostEqual(f[4], table[len(last & set(row))], places=10)
        codes, ent = eng.static_terms(cands)
        self.assertEqual(codes.tolist(), [kind(r) for r in cands.tolist()])
        self.assertTrue(np.allclose(feats[:, 3], ent))

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
        self.assertEqual(len(eng.snap_gaps), 60)
        self.assertTrue(np.allclose(eng.snap_gaps[0], 0.0))
        self.assertTrue(np.allclose(eng.snap_gaps[25], fresh.gaps.log_terms()))
        self.assertTrue(np.allclose(eng.features(np.array([h[30]], dtype=np.int16), at=25)[0, :3],
                                    fresh.features(np.array([h[30]], dtype=np.int16))[0, :3]))

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

    def test_fit_finds_a_machine_that_favours_neighbours(self):
        r = random.Random(14)
        h = []
        for _ in range(1500):
            want = r.random() < 0.5
            d = sorted(r.sample(range(1, 46), 6))
            while want and sum(b - a == 1 for a, b in zip(d, d[1:])) < 2:
                d = sorted(r.sample(range(1, 46), 6))
            h.append(d)
        eng = Engine(45, 6)
        eng.extend(h)
        self.assertGreater(float(eng.fit().z[2]), 3, eng.fit_result.as_dict())

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
        import contextlib
        import io
        import tempfile
        import backtest_models as bm
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.py").write_text("raise ImportError('needs a missing package')\n")
            (Path(tmp) / "fine.py").write_text("def predict(past, n_balls):\n    return [1.0] * n_balls\n")
            keep = bm.MODELS_DIR
            bm.MODELS_DIR = Path(tmp)
            try:
                with contextlib.redirect_stdout(out):
                    loaded = bm.load_models(None)
            finally:
                bm.MODELS_DIR = keep
        self.assertEqual(sorted(loaded), ["fine"])
        self.assertIn("broken skipped: needs a missing package", out.getvalue())


class TestAppGoal(unittest.TestCase):
    def test_plain_report_handles_every_goal(self):
        import prediction as app
        r = {"game": "Mega 6/45", "draw": 1569, "date": "2026-09-30", "n_draws": 1568, "forecast": ["line"],
             "tickets": [[1, 2, 3, 4, 5, 6]], "p_any": 0.02, "p_one": 0.0238, "p_jackpot": 1 / 8_145_060, "ev": 4000}
        for goal in app.GOALS:
            with self.subTest(goal=goal):
                text = app.plain_report(dict(r, goal=goal), 1, "not updated")
                self.assertIn("every combination is equally likely", text)
                self.assertIn(app.GOALS[goal][0].lower(), text)

    def test_engine_forecast_lines_fit_the_tui(self):
        import prediction as app
        tickets, lines = app.engine_model(fair_history(35, 5, 300, 13), 35, 5, 4, 2026, lambda text: None)
        self.assertEqual(len(tickets), 4)
        self.assertTrue(any("bayes" in line for line in lines) and any("overlap" in line for line in lines), lines)
        for line in lines:
            self.assertLessEqual(len(line), 88 - 5, line)

    def test_plain_mode_refuses_a_goal_the_game_does_not_offer(self):
        import subprocess
        run = subprocess.run([sys.executable, "prediction.py", "--game", "535", "--tickets", "3", "--goal", "value",
                              "--no-update"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 2, run.stdout[-400:])
        self.assertIn("--goal value is not offered for Lotto 5/35", run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        self.assertNotIn("best value", run.stdout)

    def test_engine_goal_without_numpy_falls_back_and_says_so(self):
        import subprocess
        code = ("import runpy, sys; sys.modules['numpy'] = None; "
                "sys.argv = ['prediction.py', '--game', '535', '--tickets', '3', '--goal', 'engine', '--no-update']; "
                "runpy.run_path('prediction.py', run_name='__main__')")
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stderr[-800:])
        self.assertIn("the Engine goal needs numpy", run.stdout)
        self.assertIn("goal: most chance to win", run.stdout)
        self.assertNotIn("engine's top-scored tickets", run.stdout)


class TestStudyScript(unittest.TestCase):
    def test_engine_study_without_numpy_is_a_skip_for_update_all(self):
        import subprocess
        code = ("import runpy, sys; sys.modules['numpy'] = None; sys.path.insert(0, 'lottery'); "
                "sys.argv = ['lottery/engine_study.py', '--no-update']; "
                "runpy.run_path('lottery/engine_study.py', run_name='__main__')")
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
        output = run.stdout + run.stderr
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("engine_study needs numpy: pip3 install numpy", output)
        self.assertNotIn("Traceback", output)
        import update_all
        self.assertTrue(any(phrase in output for phrase in update_all.FINE_EXITS), update_all.FINE_EXITS)


class TestDocs(unittest.TestCase):
    def test_every_engine_class_documents_its_formula(self):
        import importlib
        import inspect
        import pkgutil
        import engine
        found = [engine.Engine]
        for info in pkgutil.iter_modules(engine.__path__):
            mod = importlib.import_module(f"engine.{info.name}")
            found += [c for _, c in inspect.getmembers(mod, inspect.isclass) if c.__module__ == mod.__name__]
        self.assertGreaterEqual(len(found), 8)
        for cls in found:
            with self.subTest(cls=cls.__name__):
                self.assertIn("$", cls.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
