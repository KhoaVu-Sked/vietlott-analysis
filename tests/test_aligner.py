import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lottery"))

import aligner
import backtest_models as bm


class TestAligner(unittest.TestCase):
    def test_equal_weights_pick_the_numbers_most_models_share(self):
        tickets = {"a": [[1, 2, 3]], "b": [[1, 2, 4]], "c": [[1, 5, 6]]}
        out = aligner.align(tickets, [[7, 8, 9]], 10, 3)
        self.assertEqual(len(out["steps"]), 1)
        self.assertTrue({1, 2} <= set(out["steps"][0]["ticket"]))
        self.assertEqual(out["hits"], [0])

    def test_one_draw_moves_each_weight_by_its_catch_against_luck(self):
        tickets = {"right": [[1, 2, 3]], "wrong": [[4, 5, 6]]}
        out = aligner.align(tickets, [[1, 2, 3]], 10, 3)
        luck = 3 * 3 / 10
        ratio = math.exp(aligner.STEP * (3 - luck)) / math.exp(aligner.STEP * (0 - luck))
        self.assertAlmostEqual(out["weights"]["right"] / out["weights"]["wrong"], ratio, places=12)
        self.assertAlmostEqual(sum(out["weights"].values()), 1.0, places=12)

    def test_trust_moves_to_the_model_that_keeps_catching(self):
        draws = [[1, 2, 3]] * 30
        tickets = {"right": [[1, 2, 3]] * 30, "wrong": [[4, 5, 6]] * 30, "other": [[7, 8, 9]] * 30}
        out = aligner.align(tickets, draws, 10, 3, next_tickets={"right": [1, 2, 3], "wrong": [4, 5, 6],
                                                                  "other": [7, 8, 9]})
        self.assertEqual(out["steps"][-1]["leaders"][0][0], "right")
        self.assertEqual(out["steps"][-1]["ticket"], [1, 2, 3])
        self.assertEqual(out["next"], [1, 2, 3])

    def test_a_draw_never_shapes_its_own_ticket(self):
        tickets = {"a": [[1, 2, 3], [1, 2, 3], [4, 5, 6]], "b": [[4, 5, 6], [4, 5, 6], [1, 2, 3]]}
        base = aligner.align(tickets, [[1, 2, 3], [7, 8, 9], [1, 2, 3]], 10, 3)
        changed = aligner.align(tickets, [[1, 2, 3], [4, 5, 6], [1, 2, 3]], 10, 3)
        self.assertEqual(base["steps"][1]["ticket"], changed["steps"][1]["ticket"])
        self.assertEqual(base["steps"][1]["leaders"], changed["steps"][1]["leaders"])
        self.assertNotEqual(base["steps"][2]["leaders"], changed["steps"][2]["leaders"])

    def test_same_input_gives_the_same_output(self):
        tickets = {"a": [[1, 2, 3], [2, 3, 4]], "b": [[5, 6, 7], [6, 7, 8]]}
        draws = [[1, 5, 9], [2, 6, 10]]
        self.assertEqual(aligner.align(tickets, draws, 10, 3), aligner.align(tickets, draws, 10, 3))


class TestFreeze(unittest.TestCase):
    def test_freeze_writes_a_fingerprinted_pick_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ("655", 1404, [5, 12, 19, 26, 33, 40], {"hot30": 0.2, "fair": 0.1}, 1403)
            path, fresh = aligner.freeze(*args, directory=Path(tmp))
            self.assertTrue(fresh)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["draw"], 1404)
            self.assertEqual(saved["ticket"], [5, 12, 19, 26, 33, 40])
            self.assertEqual(saved["sha256"], aligner.fingerprint(saved))
            self.assertTrue(path.with_suffix(".txt").exists())
            again, fresh_again = aligner.freeze("655", 1404, [1, 2, 3, 4, 5, 6], {}, 1403, directory=Path(tmp))
            self.assertFalse(fresh_again)
            self.assertEqual(json.loads(again.read_text())["ticket"], [5, 12, 19, 26, 33, 40])


class TestHarness(unittest.TestCase):
    def test_run_model_returns_each_ticket_and_the_next_one(self):
        mod = bm.load_module(ROOT / "models" / "hot30.py")
        draws, _ = bm.load_game("535")
        hits, gains, tickets, nxt = bm.run_model(mod, draws[:120], 35, 50, 2026, 5)
        self.assertEqual(len(tickets), 70)
        self.assertEqual(hits, [len(set(t) & set(d)) for t, d in zip(tickets, draws[50:120])])
        self.assertEqual(len(nxt), 5)
        self.assertEqual(len(set(nxt)), 5)

    def test_harness_reports_the_aligned_vote_and_logs_every_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = bm.RESULT, bm.RUNS_LOG, bm.ALIGN_DIR
            bm.RESULT, bm.RUNS_LOG, bm.ALIGN_DIR = Path(tmp) / "r.json", Path(tmp) / "runs.jsonl", Path(tmp)
            try:
                result = bm.run(("535",), names=["fair", "hot30"], fakes=1, fetch=False)
            finally:
                bm.RESULT, bm.RUNS_LOG, bm.ALIGN_DIR = keep
            out = result["games"]["535"]
            aligned = out["models"]["aligned_vote"]
            for field in ("hit_rate_pct", "z_vs_fair", "p_luck_after_holm", "fake_lotteries_z", "holdout", "verdict"):
                self.assertIn(field, aligned)
            self.assertEqual(len(aligned["next_ticket"]), 5)
            steps = [json.loads(line) for line in (Path(tmp) / "aligned_535.jsonl").read_text().splitlines()]
            self.assertEqual(len(steps), aligned["draws"])
            self.assertEqual(steps[0]["draw"], 51)
            self.assertEqual(set(steps[0]), {"draw", "ticket", "hits", "leaders"})
            self.assertTrue(any(line.startswith("  aligned_vote") for line in bm.summary_lines(result)))


if __name__ == "__main__":
    unittest.main()
