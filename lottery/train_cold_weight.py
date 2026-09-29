"""Train the cold_weighted model: try many settings of its learned weight on the older draws, pick the best, then score
that choice on the last 300 draws it never saw, and do the same on fake fair lotteries for comparison.

Usage: python3 lottery/train_cold_weight.py [--fakes 2] [--no-update]
"""

import argparse
import json
import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import backtest_models as bm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cold_weight_training.json"
GRID = {"list_size": (6, 12, 20), "step": (0.5, 2.0, 8.0), "start": (20.0, 80.0)}
START, HOLDOUT = 50, 300


def run(args):
    draws, n_balls, k, params = args
    import model_tools
    model_tools.K = k
    ln = model_tools.ColdLearner(n_balls, k, **params)
    hits, weights = [], []
    for t, d in enumerate(draws):
        if t >= START:
            ticket = ln.tickets(draws[:t])[0]
            hits.append(len(set(ticket) & set(d)))
            weights.append(ln.weight)
        ln.add(d)
    return hits, weights


def rate(hits, k):
    return 100 * sum(hits) / (k * len(hits))


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
    combos = [dict(zip(GRID, values)) for values in product(*GRID.values())]
    jobs, keys = [], []
    for code, game in bm.GAMES.items():
        draws, _ = bm.load_game(code)
        histories = [("real", draws)] + [
            (f"fake {f + 1}", [sorted(r.sample(range(1, game["balls"] + 1), game["k"])) for _ in draws])
            for f, r in enumerate(random.Random(args.seed + 500 + i) for i in range(args.fakes))]
        for label, history in histories:
            for params in combos:
                jobs.append((history, game["balls"], game["k"], params))
                keys.append((code, label, tuple(params.items())))
    with ProcessPoolExecutor() as pool:
        results = dict(zip(keys, pool.map(run, jobs)))
    report = {}
    for code, game in bm.GAMES.items():
        k, n = game["k"], game["balls"]
        luck = 100 * k / n
        print(f"{game['name']}: {len(combos)} settings of the weight trained on the older draws, then checked on the last"
              f" {HOLDOUT} draws; luck catches {luck:.2f}% of the numbers")
        report[code] = {}
        for label in ["real"] + [f"fake {f + 1}" for f in range(args.fakes)]:
            rows = []
            for params in combos:
                hits, weights = results[(code, label, tuple(params.items()))]
                split = len(hits) - HOLDOUT
                rows.append((rate(hits[:split], k), rate(hits[split:], k), params, weights))
            best = max(rows, key=lambda r: r[0])
            train_rates = [r[0] for r in rows]
            hold_rates = [r[1] for r in rows]
            sd_hold = 100 * math.sqrt((k / n) * (1 - k / n) / (k * HOLDOUT))
            print(f"  {label:7s} best setting on the older draws: list {best[2]['list_size']}, step {best[2]['step']},"
                  f" start {best[2]['start']:.0f} -> {best[0]:.2f}% there, {best[1]:.2f}% on the last {HOLDOUT} draws"
                  f" (luck {luck:.2f}% +- {sd_hold:.2f}); all settings: {min(train_rates):.2f}..{max(train_rates):.2f}%"
                  f" there, {min(hold_rates):.2f}..{max(hold_rates):.2f}% after")
            if label == "real":
                w = best[3]
                marks = ", ".join(f"{w[i]:.0f}" for i in range(0, len(w), max(1, len(w) // 10)))
                print(f"          its weight over time, sampled every {max(1, len(w) // 10)} draws: {marks}; now {w[-1]:.0f}")
            report[code][label] = {"best": best[2], "train_pct": best[0], "holdout_pct": best[1], "luck_pct": luck,
                                   "all_train": train_rates, "all_holdout": hold_rates,
                                   "weights": best[3][:: max(1, len(best[3]) // 50)]}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
