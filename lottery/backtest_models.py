"""Test every prediction model in models/ on every past draw, using only the draws before it each time,
and write results/result.json (rewritten on every run).

Usage: python3 lottery/backtest_models.py                        # Mega 6/45, every model in models/
       python3 lottery/backtest_models.py --game 655             # Power 6/55 (main numbers only)
       python3 lottery/backtest_models.py --models hot30 --fakes 5
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import power645_study as s

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULT = ROOT / "results" / "result.json"
RUNS_LOG = ROOT / "results" / "backtest_runs.jsonl"
K = 6
GAMES = {"645": {"name": "Mega 6/45", "balls": 45, "prize": {3: 30_000, 4: 300_000, 5: 10_000_000}},
         "655": {"name": "Power 6/55", "balls": 55, "prize": {3: 50_000, 4: 500_000, 5: 40_000_000}}}


def load_game(game):
    if game == "645":
        draws, gaps, _ = s.load_draws()
    else:
        import power655_study as sp
        draws, gaps = sp.load_draws()
    if gaps:
        raise SystemExit(f"missing draws {gaps[:10]}; run the collector first")
    return [d["result"] for d in draws], [d["id"] for d in draws]


def load_models(names):
    out = {}
    for path in sorted(MODELS_DIR.glob("*.py")):
        if path.name.startswith("_") or (names and path.stem not in names):
            continue
        spec = importlib.util.spec_from_file_location(f"models_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[path.stem] = (mod, hashlib.sha256(path.read_bytes()).hexdigest()[:12])
    if not out:
        raise SystemExit(f"no models found in {MODELS_DIR}")
    return out


def hits_pmf(n_balls):
    total = math.comb(n_balls, K)
    return [math.comb(K, h) * math.comb(n_balls - K, K - h) / total for h in range(K + 1)]


def to_chances(scores):
    if min(scores) < 0 or sum(scores) <= 0:
        return None
    total = sum(scores)
    return [min(max(v * K / total, 1e-4), 0.999) for v in scores]


def log_gain(chances, drawn, n_balls):
    p = K / n_balls
    return sum(math.log(c / p) if i in drawn else math.log((1 - c) / (1 - p)) for i, c in enumerate(chances, 1))


def run_model(mod, draws, n_balls, start, seed):
    rng = random.Random(seed)
    hits, gains = [], []
    for t in range(start, len(draws)):
        scores = [float(v) for v in mod.predict(draws[:t], n_balls)]
        if len(scores) != n_balls:
            raise ValueError(f"predict() must return {n_balls} scores, got {len(scores)}")
        order = sorted(range(1, n_balls + 1), key=lambda i: (-scores[i - 1], rng.random()))
        drawn = set(draws[t])
        hits.append(len(set(order[:K]) & drawn))
        chances = to_chances(scores)
        gains.append(log_gain(chances, drawn, n_balls) if chances else None)
    return hits, gains


def summarise(hits, gains, n_balls, prize):
    n = len(hits)
    pmf = hits_pmf(n_balls)
    mean_h = sum(h * p for h, p in enumerate(pmf))
    var_h = sum((h - mean_h) ** 2 * p for h, p in enumerate(pmf))
    z = (sum(hits) - n * mean_h) / math.sqrt(n * var_h)
    dist = Counter(hits)
    scored = [g for g in gains if g is not None]
    return {"draws": n, "hit_rate_pct": 100 * sum(hits) / (K * n), "fair_hit_rate_pct": 100 * mean_h / K,
            "z_vs_fair": z, "p_luck": s.norm_sf(z),
            "hits_pct": {str(h): 100 * dist[h] / n for h in range(K + 1)},
            "fair_hits_pct": {str(h): 100 * pmf[h] for h in range(K + 1)},
            "prize_draws_pct": 100 * sum(h >= 3 for h in hits) / n, "fair_prize_draws_pct": 100 * sum(pmf[3:]),
            "jackpots": dist[K], "fixed_prizes_won_vnd": sum(prize.get(h, 0) for h in hits), "spent_vnd": n * 10_000,
            "log_score_vs_fair": statistics.fmean(scored) if len(scored) == n else None}


def versions_tried(game, current):
    seen = {name: {h} for name, h in current.items()}
    if RUNS_LOG.exists():
        for line in RUNS_LOG.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["game"] == game:
                    for name, h in row["models"].items():
                        seen.setdefault(name, set()).add(h)
    return sum(len(v) for v in seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(GAMES), default="645")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--fakes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    game = GAMES[args.game]
    n_balls, prize = game["balls"], game["prize"]
    draws, ids = load_game(args.game)
    models = load_models(args.models)
    fakes = [[sorted(r.sample(range(1, n_balls + 1), K)) for _ in draws]
             for r in (random.Random(args.seed + 100 + f) for f in range(args.fakes))]
    split = len(draws) - args.holdout - args.start
    tried = versions_tried(args.game, {name: h for name, (_, h) in models.items()})
    out = {"game": game["name"], "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tested_draws": f"#{ids[args.start]}..#{ids[-1]}", "holdout_draws": f"#{ids[-args.holdout]}..#{ids[-1]}",
           "fair_hit_rate_pct": 100 * K / n_balls, "model_versions_tried_on_this_history": tried,
           "warning": (f"{tried} model versions have been scored on this same history. Expect about 1 in 20 to reach"
                       " p_luck < 0.05 by chance alone; only the hold-out and new live draws can confirm a real edge."),
           "models": {}}
    p_values = {}
    for name, (mod, digest) in models.items():
        hits, gains = run_model(mod, draws, n_balls, args.start, args.seed)
        res = summarise(hits, gains, n_balls, prize)
        res["holdout"] = summarise(hits[split:], gains[split:], n_balls, prize)
        fake_z = []
        for f in fakes:
            fh, fg = run_model(mod, f, n_balls, args.start, args.seed)
            fake_z.append(summarise(fh, fg, n_balls, prize)["z_vs_fair"])
        res["fake_lotteries_z"] = fake_z
        res["file_sha"] = digest
        out["models"][name] = res
        p_values[name] = res["p_luck"]
    adjusted = dict(zip(p_values, s.holm(list(p_values.values()))))
    for name, res in out["models"].items():
        res["p_luck_after_holm"] = adjusted[name]
        hold = res["holdout"]
        if adjusted[name] < 0.05 and hold["p_luck"] < 0.05:
            res["verdict"] = "beats fair on all draws and on the hold-out: freeze it and test on new live draws"
        elif res["z_vs_fair"] > 0:
            res["verdict"] = "a little better than fair, but within what luck produces"
        else:
            res["verdict"] = "no better than a random ticket"
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(out, indent=1))
    with RUNS_LOG.open("a") as fh:
        fh.write(json.dumps({"time": out["generated_at"], "game": args.game,
                             "models": {n: r["file_sha"] for n, r in out["models"].items()}}) + "\n")
    print(f"{game['name']}: tested {out['tested_draws']}, hold-out {out['holdout_draws']};"
          f" a random ticket catches {out['fair_hit_rate_pct']:.2f}% of the numbers")
    print(f"  {'model':14s} {'hit rate':>8s} {'hold-out':>8s} {'z':>6s} {'p luck':>7s} {'fake z range':>14s}  verdict")
    for name, r in sorted(out["models"].items(), key=lambda kv: -kv[1]["z_vs_fair"]):
        print(f"  {name:14s} {r['hit_rate_pct']:7.2f}% {r['holdout']['hit_rate_pct']:7.2f}% {r['z_vs_fair']:+6.2f}"
              f" {r['p_luck_after_holm']:7.3f} {min(r['fake_lotteries_z']):+6.2f}..{max(r['fake_lotteries_z']):+5.2f}"
              f"  {r['verdict']}")
    print(f"  model versions tried on this history: {tried}. Full detail: {RESULT}")


if __name__ == "__main__":
    main()
