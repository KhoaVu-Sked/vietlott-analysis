"""Test every prediction model in models/ on every past draw, using only the draws before it each time,
and write results/result.json (rewritten on every run).

Usage: python3 lottery/backtest_models.py                  # fetch new draws, test every model on all three games
       python3 lottery/backtest_models.py --game 655       # one game only (6/55 uses the main numbers)
       python3 lottery/backtest_models.py --models hot30 --fakes 5 --no-update
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import aligner
import power645_study as s
import update

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULT = ROOT / "results" / "result.json"
RUNS_LOG = ROOT / "results" / "backtest_runs.jsonl"
ALIGN_DIR = ROOT / "results"
K = 6
GAMES = {"645": {"name": "Mega 6/45", "balls": 45, "k": 6, "prize": {3: 30_000, 4: 300_000, 5: 10_000_000}},
         "655": {"name": "Power 6/55", "balls": 55, "k": 6, "prize": {3: 50_000, 4: 500_000, 5: 40_000_000}},
         "535": {"name": "Lotto 5/35", "balls": 35, "k": 5, "prize": {3: 30_000, 4: 500_000, 5: 10_000_000}}}


def load_game(game):
    if game == "645":
        draws, gaps, _ = s.load_draws()
    elif game == "655":
        import power655_study as sp
        draws, gaps = sp.load_draws()
    else:
        rows = [json.loads(line) for line in (ROOT / "data" / "lotto535.jsonl").read_text().splitlines() if line.strip()]
        draws = sorted(rows, key=lambda r: r["id"])
        gaps = [i for i in range(1, draws[-1]["id"] + 1) if i not in {d["id"] for d in draws}]
    if gaps:
        raise SystemExit(f"missing draws {gaps[:10]}; run the collector first")
    return [d["result"] for d in draws], [d["id"] for d in draws]


_LOADED = {}


def load_module(path):
    path = Path(path)
    if path not in _LOADED:
        spec = importlib.util.spec_from_file_location(f"models_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LOADED[path] = mod
    return _LOADED[path]


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


def job(args):
    path, draws, n_balls, k, start, seed = args
    import model_tools
    model_tools.K = k
    return run_model(load_module(path), draws, n_balls, start, seed, k)


def hits_pmf(n_balls, k):
    total = math.comb(n_balls, k)
    return [math.comb(k, h) * math.comb(n_balls - k, k - h) / total for h in range(k + 1)]


def to_chances(scores, k):
    if min(scores) < 0 or sum(scores) <= 0:
        return None
    total = sum(scores)
    return [min(max(v * k / total, 1e-4), 0.999) for v in scores]


def log_gain(chances, drawn, n_balls, k):
    p = k / n_balls
    return sum(math.log(c / p) if i in drawn else math.log((1 - c) / (1 - p)) for i, c in enumerate(chances, 1))


def run_model(mod, draws, n_balls, start, seed, k=K):
    rng = random.Random(seed)
    hits, gains, tickets = [], [], []
    for t in range(start, len(draws)):
        scores = [float(v) for v in mod.predict(draws[:t], n_balls)]
        if len(scores) != n_balls:
            raise ValueError(f"predict() must return {n_balls} scores, got {len(scores)}")
        order = sorted(range(1, n_balls + 1), key=lambda i: (-scores[i - 1], rng.random()))
        drawn = set(draws[t])
        tickets.append(sorted(order[:k]))
        hits.append(len(set(order[:k]) & drawn))
        chances = to_chances(scores, k)
        gains.append(log_gain(chances, drawn, n_balls, k) if chances else None)
    scores = [float(v) for v in mod.predict(draws, n_balls)]
    nxt = sorted(sorted(range(1, n_balls + 1), key=lambda i: (-scores[i - 1], rng.random()))[:k])
    return hits, gains, tickets, nxt


def summarise(hits, gains, n_balls, prize, k=K):
    n = len(hits)
    pmf = hits_pmf(n_balls, k)
    mean_h = sum(h * p for h, p in enumerate(pmf))
    var_h = sum((h - mean_h) ** 2 * p for h, p in enumerate(pmf))
    z = (sum(hits) - n * mean_h) / math.sqrt(n * var_h)
    dist = Counter(hits)
    scored = [g for g in gains if g is not None]
    return {"draws": n, "hit_rate_pct": 100 * sum(hits) / (k * n), "fair_hit_rate_pct": 100 * mean_h / k,
            "z_vs_fair": z, "p_luck": s.norm_sf(z),
            "hits_pct": {str(h): 100 * dist[h] / n for h in range(k + 1)},
            "fair_hits_pct": {str(h): 100 * pmf[h] for h in range(k + 1)},
            "prize_draws_pct": 100 * sum(h >= 3 for h in hits) / n, "fair_prize_draws_pct": 100 * sum(pmf[3:]),
            "jackpots": dist[k], "fixed_prizes_won_vnd": sum(prize.get(h, 0) for h in hits), "spent_vnd": n * 10_000,
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


def prepare(code, fakes_n, seed):
    n_balls, k = GAMES[code]["balls"], GAMES[code]["k"]
    draws, ids = load_game(code)
    fakes = [[sorted(r.sample(range(1, n_balls + 1), k)) for _ in draws]
             for r in (random.Random(seed + 100 + f) for f in range(fakes_n))]
    return {"draws": draws, "ids": ids, "fakes": fakes}


def assemble(code, prep, models, runs, start, holdout, aligned=None, aligner_sha=None):
    game = GAMES[code]
    n_balls, prize, ids, k = game["balls"], game["prize"], prep["ids"], game["k"]
    split = len(prep["draws"]) - holdout - start
    current = {name: h for name, (_, h) in models.items()}
    if aligned:
        current["aligned_vote"] = aligner_sha
    tried = versions_tried(code, current)
    out = {"game": game["name"], "tested_draws": f"#{ids[start]}..#{ids[-1]}",
           "holdout_draws": f"#{ids[-holdout]}..#{ids[-1]}", "fair_hit_rate_pct": 100 * k / n_balls,
           "model_versions_tried_on_this_history": tried,
           "warning": (f"{tried} model versions have been scored on this same history. Expect about 1 in 20 to reach"
                       " p_luck < 0.05 by chance alone; only the hold-out and new live draws can confirm a real edge."),
           "models": {}}
    p_values = {}
    for name, (_, digest) in models.items():
        hits, gains = runs[(code, name, 0)][:2]
        res = summarise(hits, gains, n_balls, prize, k)
        res["holdout"] = summarise(hits[split:], gains[split:], n_balls, prize, k)
        res["fake_lotteries_z"] = [summarise(*runs[(code, name, h)][:2], n_balls, prize, k)["z_vs_fair"]
                                   for h in range(1, len(prep["fakes"]) + 1)]
        res["file_sha"] = digest
        out["models"][name] = res
        p_values[name] = res["p_luck"]
    if aligned:
        hits = aligned[0]["hits"]
        none = [None] * len(hits)
        res = summarise(hits, none, n_balls, prize, k)
        res["holdout"] = summarise(hits[split:], none[split:], n_balls, prize, k)
        res["fake_lotteries_z"] = [summarise(a["hits"], [None] * len(a["hits"]), n_balls, prize, k)["z_vs_fair"]
                                   for a in aligned[1:]]
        res["file_sha"] = aligner_sha
        res["next_ticket"] = aligned[0]["next"]
        res["weights_now"] = dict(sorted(aligned[0]["weights"].items(), key=lambda kv: -kv[1]))
        out["models"]["aligned_vote"] = res
        p_values["aligned_vote"] = res["p_luck"]
    adjusted = dict(zip(p_values, s.holm(list(p_values.values()))))
    for name, res in out["models"].items():
        res["p_luck_after_holm"] = adjusted[name]
        if adjusted[name] < 0.05 and res["holdout"]["p_luck"] < 0.05:
            res["verdict"] = "beats fair on all draws and on the hold-out: freeze it and test on new live draws"
        elif res["z_vs_fair"] > 0:
            res["verdict"] = "a little better than fair, but within what luck produces"
        else:
            res["verdict"] = "no better than a random ticket"
    return out


def run(games, names=None, start=50, holdout=300, fakes=3, seed=2026, fetch=True, status=lambda text: None):
    models = load_models(names)
    result = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "games": {}}
    notes = {}
    for code in games:
        if fetch:
            status(f"Checking vietlott.vn for new {GAMES[code]['name']} draws")
        notes[code] = update.refresh(code) if fetch else "not updated (--no-update)"
    prepared = {code: prepare(code, fakes, seed) for code in games}
    workers = os.cpu_count() or 2
    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for code, prep in prepared.items():
            for name, (path, _) in models.items():
                for h, history in enumerate([prep["draws"], *prep["fakes"]]):
                    futures[pool.submit(job, (str(path), history, GAMES[code]["balls"], GAMES[code]["k"], start,
                                              seed))] = (code, name, h)
        runs = {}
        for done, fut in enumerate(as_completed(futures), 1):
            runs[futures[fut]] = fut.result()
            status(f"Testing {len(models)} models on {len(games)} game(s): {done} of {len(futures)} runs done")
    aligner_sha = hashlib.sha256(Path(aligner.__file__).read_bytes()).hexdigest()[:12]
    for code in games:
        prep, n_balls, k = prepared[code], GAMES[code]["balls"], GAMES[code]["k"]
        aligned = [aligner.align({name: runs[(code, name, h)][2] for name in models}, history[start:], n_balls, k,
                                 next_tickets={name: runs[(code, name, h)][3] for name in models} if h == 0 else None,
                                 seed=seed)
                   for h, history in enumerate([prep["draws"], *prep["fakes"]])]
        out = assemble(code, prep, models, runs, start, holdout, aligned, aligner_sha)
        out["data"] = notes[code]
        ALIGN_DIR.mkdir(parents=True, exist_ok=True)
        with (ALIGN_DIR / f"aligned_{code}.jsonl").open("w") as fh:
            for j, step in enumerate(aligned[0]["steps"]):
                fh.write(json.dumps({"draw": prep["ids"][start + j], "ticket": step["ticket"], "hits": step["hits"],
                                     "leaders": [[n, round(w, 4)] for n, w in step["leaders"]]}) + "\n")
        if fetch:
            path, fresh = aligner.freeze(code, prep["ids"][-1] + 1, aligned[0]["next"], aligned[0]["weights"],
                                         prep["ids"][-1], name=GAMES[code]["name"])
            out["aligned_frozen"] = {"file": str(path), "new": fresh}
        result["games"][code] = out
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1))
    with RUNS_LOG.open("a") as fh:
        for code, out in result["games"].items():
            fh.write(json.dumps({"time": result["generated_at"], "game": code,
                                 "models": {n: r["file_sha"] for n, r in out["models"].items()}}) + "\n")
    return result


def ranked(out):
    return sorted(out["models"].items(), key=lambda kv: -kv[1]["z_vs_fair"])


def summary_lines(result):
    lines = []
    for out in result["games"].values():
        lines += [f"{out['game']}: tested {out['tested_draws']}, hold-out {out['holdout_draws']}; data {out['data']};"
                  f" a random ticket catches {out['fair_hit_rate_pct']:.2f}% of the numbers",
                  f"  {'model':24s} {'hit rate':>8s} {'hold-out':>8s} {'z':>6s} {'p luck':>7s} {'fake z range':>14s}"
                  f" {'log score':>10s}  verdict"]
        for name, r in ranked(out):
            score = "" if r["log_score_vs_fair"] is None else f"{r['log_score_vs_fair'] * 1000:+10.1f}"
            lines.append(f"  {name:24s} {r['hit_rate_pct']:7.2f}% {r['holdout']['hit_rate_pct']:7.2f}%"
                         f" {r['z_vs_fair']:+6.2f} {r['p_luck_after_holm']:7.3f}"
                         f" {min(r['fake_lotteries_z']):+6.2f}..{max(r['fake_lotteries_z']):+5.2f} {score:>10s}"
                         f"  {r['verdict']}")
        vote = out["models"].get("aligned_vote")
        if vote:
            frozen = out.get("aligned_frozen")
            where = "" if not frozen else (f"; frozen to {Path(frozen['file']).name}" if frozen["new"]
                                           else f"; already frozen in {Path(frozen['file']).name}")
            lines.append(f"  aligned vote for the next draw: {' '.join(f'{x:02d}' for x in vote['next_ticket'])}{where}")
        lines += [f"  model versions tried on this history: {out['model_versions_tried_on_this_history']};"
                  " log score = milli-nats per draw vs a random ticket, 0 = as good, below 0 = worse", ""]
    lines.append(f"Full detail: {RESULT}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["645", "655", "535", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--fakes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    games = ("645", "655", "535") if args.game == "all" else (args.game,)
    shown = set()

    def status(text):
        head = text.split(":")[0]
        if sys.stdout.isatty():
            print(f"\r\x1b[2K{text}...", end="" if "runs done" in text else "\n", flush=True)
        elif head not in shown:
            shown.add(head)
            print(head + "...", flush=True)

    result = run(games, args.models, args.start, args.holdout, args.fakes, args.seed, not args.no_update, status)
    if sys.stdout.isatty():
        print()
    print("\n".join(summary_lines(result)))


if __name__ == "__main__":
    main()
