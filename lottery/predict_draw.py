"""Freeze every model's chance for all 45 numbers before a Mega 6/45 draw, so the result can be reviewed honestly.

Usage: python3 lottery/predict_draw.py                                   # next draw -> predictions/
       python3 lottery/predict_draw.py --target 1566 --out /tmp/t.json   # as if before an old draw (testing)
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path

import backtest_checkpoints as b
import power645_study as s

PRED_DIR = s.ROOT / "predictions"
DRAW_WEEKDAYS = (2, 4, 6)
HOT_MIN_LAST30, COLD_MAX_LAST30, OVERDUE_MIN_GAP = 6, 2, 15
SHORT = {"uniform (fair lottery)": "fair", "hierarchical Beta-Binomial (partial pooling)": "hier",
         "frequency / hot all-time (no pooling)": "freq", "hot last 30 draws": "hot30",
         "exponentially weighted hot": "ewma", "overdue / cold (gambler's fallacy)": "overdue",
         "Markov: follows last draw": "markov", "logistic regression": "logit", "naive Bayes": "nbayes"}
TICKET_SETS = ((10, "spread + unpopular (recommended)", "10 recommended tickets"),
               (5, "spread + unpopular (recommended)", "5 recommended tickets"),
               (7, "Bao 7 on least-picked numbers", "Bao 7 on the least-picked numbers"))


def history_hash(rows):
    keep = ("id", "date", "result", "jackpot_value", "jackpot_winners", "first_winners", "second_winners",
            "third_winners")
    blob = json.dumps([{k: r.get(k) for k in keep} for r in rows], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def code_hash():
    h = hashlib.sha256()
    for name in ("power645_study.py", "backtest_checkpoints.py", "predict_draw.py"):
        h.update((s.CODE_DIR / name).read_bytes())
    return h.hexdigest()


def next_draw_date(last_date):
    d = dt.date.fromisoformat(last_date) + dt.timedelta(days=1)
    while d.weekday() not in DRAW_WEEKDAYS:
        d += dt.timedelta(days=1)
    return d.isoformat()


def model_probabilities(hist, seed):
    n = hist.n
    state = {"ewma": [0.0] * (s.N_BALLS + 1), "trans": [[0] * (s.N_BALLS + 1) for _ in range(s.N_BALLS + 1)],
             "nprev": [0] * (s.N_BALLS + 1)}
    for t in range(n):
        s.update_state(state, hist, t)
    state["kappa"] = s.eb_kappa([hist.cum[n][i] for i in range(1, s.N_BALLS + 1)], n)
    qs = s.simple_models(hist, n, state)
    for m in (s.LogisticModel(), s.NaiveBayesModel()):
        m.fit(hist, n)
        qs[m.name] = m.predict(hist, n)
    rng = random.Random(seed)
    out = {}
    for name, q in qs.items():
        qn = s.normalise(q)
        if name.startswith("uniform"):
            top = sorted(rng.sample(range(1, s.N_BALLS + 1), s.K))
        else:
            top = sorted(s.top6(qn, rng))
        out[name] = {"p": [round(v, 7) for v in qn], "top6": top}
    return out, state["kappa"]


def number_context(hist, bundle):
    n = hist.n
    c5 = bundle["cal5"]["c"]
    out = []
    for i in range(1, s.N_BALLS + 1):
        last = hist.last[n][i]
        last30 = hist.cum[n][i] - hist.cum[max(0, n - 30)][i]
        since = n - last if last >= 0 else n + 1
        tags = [tag for tag, hit in (("hot", last30 >= HOT_MIN_LAST30), ("cold", last30 <= COLD_MAX_LAST30),
                                     ("overdue", since >= OVERDUE_MIN_GAP)) if hit]
        out.append({"number": i, "count": hist.cum[n][i], "rate": hist.cum[n][i] / n, "last30": last30,
                    "since_seen": since, "co_pickers": math.exp(c5 * bundle["beta"][i - 1]), "tags": tags})
    return out


def ticket_sets(target):
    out = {}
    for k, design, label in TICKET_SETS:
        path = s.RESULTS / f"tickets_{k}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["draw"] == target and design in saved["designs"]:
                out[label] = saved["designs"][design]["tickets"]
    return out


def build(target, seed):
    draws, gaps, conflicts = s.load_draws()
    if gaps or conflicts:
        raise ValueError(f"data has gaps {gaps[:5]} or conflicts {conflicts[:5]}")
    last = draws[-1]["id"]
    target = target or last + 1
    if not 2 <= target <= last + 1:
        raise ValueError(f"target must be between 2 and {last + 1}")
    rows = draws[: target - 1]
    known = next((d for d in draws if d["id"] == target), None)
    target_date = known["date"] if known else next_draw_date(rows[-1]["date"])
    hist = s.History([d["result"] for d in rows])
    models, kappa = model_probabilities(hist, seed)
    sales = [r for r in s.reconstruct_sales(draws) if r["id"] < target]
    bundle = b.popularity_bundle(sales)
    fc = s.forecast_sales(sales)
    share = s.jackpot_share_check(sales)
    coverage = share["winners"] / share["expected_winners"]
    return {
        "target_draw": target, "target_date": target_date,
        "draw_time": f"{target_date}T18:00:00+07:00", "sales_close": f"{target_date}T17:45:00+07:00",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "result_known_when_created": known is not None,
        "data_through": {"id": rows[-1]["id"], "date": rows[-1]["date"], "draws": len(rows)},
        "history_sha256": history_hash(rows), "code_sha256": code_hash(), "seed": seed,
        "hierarchical_kappa": kappa, "fair_p": s.P,
        "tag_rules": {"hot": f"drawn {HOT_MIN_LAST30}+ times in the last 30 draws (expected 4)",
                      "cold": f"drawn {COLD_MAX_LAST30} or fewer times in the last 30 draws",
                      "overdue": f"not drawn for {OVERDUE_MIN_GAP}+ draws (average gap 7.5)"},
        "models": models,
        "numbers": number_context(hist, bundle),
        "forecast": {**fc, "coverage": coverage,
                     "p_jackpot_won": 1 - math.exp(-fc["tickets"] * s.P_MATCH[6] * coverage),
                     "winners": {k: fc["tickets"] * s.P_MATCH[k] * bundle["level"][k] for k in (3, 4, 5)},
                     "ev_random_ticket": s.expected_value(fc["jackpot"], fc["tickets"])["ev"]},
        "popularity": {"beta": bundle["beta"], "cal5": bundle["cal5"], "cal4": bundle["cal4"],
                       "pattern": {str(k): list(v) for k, v in bundle["pattern"].items()},
                       "m3": bundle["m3"], "level": {str(k): v for k, v in bundle["level"].items()}},
        "tickets": ticket_sets(target),
    }


def render(snap, digest):
    fc = snap["forecast"]
    names = list(snap["models"])
    lines = [f"Mega 6/45 draw #{snap['target_draw']} on {snap['target_date']} at 18:00."
             f" Prediction frozen {snap['created_at']} from {snap['data_through']['draws']:,} draws"
             f" (through #{snap['data_through']['id']}, {snap['data_through']['date']}).",
             f"SHA-256 of the JSON file: {digest}",
             "" if not snap["result_known_when_created"] else "NOTE: made AFTER the result was known (test only).",
             "",
             f"Chance that each number is among the 6 drawn (fair = {snap['fair_p']*100:.2f}% for every number)",
             f"  {'no':>2s} {'drawn':>5s} {'rate':>6s} {'last30':>6s} {'since':>5s} {'co-pick':>7s} {'tags':14s}"
             + "".join(f"{SHORT[n]:>8s}" for n in names)]
    for ctx in snap["numbers"]:
        i = ctx["number"]
        lines.append(f"  {i:02d} {ctx['count']:5d} {ctx['rate']*100:5.2f}% {ctx['last30']:6d} {ctx['since_seen']:5d}"
                     f" {ctx['co_pickers']:6.2f}x {','.join(ctx['tags']):14s}"
                     + "".join(f"{snap['models'][n]['p'][i - 1]*100:7.2f}%" for n in names))
    lines += ["", "  tags: " + "; ".join(f"{k} = {v}" for k, v in snap["tag_rules"].items()),
              "  co-pick = how popular the number is with other players (1.00 = average; lower = fewer people to share with)",
              "", "Each model's top 6:"]
    lines += [f"  {n:46s} {' '.join(f'{x:02d}' for x in snap['models'][n]['top6'])}" for n in names]
    lines += ["", f"Forecast: {fc['tickets']:,.0f} tickets sold; jackpot at the draw {fc['jackpot']/1e9:.1f} ty VND"
              f" (advertised {fc['advertised']/1e9:.1f}); chance someone wins it {fc['p_jackpot_won']*100:.1f}%",
              "  expected winners: " + ", ".join(f"{k} correct {v:,.0f}" for k, v in sorted(
                  ((int(k), v) for k, v in fc["winners"].items()), reverse=True))]
    if snap["tickets"]:
        lines += ["", "Tickets to check after the draw:"]
        for label, tickets in snap["tickets"].items():
            lines.append(f"  {label}:")
            lines += [f"      {' '.join(f'{x:02d}' for x in t)}" for t in tickets]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        print(f"data: {update.refresh('645')}")
    snap = build(args.target, args.seed)
    path = args.out or PRED_DIR / f"draw_{snap['target_draw']:05d}.json"
    if args.out is None and snap["result_known_when_created"]:
        raise SystemExit(f"draw #{snap['target_draw']} already happened; use --out for a test snapshot")
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; a frozen prediction is never overwritten (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    text = render(snap, digest)
    path.with_suffix(".txt").write_text(text)
    print(text)
    print(f"saved {path} and {path.with_suffix('.txt').name}")


if __name__ == "__main__":
    main()
