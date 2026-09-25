"""Review a Mega 6/45 draw against the prediction frozen before it: tickets, models, number history, patterns, sales.

Usage: python3 lottery/review_draw.py                   # newest frozen prediction; fetches the result if needed
       python3 lottery/review_draw.py --draw 1567
       python3 lottery/review_draw.py --snapshot /tmp/t.json --no-scorecard   # test on an old draw
"""

import argparse
import datetime as dt
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import backtest_checkpoints as b
import collect_prizes as c
import power645_study as s
import predict_draw as pr
import verify_tickets as v

SCORECARD = pr.PRED_DIR / "scorecard.jsonl"
REASON = {
    "uniform (fair lottery)": "gives every number the same 13.33%",
    "hierarchical Beta-Binomial (partial pooling)": "asks the whole history how unequal the numbers are; so far the"
                                                    " answer is 'not at all', so it stays at 13.33%",
    "frequency / hot all-time (no pooling)": "favours numbers drawn most often since 2016",
    "hot last 30 draws": "favours numbers drawn most often in the last 30 draws",
    "exponentially weighted hot": "favours recently drawn numbers, fading older draws",
    "overdue / cold (gambler's fallacy)": "favours numbers that have not come up for the longest time",
    "Markov: follows last draw": "favours numbers that have tended to follow the previous draw's numbers",
    "logistic regression": "weighs recent and long-run frequency and gaps, fitted on the last 400 draws",
    "naive Bayes": "uses the same signals as logistic regression, combined with a naive Bayes rule",
}
PATTERN_LABEL = {"sum": "sum of the 6 numbers", "odd": "odd numbers", "low": "low numbers (1-22)",
                 "adjacent": "consecutive pairs (like 25-26)", "bands": "bands of five used (1-5, 6-10, ...)",
                 "spread": "spread (largest minus smallest)", "repeats": "numbers repeated from the previous draw"}


def find_snapshot(args):
    if args.snapshot:
        return args.snapshot
    if args.draw:
        return pr.PRED_DIR / f"draw_{args.draw:05d}.json"
    snaps = sorted(pr.PRED_DIR.glob("draw_?????.json"))
    if not snaps:
        raise SystemExit("no frozen prediction found; run predict_draw.py before the draw")
    return snaps[-1]


def ensure_result(target):
    rows = c.load()
    if target in rows:
        return True
    latest = c.latest_id()
    if latest < target:
        return False
    for i in range(max(rows) + 1, latest + 1):
        rows[i] = c.fetch(i)
    c.save(rows)
    return True


def mean_rank(p, drawn):
    r = [round(x, 6) for x in p]
    ranks = [1 + sum(x > r[i - 1] for x in r) + (sum(x == r[i - 1] for x in r) - 1) / 2 for i in drawn]
    return sum(ranks) / len(ranks)


def score(p, top6, drawn):
    return {"gain": s.log_score(p, drawn) - s.log_score([s.P] * s.N_BALLS, drawn),
            "hits": len(set(top6) & drawn), "rank": mean_rank(p, drawn), "mass": sum(p[i - 1] for i in drawn)}


def model_history(rows, seed, runs=10):
    hist = s.History([d["result"] for d in rows])
    n = len(rows)
    first_test = n - runs * ((n - 266) // runs)
    block = (n - first_test) // runs
    state = {"ewma": [0.0] * (s.N_BALLS + 1), "trans": [[0] * (s.N_BALLS + 1) for _ in range(s.N_BALLS + 1)],
             "nprev": [0] * (s.N_BALLS + 1), "kappa": 1e6}
    for t in range(first_test):
        s.update_state(state, hist, t)
    learned = [s.LogisticModel(), s.NaiveBayesModel()]
    rng = random.Random(seed)
    rec = defaultdict(lambda: defaultdict(list))
    for r in range(runs):
        lo, hi = first_test + r * block, first_test + (r + 1) * block
        state["kappa"] = s.eb_kappa([hist.cum[lo][i] for i in range(1, s.N_BALLS + 1)], lo)
        for t in range(lo, hi):
            if (t - lo) % 26 == 0:
                for m in learned:
                    m.fit(hist, t)
            qs = s.simple_models(hist, t, state)
            for m in learned:
                qs[m.name] = m.predict(hist, t)
            drawn = set(rows[t]["result"])
            for name, qraw in qs.items():
                qn = s.normalise(qraw)
                pick = rng.sample(range(1, s.N_BALLS + 1), s.K) if name.startswith("uniform") else s.top6(qn, rng)
                for key, val in score(qn, pick, drawn).items():
                    rec[name][key].append(val)
            s.update_state(state, hist, t)
    return rec, (rows[first_test]["id"], rows[first_test + runs * block - 1]["id"])


def band_pmf():
    base = [0] + [math.comb(5, k) for k in range(1, 6)]
    poly, out = [1], {}
    for j in range(1, 10):
        poly = [sum(poly[a] * base[k - a] for a in range(len(poly)) if 0 <= k - a < len(base))
                for k in range(len(poly) + 5)]
        if len(poly) > s.K and poly[s.K]:
            out[j] = math.comb(9, j) * poly[s.K] / s.C_TOTAL
    return out


def exact_pmfs():
    comb, total = math.comb, s.C_TOTAL
    pmfs = {"sum": s.sum_distribution(),
            "odd": {k: comb(23, k) * comb(22, s.K - k) / total for k in range(s.K + 1)},
            "low": {k: comb(22, k) * comb(23, s.K - k) / total for k in range(s.K + 1)},
            "adjacent": {m: comb(5, m) * comb(40, s.K - m) / total for m in range(s.K)},
            "bands": band_pmf(),
            "spread": {r: (s.N_BALLS - r) * comb(r - 1, 4) / total for r in range(5, s.N_BALLS)},
            "repeats": {k: s.P_MATCH[k] for k in range(s.K + 1)}}
    for key, pmf in pmfs.items():
        assert abs(sum(pmf.values()) - 1) < 1e-9, key
    return pmfs


def pattern_values(draw, prev):
    d = sorted(draw)
    return {"sum": sum(d), "odd": sum(x % 2 for x in d), "low": sum(x <= 22 for x in d),
            "adjacent": s.adjacent_pairs(d), "bands": len({(x - 1) // 5 for x in d}), "spread": d[-1] - d[0],
            "repeats": len(set(d) & set(prev))}


def hypergeom(m, k):
    return math.comb(m, k) * math.comb(s.N_BALLS - m, s.K - k) / s.C_TOTAL


def two_sided(pmf, value):
    lo = sum(p for k, p in pmf.items() if k <= value)
    hi = sum(p for k, p in pmf.items() if k >= value)
    return min(1.0, 2 * min(lo, hi)), lo


def fmt_nums(nums):
    return " ".join(f"{x:02d}" for x in sorted(nums))


def pct(a, b):
    return (a / b - 1) * 100


def popularity_bundle(snap):
    pop = snap["popularity"]
    return {"beta": pop["beta"], "cal5": pop["cal5"], "cal4": pop["cal4"], "m3": pop["m3"],
            "pattern": {int(k): tuple(val) for k, val in pop["pattern"].items()},
            "level": {int(k): val for k, val in pop["level"].items()}}


def review(snap, draws, seed):
    target = snap["target_draw"]
    before = [d for d in draws if d["id"] < target]
    actual = next(d for d in draws if d["id"] == target)
    drawn = set(actual["result"])
    out, lines = {"draw": target, "date": actual["date"], "result": actual["result"]}, []
    add = lines.append

    created = dt.datetime.fromisoformat(snap["created_at"])
    close = dt.datetime.fromisoformat(snap["sales_close"])
    pre_registered = created < close and not snap["result_known_when_created"]
    hist_ok = pr.history_hash(before) == snap["history_sha256"]
    models_now, _ = pr.model_probabilities(s.History([d["result"] for d in before]), snap["seed"])
    reproduce = all(models_now[n]["p"] == snap["models"][n]["p"] and models_now[n]["top6"] == snap["models"][n]["top6"]
                    for n in snap["models"])
    out["integrity"] = {"pre_registered": pre_registered, "history_unchanged": hist_ok, "predictions_reproduce": reproduce,
                        "code_unchanged": pr.code_hash() == snap["code_sha256"]}
    lead = (created - close).total_seconds() / 3600
    add(f"DRAW #{target} ({actual['date']}): {fmt_nums(drawn)}   jackpot {actual['jackpot_value']/1e9:.1f} ty VND,"
        f" won by {actual['jackpot_winners']}")
    add(f"  prediction frozen {snap['created_at']}"
        + (f", {-lead:.1f} h before sales closed: a genuine pre-draw prediction" if pre_registered
           else " AFTER the result was known: test only, not a real prediction"))
    add(f"  integrity: past data {'unchanged' if hist_ok else 'CHANGED since the prediction'};"
        f" predictions {'reproduce exactly from that data' if reproduce else 'DO NOT reproduce'}"
        f"{'' if out['integrity']['code_unchanged'] else ' (analysis code edited since; predictions still checked)'}")

    sales_all = s.reconstruct_sales(draws)
    row = next(r for r in sales_all if r["id"] == target)
    if snap["tickets"]:
        add("\n1. YOUR TICKETS")
        out["tickets"] = {}
        for label, tickets in snap["tickets"].items():
            won, results = 0, []
            for t in tickets:
                m = len(drawn & set(t))
                prize = row["pool"] / max(1, actual["jackpot_winners"]) if m == 6 else s.FIXED_PRIZE.get(m, 0)
                won += prize
                results.append((t, m, prize))
            exact = v.summarise(v.enumerate_all(tickets), len(tickets))
            past = v.replay(tickets, [d["result"] for d in before])
            spent = len(tickets) * s.TICKET
            add(f"  {label}: won {won:,.0f} VND for {spent:,} VND; best ticket {max(m for _, m, _ in results)} correct")
            for t, m, prize in results:
                marks = " ".join(f"[{x:02d}]" if x in drawn else f" {x:02d} " for x in t)
                add(f"      {marks}   {m} correct" + (f"  -> {prize:,.0f} VND" if prize else "")
                    + (" before 10% tax on the part above 10 million" if m == 6 else ""))
            add(f"      chance this set wins something in any draw: {float(exact['any'])*100:.2f}%;"
                f" the same set would have won something in {past['any']} of {len(before):,} past draws"
                f" ({past['any']/len(before)*100:.1f}%)")
            out["tickets"][label] = {"won": won, "spent": spent, "best": max(m for _, m, _ in results),
                                     "matches": [m for _, m, _ in results], "p_any": float(exact["any"])}

    add("\n2. EACH MODEL'S CHANCES vs WHAT HAPPENED (why it was right or wrong)")
    rec, span = model_history(before, seed)
    add(f"  track record = the same model predicting each of draws #{span[0]}..#{span[1]} from earlier draws only")
    out["models"] = {}
    for name, m in snap["models"].items():
        sc = score(m["p"], m["top6"], drawn)
        gains = rec[name]["gain"]
        flat = max(m["p"]) - min(m["p"]) < 1e-6
        percentile = sum(g <= sc["gain"] + 1e-12 for g in gains) / len(gains)
        better_share = sum(g > 1e-12 for g in gains) / len(gains)
        hit_mean = statistics.fmean(rec[name]["hits"])
        if flat:
            verdict = "no preference, so nothing to be right or wrong about"
        else:
            side = "better" if sc["gain"] > 0 else "worse"
            usual = ("typical for it" if 0.05 <= percentile <= 0.95
                     else "unusually good for it" if percentile > 0.95 else "unusually bad for it")
            verdict = f"{side} than fair this draw ({usual}: this score beat {percentile*100:.0f}% of its past draws)"
        add(f"  {name}: {verdict}")
        add(f"      it {REASON[name]}.")
        add(f"      its top 6 {fmt_nums(m['top6'])} caught {sc['hits']} (fair expectation 0.8);"
            f" drawn numbers sat at average position {sc['rank']:.1f} of 45 in its list (fair 23.0);"
            f" it gave them {sc['mass']:.3f} in total (fair 0.800)")
        add(f"      record: better than fair in {better_share*100:.0f}% of {len(gains):,} past draws,"
            f" average {statistics.fmean(gains)*1000:+.1f} milli-nats per draw, top-6 hits {hit_mean:.3f} per draw")
        out["models"][name] = {**sc, "flat": flat, "percentile": percentile, "record_better_share": better_share,
                               "record_mean_gain": statistics.fmean(gains), "record_hits": hit_mean}

    winners = [n for n, r in out["models"].items() if r["record_mean_gain"] > 1e-9]
    add("  why: every draw is a fresh random pick of 6 balls from 45, and the machine keeps no memory. A model 'wins'"
        " a draw when the balls happen to land where it leaned, and 'loses' when they don't. The score punishes a model"
        " hardest when it gave a drawn number a very low chance.")
    add(f"  over the whole track record {len(winners)} of {len(out['models'])} models beat fair on average"
        + (f": {', '.join(winners)}" if winners else "; leaning on any pattern has cost more than it gained"))

    add("\n3. THE 6 DRAWN NUMBERS: history and rates before this draw")
    ctx = {x["number"]: x for x in snap["numbers"]}
    short = [pr.SHORT[n] for n in snap["models"]]
    add(f"  {'no':>2s} {'times':>5s} {'rate':>6s} {'last30':>6s} {'waited':>6s} {'co-pick':>7s} {'tags':14s}"
        + "".join(f"{x:>9s}" for x in short) + "   <- each model's chance and its rank of 45")
    for i in sorted(drawn):
        x = ctx[i]
        cells = []
        for name in snap["models"]:
            p = snap["models"][name]["p"]
            cells.append(f"{p[i - 1]*100:5.1f}%{mean_rank(p, [i]):>3.0f}")
        add(f"  {i:02d} {x['count']:5d} {x['rate']*100:5.2f}% {x['last30']:6d} {x['since_seen']:6d}"
            f" {x['co_pickers']:6.2f}x {','.join(x['tags']):14s}" + "".join(f"{cell:>9s}" for cell in cells))
    add(f"  fair: rate 13.33%, last30 4.0, waited 7.5 draws on average")
    tag_sets = {tag: {i for i, x in ctx.items() if tag in x["tags"]} for tag in ("hot", "cold", "overdue")}
    out["tags"] = {}
    for tag, members in tag_sets.items():
        k = len(drawn & members)
        m = len(members)
        p_two, _ = two_sided({j: hypergeom(m, j) for j in range(s.K + 1)}, k)
        add(f"  {tag}: {k} of the 6 drawn {'was' if k == 1 else 'were'} {tag} (expected {s.K * m / s.N_BALLS:.1f}"
            f" from {m} {tag} numbers;"
            f" chance of a count this far from expected {p_two*100:.0f}%)")
        out["tags"][tag] = {"drawn": k, "pool": m, "expected": s.K * m / s.N_BALLS}

    add("\n4. PATTERN vs THE PAST")
    pmfs = exact_pmfs()
    now = pattern_values(actual["result"], before[-1]["result"])
    past = [pattern_values(before[t]["result"], before[t - 1]["result"]) for t in range(1, len(before))]
    add(f"  {'pattern':40s} {'this draw':>9s} {'chance of it':>12s} {'past draws same':>15s} {'fair average':>12s}"
        f" {'how unusual':>12s}")
    out["patterns"] = {}
    for key, value in now.items():
        pmf = pmfs[key]
        mean = sum(k * p for k, p in pmf.items())
        p_two, p_le = two_sided(pmf, value)
        same = sum(pv[key] == value for pv in past) / len(past)
        exact_same = pmf.get(value, 0.0)
        add(f"  {PATTERN_LABEL[key]:40s} {value:9d} {exact_same*100:11.1f}% {same*100:14.1f}% {mean:12.1f}"
            f" {'normal' if p_two >= 0.05 else 'rare':>7s} {p_two:4.2f}")
        out["patterns"][key] = {"value": value, "exact_p": exact_same, "past_share": same, "mean": mean, "p_two": p_two}
    add("  how unusual = chance a fair draw lands at least this far from average (below 0.05 = rare)")
    overlaps = sorted(((len(drawn & set(d["result"])), d["id"], d["date"]) for d in before), reverse=True)
    kmax = overlaps[0][0]
    closest = [(i, d) for k, i, d in overlaps if k == kmax]
    p_max = 1 - (1 - sum(s.P_MATCH[j] for j in range(kmax, s.K + 1))) ** len(before)
    add(f"  closest past draws share {kmax} numbers: " + ", ".join(f"#{i} ({d})" for i, d in closest[:6])
        + (f" and {len(closest) - 6} more" if len(closest) > 6 else ""))
    add(f"      chance that some past draw shares {kmax}+ numbers with a random draw: {p_max*100:.1f}%")
    pairs = Counter(p for d in before for p in combinations(d["result"], 2))
    trips = Counter(t for d in before for t in combinations(d["result"], 3))
    e2 = len(before) * math.comb(s.K, 2) / math.comb(s.N_BALLS, 2)
    e3 = len(before) * math.comb(s.K, 3) / math.comb(s.N_BALLS, 3)
    top_pairs = sorted(((pairs[p], p) for p in combinations(sorted(drawn), 2)), reverse=True)
    seen_trips = sorted(((trips[t], t) for t in combinations(sorted(drawn), 3) if trips[t]), reverse=True)
    add(f"  pairs inside this draw: most seen before {top_pairs[0][1]} x{top_pairs[0][0]}, least"
        f" {top_pairs[-1][1]} x{top_pairs[-1][0]} (fair average {e2:.1f} each)")
    add(f"  triples inside this draw seen before: {len(seen_trips)} of 20 (fair average {20 * (1 - math.exp(-e3)):.1f})"
        + (f"; most seen {seen_trips[0][1]} x{seen_trips[0][0]}" if seen_trips else ""))
    out["closest"] = {"overlap": kmax, "draws": [i for i, _ in closest], "p_max": p_max}

    add("\n5. DEVIATION SUMMARY: how far each prediction was from the result")
    add(f"  {'model':46s} {'score vs fair':>13s} {'avg position':>12s} {'chance given':>12s} {'top-6 hits':>10s}"
        f" {'percentile':>10s}")
    for name, sc in out["models"].items():
        where = "n/a" if sc["flat"] else f"{sc['percentile']*100:.0f}%"
        add(f"  {name:46s} {sc['gain']*1000:+10.1f} mn {sc['rank']:12.1f} {sc['mass']:12.3f} {sc['hits']:10d}"
            f" {where:>10s}")
    add("  fair lottery: score 0, position 23.0, chance 0.800, hits 0.8 on average."
        " mn = milli-nats of log-score: above 0 beat fair, below 0 did worse. percentile = share of the model's"
        " past draws that scored lower")

    add("\n6. SALES, JACKPOT AND WINNERS: forecast vs actual")
    fc = snap["forecast"]
    bundle = popularity_bundle(snap)
    post = b.conditional_winners(bundle, actual["result"], row["tickets"])
    tier_field = {3: "third_winners", 4: "second_winners", 5: "first_winners"}
    add(f"  tickets sold: forecast {fc['tickets']:,.0f}, actual {row['tickets']:,.0f}"
        f" ({pct(fc['tickets'], row['tickets']):+.1f}%; past median error about 3.4%)")
    add(f"  jackpot at the draw: forecast {fc['jackpot']/1e9:.2f} ty, actual {row['pool']/1e9:.2f} ty"
        f" ({pct(fc['jackpot'], row['pool']):+.2f}%)")
    add(f"  jackpot won: forecast chance {fc['p_jackpot_won']*100:.1f}%, actual"
        f" {'yes, by ' + str(actual['jackpot_winners']) if actual['jackpot_winners'] else 'no'}")
    add(f"  drawn numbers' popularity: players' co-pick factor for this combination"
        f" x{b.co_pickers(bundle, actual['result']):.2f} (1.00 = average)")
    out["forecast"] = {"tickets": (fc["tickets"], row["tickets"]), "jackpot": (fc["jackpot"], row["pool"]),
                       "p_jackpot_won": fc["p_jackpot_won"], "jackpot_won": actual["jackpot_winners"] > 0,
                       "winners": {}}
    for k in (5, 4, 3):
        pre = fc["winners"][str(k)]
        act = actual[tier_field[k]]
        add(f"  {k} correct: forecast before the draw {pre:,.0f}; knowing the numbers {post[k]:,.0f}; actual {act:,}"
            f" ({pct(post[k], max(act, 0.5)):+.0f}% after, {pct(pre, max(act, 0.5)):+.0f}% before)")
        out["forecast"]["winners"][k] = {"pre": pre, "post": post[k], "actual": act}
    return out, lines


def update_scorecard(entry):
    rows = []
    if SCORECARD.exists():
        rows = [json.loads(x) for x in SCORECARD.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["draw"] != entry["draw"]] + [entry]
    rows.sort(key=lambda r: r["draw"])
    SCORECARD.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def scorecard_lines(rows):
    lines = [f"\n7. RUNNING SCORECARD: {len(rows)} genuine pre-draw prediction(s) so far"]
    for name in rows[-1]["models"]:
        gains = [r["models"][name]["gain"] for r in rows]
        hits = [r["models"][name]["hits"] for r in rows]
        lines.append(f"  {name:46s} total score vs fair {sum(gains)*1000:+8.1f} mn; top-6 hits {sum(hits)}"
                     f" in {len(rows)} draw(s) (fair expectation {0.8*len(rows):.1f})")
    labels = sorted({label for r in rows for label in r.get("tickets", {})})
    for label in labels:
        won = sum(r["tickets"][label]["won"] for r in rows if label in r.get("tickets", {}))
        spent = sum(r["tickets"][label]["spent"] for r in rows if label in r.get("tickets", {}))
        lines.append(f"  {label}: won {won:,.0f} VND for {spent:,} VND spent")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-scorecard", action="store_true")
    args = ap.parse_args()
    path = find_snapshot(args)
    snap = json.loads(path.read_text())
    target = snap["target_draw"]
    if not args.no_fetch and not ensure_result(target):
        raise SystemExit(f"draw #{target} is not published yet (draws are at 18:00; try again after 18:30)")
    draws, gaps, conflicts = s.load_draws()
    if not any(d["id"] == target for d in draws):
        raise SystemExit(f"draw #{target} is not in the data yet")
    out, lines = review(snap, draws, snap["seed"])
    if out["integrity"]["pre_registered"] and not args.no_scorecard:
        lines += scorecard_lines(update_scorecard(out))
    text = "\n".join(lines) + "\n"
    stem = path.with_suffix("")
    Path(f"{stem}_review.txt").write_text(text)
    Path(f"{stem}_review.json").write_text(json.dumps(out, indent=1, default=str))
    print(text)
    print(f"saved {stem}_review.txt and {stem}_review.json")


if __name__ == "__main__":
    main()
