"""Review a Power 6/55 draw against the prediction frozen before it: tickets (with Jackpot 2), models, patterns, sales.

Usage: python3 lottery/review_power655.py                  # newest frozen 6/55 prediction; fetches the result
       python3 lottery/review_power655.py --draw 1403
"""

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

import collect_power655 as cp
import power645_study as s
import power655_study as sp
import predict_draw as pr
import review_draw as rd


def find_snapshot(args):
    if args.draw:
        return sp.ROOT / "predictions" / f"power655_draw_{args.draw:05d}.json"
    snaps = sorted((sp.ROOT / "predictions").glob("power655_draw_?????.json"))
    if not snaps:
        raise SystemExit("no frozen 6/55 prediction found; run power655_study.py before the draw")
    return snaps[-1]


def ensure_result(target):
    rows = cp.c.load(cp.OUT)
    if target in rows:
        return True
    latest = cp.latest_id()
    if latest < target:
        return False
    for i in range(max(rows) + 1, latest + 1):
        rows[i] = cp.fetch(i)
    cp.c.save(rows, cp.OUT)
    return True


def tier(main_hits, has_bonus):
    if main_hits == 6:
        return "Jackpot 1"
    if main_hits == 5 and has_bonus:
        return "Jackpot 2"
    return {5: "First", 4: "Second", 3: "Third"}.get(main_hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, default=None)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    sp.use_655_constants()
    path = find_snapshot(args)
    snap = json.loads(path.read_text())
    target = snap["target_draw"]
    if not args.no_fetch and not ensure_result(target):
        raise SystemExit(f"draw #{target} is not published yet (draws are at 18:00; try again after 18:30)")
    draws, *_ = sp.load_draws()
    if not any(d["id"] == target for d in draws):
        raise SystemExit(f"draw #{target} is not in the data yet")
    before = [d for d in draws if d["id"] < target]
    actual = next(d for d in draws if d["id"] == target)
    drawn, bonus = set(actual["result"]), actual["bonus"]
    lines = []
    add = lines.append
    created = dt.datetime.fromisoformat(snap["created_at"])
    close = dt.datetime.fromisoformat(snap["sales_close"])
    models_now, _ = pr.model_probabilities(s.History([d["result"] for d in before]), snap["seed"])
    reproduce = all(models_now[n]["p"] == snap["models"][n]["p"] for n in snap["models"])
    add(f"POWER 6/55 DRAW #{target} ({actual['date']}): {sp.fmt(drawn)} + bonus {bonus:02d}   Jackpot 1"
        f" {actual['j1_prize']/1e9:.1f} ty won by {actual['j1_winners']}, Jackpot 2 {actual['j2_prize']/1e9:.2f} ty"
        f" won by {actual['j2_winners']}")
    add(f"  prediction frozen {snap['created_at']}"
        + (f", {(close - created).total_seconds()/3600:.1f} h before sales closed" if created < close
           else " AFTER sales closed: test only")
        + f"; predictions {'reproduce exactly' if reproduce else 'DO NOT reproduce'} from the earlier data")

    add("\n1. YOUR TICKETS")
    for label, tickets in snap["tickets"].items():
        won = 0.0
        add(f"  {label}:")
        for t in tickets:
            h = len(drawn & set(t))
            name = tier(h, bonus in t)
            prize = {"Jackpot 1": actual["j1_prize"] / max(1, actual["j1_winners"]),
                     "Jackpot 2": actual["j2_prize"] / max(1, actual["j2_winners"])}.get(name, sp.FIXED.get(h, 0))
            won += prize if name else 0
            marks = " ".join(f"[{x:02d}]" if x in drawn else (f"<{x:02d}>" if x == bonus else f" {x:02d} ") for x in t)
            add(f"      {marks}   {h} correct" + (f"  -> {name} {prize:,.0f} VND" if name else ""))
        ex, _ = sp.exact_summary(tickets)
        add(f"      won {won:,.0f} VND for {len(tickets) * sp.TICKET:,} VND; chance this set wins something in any draw"
            f" {ex['any']*100:.2f}%   ([x] = main number drawn, <x> = the bonus)")

    add("\n2. EACH MODEL'S CHANCES vs WHAT HAPPENED")
    rec, span = rd.model_history(before, snap["seed"])
    add(f"  track record = the same model predicting draws #{span[0]}..#{span[1]} from earlier draws only;"
        f" fair: position 28.0 of 55, total chance {6*6/55:.3f}, top-6 hits {6*6/55:.2f}")
    for name, m in snap["models"].items():
        sc = rd.score(m["p"], m["top6"], drawn)
        gains = rec[name]["gain"]
        flat = max(m["p"]) - min(m["p"]) < 1e-6
        pct = sum(g <= sc["gain"] + 1e-12 for g in gains) / len(gains)
        verdict = ("no preference" if flat else
                   f"{'better' if sc['gain'] > 0 else 'worse'} than fair; this score beat {pct*100:.0f}% of its past draws")
        add(f"  {name:46s} {verdict}; top 6 caught {sc['hits']}, drawn numbers at position {sc['rank']:.1f},"
            f" chance given {sc['mass']:.3f}; record better than fair in"
            f" {sum(g > 1e-12 for g in gains)/len(gains)*100:.0f}% of {len(gains):,} draws")

    add("\n3. PATTERN vs THE PAST")
    prev = before[-1]["result"]
    d = sorted(drawn)
    values = {"sum": sum(d), "odd numbers": sum(x % 2 for x in d), "consecutive pairs": s.adjacent_pairs(d),
              "repeats from the previous draw": len(drawn & set(prev))}
    pmfs = {"sum": s.sum_distribution(),
            "odd numbers": {k: math.comb(28, k) * math.comb(27, 6 - k) / sp.C_TOTAL for k in range(7)},
            "consecutive pairs": {m: math.comb(5, m) * math.comb(50, 6 - m) / sp.C_TOTAL for m in range(6)},
            "repeats from the previous draw": {k: sp.P_MATCH[k] for k in range(7)}}
    for key, value in values.items():
        p_two, _ = rd.two_sided(pmfs[key], value)
        mean = sum(k * p for k, p in pmfs[key].items())
        add(f"  {key:32s} {value:5d}   fair average {mean:6.1f}   {'normal' if p_two >= 0.05 else 'rare'} ({p_two:.2f})")

    add("\n4. SALES AND JACKPOTS: forecast vs actual")
    sales = sp.reconstruct_sales(draws)
    row = next(r for r in sales if r["id"] == target)
    fc = snap["forecast"]
    add(f"  tickets sold: forecast {fc['tickets']:,.0f}, actual {row['tickets']:,.0f}"
        f" ({(fc['tickets']/row['tickets']-1)*100:+.1f}%)")
    add(f"  Jackpot 1 at the draw: forecast {fc['j1']/1e9:.2f} ty, actual {row['pool1']/1e9:.2f} ty;"
        f" Jackpot 2: forecast {fc['j2']/1e9:.2f} ty, actual {row['pool2']/1e9:.2f} ty")
    add(f"  forecast chance Jackpot 1 won {fc['p_won1']*100:.1f}% (actual {'yes' if actual['j1_winners'] else 'no'}),"
        f" Jackpot 2 {fc['p_won2']*100:.1f}% (actual {'yes' if actual['j2_winners'] else 'no'})")
    text = "\n".join(lines) + "\n"
    out = path.with_name(path.stem + "_review.txt")
    out.write_text(text)
    print(text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
