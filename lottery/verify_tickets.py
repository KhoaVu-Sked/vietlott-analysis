"""Exact check of saved ticket sets: walk all C(45,6) draws, compare with the optimiser's simulation, replay on real draws.

Shares no code with the other analysis scripts, so it is an independent check of their numbers.

Usage: python3 lottery/verify_tickets.py 5 10            # results/tickets_5.json and tickets_10.json
       python3 lottery/verify_tickets.py path/to/tickets.json
"""

import json
import math
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIZE = [0, 0, 0, 30_000, 300_000, 10_000_000, 10_000_000]
TOTAL = math.comb(45, 6)


def enumerate_all(tickets):
    n = len(tickets)
    inc = [0] * 47
    for j, t in enumerate(tickets):
        for x in t:
            inc[x] += 1 << (4 * j)
    cnt = Counter()
    for a in range(1, 41):
        sa = inc[a]
        for b in range(a + 1, 42):
            sb = sa + inc[b]
            for c in range(b + 1, 43):
                sc = sb + inc[c]
                for d in range(c + 1, 44):
                    sd = sc + inc[d]
                    for e in range(d + 1, 45):
                        cnt.update(map((sd + inc[e]).__add__, inc[e + 1:46]))
    out = Counter()
    for key, w in cnt.items():
        hits = [(key >> (4 * j)) & 15 for j in range(n)]
        out[tuple(sorted(hits, reverse=True))] += w
    return out


def summarise(dist, n):
    total = sum(dist.values())
    assert total == TOTAL, total
    cost = n * 10_000
    r = Counter()
    for hits, w in dist.items():
        money = sum(PRIZE[h] for h in hits)
        best = hits[0]
        r["any"] += w * (best >= 3)
        r["four"] += w * (best >= 4)
        r["five"] += w * (best >= 5)
        r["six"] += w * (best == 6)
        r["profit"] += w * (money >= cost)
        r["two_winners"] += w * (sum(h >= 3 for h in hits) >= 2)
        r["money"] += w * money
    return {k: Fraction(v, total) for k, v in r.items()}


def replay(tickets, draws):
    sets = [set(t) for t in tickets]
    r = Counter()
    for d in draws:
        hits = [len(s & set(d)) for s in sets]
        r["any"] += max(hits) >= 3
        r["four"] += max(hits) >= 4
        r["money"] += sum(PRIZE[h] for h in hits)
    return r


def main():
    lines = (ROOT / "data" / "mega645.jsonl").read_text().splitlines()
    draws = [json.loads(line)["result"] for line in lines if line.strip()]
    single = Fraction(sum(math.comb(6, k) * math.comb(39, 6 - k) * PRIZE[k] for k in (3, 4, 5)), TOTAL)
    print(f"C(45,6) = {TOTAL:,}; exact fixed-prize value of one ticket = {float(single):,.4f} VND"
          f" (plus {float(Fraction(10_000_000, TOTAL)):.4f} if a jackpot hit is counted as 10M, as the simulation does)")
    for arg in sys.argv[1:]:
        path = Path(arg) if arg.endswith(".json") else ROOT / "results" / f"tickets_{arg}.json"
        saved = json.loads(path.read_text())
        sims = saved["sims"]
        n_tickets = len(next(iter(saved["designs"].values()))["tickets"])
        print(f"\n{n_tickets} tickets, seed {saved['seed']} (draw #{saved['draw']}): exact over all {TOTAL:,} draws"
              f" vs the {sims:,}-draw simulation; replay on the {len(draws):,} real past draws")
        print(f"  {'design':42s} {'win something':>22s} {'4+ correct':>21s} {'5+ correct':>21s} {'end up ahead':>21s}"
              f" {'fixed money':>18s} {'real draws: won / expected':>28s}")
        for name, d in saved["designs"].items():
            t0 = time.time()
            ex = summarise(enumerate_all(d["tickets"]), len(d["tickets"]))
            cells = []
            worst = 0.0
            for key in ("any", "four", "five", "profit"):
                p = float(ex[key])
                z = (d[key] - p) / math.sqrt(p * (1 - p) / sims)
                worst = max(worst, abs(z))
                cells.append(f"{p*100:7.3f}% sim {d[key]*100:6.3f}% z{z:+5.1f}")
            rp = replay(d["tickets"], draws)
            exp_any = float(ex["any"]) * len(draws)
            sd_any = math.sqrt(exp_any * (1 - float(ex["any"])))
            print(f"  {name:42s} " + " ".join(cells) + f" {float(ex['money']):9,.1f} sim {d['money']:8,.1f}"
                  f"   {rp['any']:4d} / {exp_any:6.1f} (sd {sd_any:4.1f})   [{time.time()-t0:.0f}s]")
            per_ticket = ex["money"] / (len(d["tickets"]) * (single + Fraction(10_000_000, TOTAL)))
            print(f"  {'':42s} exact: jackpot 1 in {1/float(ex['six']):,.0f}; two or more winning tickets"
                  f" {float(ex['two_winners'])*100:.3f}%; fixed money / ({len(d['tickets'])} x one ticket)"
                  f" = {float(per_ticket):.6f}; real past draws paid {rp['money']:,} VND for"
                  f" {len(draws)*len(d['tickets'])*10_000:,} spent; worst |z| {worst:.1f}")


if __name__ == "__main__":
    main()
