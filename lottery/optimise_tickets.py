"""Choose k Mega 6/45 tickets for the next draw: spread coverage for small prizes, unpopular numbers for the jackpot share.

Usage: python3 lottery/optimise_tickets.py [--tickets 10] [--sims 1000000] [--seed 2026]
"""

import argparse
import json
import math
import random
from collections import Counter
from itertools import combinations

import backtest_checkpoints as b
import power645_study as s

HOT_WINDOW = 30


def hot_coefficient(sales, bundle):
    rows = []
    for i in range(HOT_WINDOW, len(sales)):
        recent = [0] * (s.N_BALLS + 1)
        for d in sales[i - HOT_WINDOW:i]:
            for x in d["result"]:
                recent[x] += 1
        h = sum(recent[x] for x in sales[i]["result"]) - s.K * HOT_WINDOW * s.P
        base = sales[i]["tickets"] * s.P_MATCH[5] * b.co_pickers(bundle, sales[i]["result"])
        rows.append((base, h, sales[i]["first_winners"]))
    g = 0.0
    for _ in range(30):
        grad = sum((y - e * math.exp(g * h)) * h for e, h, y in rows)
        hess = sum(e * math.exp(g * h) * h * h for e, h, y in rows)
        step = grad / hess
        g += step
        if abs(step) < 1e-12:
            break
    return g


def longest_run(t):
    best = run = 1
    for a, c in zip(t, t[1:]):
        run = run + 1 if c == a + 1 else 1
        best = max(best, run)
    return best


def is_arithmetic(t):
    gaps = {c - a for a, c in zip(t, t[1:])}
    return len(gaps) == 1


class Scorer:
    def __init__(self, bundle, g, recent, jackpot, tickets):
        self.bundle, self.g, self.recent = bundle, g, recent
        self.jackpot, self.tickets = jackpot, tickets
        self.cache = {}

    def co_pickers(self, t):
        h = sum(self.recent[x] for x in t) - s.K * HOT_WINDOW * s.P
        return b.co_pickers(self.bundle, t) * math.exp(self.g * h)

    def ev(self, t):
        key = tuple(t)
        if key not in self.cache:
            self.cache[key] = s.expected_value(self.jackpot, self.tickets, self.co_pickers(t))
        return self.cache[key]


def valid(t):
    return len(set(t)) == s.K and longest_run(t) <= 2 and not is_arithmetic(t)


def optimise(scorer, k, max_overlap, rng, max_uses=None, iters=40_000):
    def ok(t, design, skip=None):
        if not valid(t) or t in design:
            return False
        others = [u for i, u in enumerate(design) if i != skip]
        if any(len(set(t) & set(u)) > max_overlap for u in others):
            return False
        if max_uses is not None:
            uses = Counter(x for u in others for x in u)
            if any(uses[x] >= max_uses for x in t):
                return False
        return True

    design = []
    while len(design) < k:
        best = None
        for _ in range(4000):
            t = sorted(rng.sample(range(1, s.N_BALLS + 1), s.K))
            if ok(t, design):
                v = scorer.ev(t)["ev"]
                if best is None or v > best[0]:
                    best = (v, t)
        if best is None:
            raise ValueError(f"could not place ticket {len(design) + 1} with overlap <= {max_overlap}")
        design.append(best[1])
    for _ in range(iters):
        j = rng.randrange(k)
        t = design[j][:]
        t[rng.randrange(s.K)] = rng.randint(1, s.N_BALLS)
        t = sorted(t)
        if ok(t, design, skip=j) and scorer.ev(t)["ev"] > scorer.ev(design[j])["ev"]:
            design[j] = t
    return sorted(design)


def simulate(designs, sims, rng):
    masks = {name: [sum(1 << x for x in t) for t in d] for name, d in designs.items()}
    stats = {name: {"any": 0, "four": 0, "five": 0, "profit": 0, "money": 0} for name in designs}
    cost = {name: len(d) * s.TICKET for name, d in designs.items()}
    prize = [0, 0, 0, s.FIXED_PRIZE[3], s.FIXED_PRIZE[4], s.FIXED_PRIZE[5], s.FIXED_PRIZE[5]]
    balls = list(range(1, s.N_BALLS + 1))
    for _ in range(sims):
        dm = 0
        for x in rng.sample(balls, s.K):
            dm |= 1 << x
        for name, ms in masks.items():
            best = 0
            money = 0
            for m in ms:
                h = bin(dm & m).count("1")
                money += prize[h]
                if h > best:
                    best = h
            st = stats[name]
            st["money"] += money
            if best >= 3:
                st["any"] += 1
                if best >= 4:
                    st["four"] += 1
                    if best >= 5:
                        st["five"] += 1
                if money >= cost[name]:
                    st["profit"] += 1
    return {name: {k: v / sims for k, v in st.items()} for name, st in stats.items()}


def bao(numbers):
    return [sorted(c) for c in combinations(sorted(numbers), s.K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", type=int, default=10)
    ap.add_argument("--sims", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    k = args.tickets
    rng = random.Random(args.seed)

    draws, _, _ = s.load_draws()
    sales = s.reconstruct_sales(draws)
    bundle = b.popularity_bundle(sales)
    g = hot_coefficient(sales, bundle)
    recent = [0] * (s.N_BALLS + 1)
    for d in sales[-HOT_WINDOW:]:
        for x in d["result"]:
            recent[x] += 1
    fc = s.forecast_sales(sales)
    scorer = Scorer(bundle, g, recent, fc["jackpot"], fc["tickets"])
    print(f"draw #{sales[-1]['id'] + 1}: forecast {fc['tickets']:,.0f} tickets, jackpot {fc['jackpot']/1e9:.1f} ty VND;"
          f" hot-number effect exp({g:+.4f} x excess appearances in last {HOT_WINDOW} draws)")

    designs = {
        "spread + unpopular (recommended)": optimise(scorer, k, 0 if k <= 6 else 1, rng, max_uses=2),
        "all unpopular, overlapping": optimise(scorer, k, s.K - 1, rng),
        "random quick picks": [sorted(rng.sample(range(1, s.N_BALLS + 1), s.K)) for _ in range(k)],
    }
    if k >= 7:
        lowest7 = sorted(sorted(range(1, s.N_BALLS + 1), key=lambda i: scorer.co_pickers([i]))[:7])
        designs["Bao 7 on least-picked numbers" + (f" + {k - 7} spread" if k > 7 else "")] = (
            bao(lowest7) + designs["spread + unpopular (recommended)"][: k - 7])

    sim = simulate(designs, args.sims, random.Random(args.seed + 1))
    p_jackpot = k / math.comb(s.N_BALLS, s.K)
    print(f"\n{k} tickets = {k * s.TICKET:,} VND. Jackpot chance for any {k} different tickets: 1 in {1/p_jackpot:,.0f}")
    print(f"  {'option':44s} {'win something':>13s} {'4+ correct':>10s} {'5+ correct':>10s} {'end up ahead':>12s}"
          f" {'expected value':>15s} {'jackpot share':>13s}")
    for name, d in designs.items():
        st = sim[name]
        ev = sum(scorer.ev(t)["ev"] for t in d)
        share = sum(scorer.ev(t)["share"] for t in d) / len(d)
        print(f"  {name:44s} {st['any']*100:12.1f}% {st['four']*100:9.2f}% {st['five']*100:9.3f}% {st['profit']*100:11.2f}%"
              f" {ev:14,.0f} {share*100:12.0f}%")
    saved = {"draw": sales[-1]["id"] + 1, "seed": args.seed, "sims": args.sims, "forecast": fc,
             "designs": {name: {"tickets": d, **sim[name],
                                "ev": sum(scorer.ev(t)["ev"] for t in d)} for name, d in designs.items()}}
    s.RESULTS.mkdir(exist_ok=True)
    (s.RESULTS / f"tickets_{k}.json").write_text(json.dumps(saved, indent=1))
    print("\n  tickets:")
    for name, d in designs.items():
        print(f"  {name}:")
        for t in d:
            e = scorer.ev(t)
            print(f"      {' '.join(f'{x:02d}' for x in t)}   co-pickers x{scorer.co_pickers(t):.2f}  EV {e['ev']:,.0f} VND")


if __name__ == "__main__":
    main()
