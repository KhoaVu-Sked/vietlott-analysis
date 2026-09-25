"""Time-machine check: at past draws, rebuild every estimate from earlier draws only, then compare with what happened.

Usage: python3 lottery/backtest_checkpoints.py [--count 10] [--gap 1] [--end LAST_ID] [--seed 2026]
"""

import argparse
import json
import math
import random
import statistics

import power645_study as s

TIER_FIELD = {3: "third_winners", 4: "second_winners", 5: "first_winners"}


def popularity_bundle(rows):
    pop = s.popularity_model(rows)
    beta = pop["beta"]
    cal5 = s.popularity_calibration(rows, beta, 5)
    cal4 = s.popularity_calibration(rows, beta, 4)
    pattern = s.pattern_factors(rows, beta, cal5["c"])
    m3 = sum(r["third_winners"] for r in rows) / sum(
        r["tickets"] * s.P_MATCH[3] * math.exp(sum(beta[i - 1] for i in r["result"])) for r in rows)
    level = {k: sum(r[TIER_FIELD[k]] for r in rows) / sum(r["tickets"] * s.P_MATCH[k] for r in rows)
             for k in (3, 4, 5)}
    return {"beta": beta, "cal5": cal5, "cal4": cal4, "pattern": pattern, "m3": m3, "level": level}


def co_pickers(bundle, ticket):
    score = sum(bundle["beta"][i - 1] for i in ticket)
    adj = min(s.adjacent_pairs(sorted(ticket)), 3)
    return math.exp(bundle["cal5"]["c"] * score) * bundle["pattern"].get(adj, (1.0, 0.0))[0]


def conditional_winners(bundle, drawn, tickets):
    score = sum(bundle["beta"][i - 1] for i in drawn)
    return {5: tickets * s.P_MATCH[5] * co_pickers(bundle, drawn),
            4: tickets * s.P_MATCH[4] * math.exp(bundle["cal4"]["a"] + bundle["cal4"]["c"] * score),
            3: tickets * s.P_MATCH[3] * bundle["m3"] * math.exp(score)}


def ticket_prize(matches, pool, winners):
    if matches == 6:
        return pool / (winners + 1)
    return s.FIXED_PRIZE.get(matches, 0)


def run_checkpoint(sales, t, rng):
    rows, actual = sales[:t], sales[t]
    fc = s.forecast_sales(rows)
    share = s.jackpot_share_check(rows)
    coverage = share["winners"] / share["expected_winners"]
    p_won = 1 - math.exp(-fc["tickets"] * s.P_MATCH[6] * coverage)
    bundle = popularity_bundle(rows)
    ranked = sorted(range(1, s.N_BALLS + 1), key=lambda i: bundle["beta"][i - 1])
    counts = [0] * (s.N_BALLS + 1)
    for r in rows:
        for x in r["result"]:
            counts[x] += 1
    tickets = {"least-picked": sorted(ranked[:6]),
               "random": sorted(rng.sample(range(1, s.N_BALLS + 1), s.K)),
               "hot all-time": sorted(sorted(range(1, s.N_BALLS + 1), key=lambda i: (-counts[i], i))[:6])}
    ev_least = s.expected_value(fc["jackpot"], fc["tickets"], co_pickers(bundle, tickets["least-picked"]))
    ev_random = s.expected_value(fc["jackpot"], fc["tickets"])
    drawn = set(actual["result"])
    pre = {k: fc["tickets"] * s.P_MATCH[k] * bundle["level"][k] for k in (3, 4, 5)}
    post = conditional_winners(bundle, actual["result"], actual["tickets"])
    naive = {k: actual["tickets"] * s.P_MATCH[k] for k in (3, 4, 5)}
    results = {}
    for name, tk in tickets.items():
        m = len(drawn & set(tk))
        results[name] = {"numbers": tk, "matches": m,
                         "prize": ticket_prize(m, actual["pool"], actual["jackpot_winners"])}
    return {"id": actual["id"], "date": actual["date"], "drawn": actual["result"], "forecast": fc,
            "p_won": p_won, "won": actual["jackpot_winners"] > 0, "actual_tickets": actual["tickets"],
            "actual_jackpot": actual["pool"], "ev_least": ev_least["ev"], "ev_random": ev_random["ev"],
            "play": ev_least["ev"] >= s.TICKET, "pre": pre, "post": post, "naive": naive,
            "actual_winners": {k: actual[TIER_FIELD[k]] for k in (3, 4, 5)}, "tickets": results}


def pct(a, b):
    return (a / b - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    draws, _, _ = s.load_draws()
    sales = s.reconstruct_sales(draws)
    ids = [r["id"] for r in sales]
    end = ids.index(args.end) if args.end else len(sales) - 1
    checkpoints = [end - k * args.gap for k in range(args.count)][::-1]
    rng = random.Random(args.seed)
    out = [run_checkpoint(sales, t, rng) for t in checkpoints]
    detail = args.count <= 20

    p5 = s.P_MATCH[5]
    print(f"5 numbers correct (Giai Nhat): 1 in {1/p5:,.0f} per ticket, fixed 10,000,000 VND each, never shared;"
          f" worth {p5*1e7:,.0f} VND of every 10,000 VND ticket")
    print(f"{len(out)} checkpoints, gap {args.gap} draw(s): #{out[0]['id']} ({out[0]['date']}) .. #{out[-1]['id']} ({out[-1]['date']})."
          " Every forecast below uses only draws before that checkpoint.\n")
    if detail:
        print("A. SALES AND JACKPOT: forecast -> actual")
        print(f"  {'draw':>5s} {'date':10s} {'advertised':>10s} {'tickets forecast -> actual':>30s} {'jackpot at draw':>24s}"
              f" {'P(won)':>7s} {'won':>4s} {'EV/price':>8s} {'play':>5s}")
        for r in out:
            fc = r["forecast"]
            print(f"  {r['id']:>5d} {r['date']:10s} {fc['advertised']/1e9:8.1f}ty {fc['tickets']:>12,.0f} -> {r['actual_tickets']:>10,.0f}"
                  f" ({pct(fc['tickets'], r['actual_tickets']):+5.1f}%) {fc['jackpot']/1e9:8.1f} -> {r['actual_jackpot']/1e9:6.1f}ty"
                  f" {r['p_won']*100:6.1f}% {'yes' if r['won'] else 'no':>4s} {r['ev_least']/s.TICKET:8.2f} {'YES' if r['play'] else 'no':>5s}")
        print("\nB. PRIZE WINNERS IN THE WHOLE DRAW: forecast before draw / popularity model after seeing the numbers / actual")
        print(f"  {'draw':>5s} {'numbers drawn':17s} {'5 correct (10M)':>20s} {'4 correct (300k)':>22s} {'3 correct (30k)':>27s}")
        for r in out:
            cells = []
            for k, w in ((5, 20), (4, 22), (3, 27)):
                cells.append(f"{r['pre'][k]:,.0f}/{r['post'][k]:,.0f}/{r['actual_winners'][k]:,}".rjust(w))
            print(f"  {r['id']:>5d} {' '.join(f'{x:02d}' for x in r['drawn']):17s}" + "".join(cells))
        print("\nC. TICKETS CHOSEN AT EACH CHECKPOINT: numbers (matches)")
        for r in out:
            cells = [f"{name} {' '.join(f'{x:02d}' for x in v['numbers'])} ({v['matches']})" for name, v in r["tickets"].items()]
            print(f"  {r['id']:>5d}: " + " | ".join(cells))

    print("\nSUMMARY")
    err_t = [abs(pct(r["forecast"]["tickets"], r["actual_tickets"])) for r in out]
    err_rule = [abs(pct(r["forecast"]["rule_1_2"], r["actual_tickets"])) for r in out]
    bias_t = [pct(r["forecast"]["tickets"], r["actual_tickets"]) for r in out]
    err_j = [abs(pct(r["forecast"]["jackpot"], r["actual_jackpot"])) for r in out]
    print(f"  ticket sales forecast: median error {statistics.median(err_t):.1f}% (average bias {statistics.fmean(bias_t):+.1f}%);"
          f" the simple 'x1.2 last draw' rule: median error {statistics.median(err_rule):.1f}%")
    print(f"  jackpot-at-draw forecast: median error {statistics.median(err_j):.2f}%")
    base_rate = statistics.fmean(r["jackpot_winners"] > 0 for r in sales[: out[0]["id"] - 1])
    brier = statistics.fmean((r["p_won"] - r["won"]) ** 2 for r in out)
    brier0 = statistics.fmean((base_rate - r["won"]) ** 2 for r in out)
    print(f"  jackpot won: forecast {sum(r['p_won'] for r in out):.2f} wins, actual {sum(r['won'] for r in out)};"
          f" Brier score {brier:.3f} (always-average-rate baseline {brier0:.3f}; lower is better)")
    for k, label in ((5, "5 correct"), (4, "4 correct"), (3, "3 correct")):
        e_pre = statistics.median(abs(math.log(max(r["pre"][k], 1e-9) / max(r["actual_winners"][k], 0.5))) for r in out)
        e_post = statistics.median(abs(math.log(max(r["post"][k], 1e-9) / max(r["actual_winners"][k], 0.5))) for r in out)
        e_naive = statistics.median(abs(math.log(max(r["naive"][k], 1e-9) / max(r["actual_winners"][k], 0.5))) for r in out)
        print(f"  winners with {label}: median error before draw {100*(math.exp(e_pre)-1):.1f}%;"
              f" after draw with popularity model {100*(math.exp(e_post)-1):.1f}% (without popularity {100*(math.exp(e_naive)-1):.1f}%)")
    n_play = sum(r["play"] for r in out)
    print(f"  EV rule (buy only when expected value > price): would have bought at {n_play} of {len(out)} checkpoints")
    for name in out[0]["tickets"]:
        m = [r["tickets"][name]["matches"] for r in out]
        money = sum(r["tickets"][name]["prize"] for r in out)
        print(f"  ticket '{name}': {sum(m)} matches in {len(m)} draws (fair expectation {0.8*len(m):.1f}),"
              f" best {max(m)}, 5+ correct {sum(x >= 5 for x in m)} (expectation {len(m)*sum(s.P_MATCH[5:]):.4f}),"
              f" won {money:,.0f} VND for {len(m)*s.TICKET:,} VND spent")
    s.RESULTS.mkdir(exist_ok=True)
    path = s.RESULTS / f"checkpoints_{len(out)}x{args.gap}.json"
    path.write_text(json.dumps(out, default=str, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
