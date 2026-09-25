"""Power 6/55 study: exact odds with Jackpot 2 and the bonus ball, randomness tests, the 6/45 prediction models,
sales and expected value, ticket choice, a draw simulator and a long-run simulator.

Usage: python3 lottery/power655_study.py [--sims 200] [--runs 10] [--seed 2026] [--controls 3] [--mc 1000000]
"""

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import backtest_checkpoints as b
import optimise_tickets as o
import power645_study as s
import predict_draw as pr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "power655.jsonl"
RESULTS = ROOT / "results"

N, K = 55, 6
C_TOTAL = math.comb(N, K)
TICKET = 10_000
FIXED = {3: 50_000, 4: 500_000, 5: 40_000_000}
J1_MIN, J2_MIN, J1_CAP = 30_000_000_000, 3_000_000_000, 300_000_000_000
J1_PART = 0.9
P_MATCH = [math.comb(K, k) * math.comb(N - K, K - k) / C_TOTAL for k in range(K + 1)]
P_J1, P_J2, P_FIRST = 1 / C_TOTAL, K / C_TOTAL, K * (N - K - 1) / C_TOTAL
FIXED_EV = P_FIRST * FIXED[5] + P_MATCH[4] * FIXED[4] + P_MATCH[3] * FIXED[3]
ACCUM = 0.55 - FIXED_EV / TICKET
DRAW_WEEKDAYS = (1, 3, 5)
DRAWS_PER_YEAR = 156


def use_655_constants():
    s.N_BALLS, s.K, s.P, s.C_TOTAL = N, K, K / N, C_TOTAL
    s.P_MATCH = P_MATCH
    s.FIXED_PRIZE = dict(FIXED)
    s.JACKPOT_MIN = J1_MIN
    s.TICKET = TICKET


def load_draws():
    rows = {}
    for line in DATA.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    ids = sorted(rows)
    gaps = [i for i in range(1, ids[-1] + 1) if i not in rows]
    draws = [dict(rows[i], jackpot_value=rows[i]["j1_prize"]) for i in ids]
    return draws, gaps


def odds_table():
    ways = [("Jackpot 1: 6 numbers", 1), ("Jackpot 2: 5 numbers + bonus", K),
            ("First: 5 numbers", K * (N - K - 1)), ("Second: 4 numbers", math.comb(K, 4) * math.comb(N - K, 2)),
            ("Third: 3 numbers", math.comb(K, 3) * math.comb(N - K, 3))]
    return [(name, w, C_TOTAL / w) for name, w in ways]


def run_tests(draws, sims, rng):
    res = [d["result"] for d in draws]
    n = len(res)
    p = K / N
    tests = []
    counts = Counter(x for d in res for x in d)
    e = n * p
    stat = sum((counts[i] - e) ** 2 / e for i in range(1, N + 1)) * (N - 1) / (N - K)
    tests.append(("B1 each main number equally likely", f"chi2={stat:.1f}, df={N - 1}", s.chi2_sf(stat, N - 1)))
    bonus = Counter(d["bonus"] for d in draws)
    eb = n / N
    stat_b = sum((bonus[i] - eb) ** 2 / eb for i in range(1, N + 1))
    tests.append(("B2 each bonus number equally likely", f"chi2={stat_b:.1f}, df={N - 1}", s.chi2_sf(stat_b, N - 1)))
    zs = {i: (counts[i] - e) / math.sqrt(n * p * (1 - p)) for i in range(1, N + 1)}
    zi = max(zs, key=lambda i: abs(zs[i]))
    tests.append(("B3 most extreme main number", f"#{zi} z={zs[zi]:+.2f}",
                  min(1.0, 1 - (1 - 2 * s.norm_sf(abs(zs[zi]))) ** N)))
    years = defaultdict(list)
    for d in draws:
        years[int(d["date"][:4])].append(d["result"])
    stat_y = 0.0
    for yd in years.values():
        cy = Counter(x for d in yd for x in d)
        ey = len(yd) * p
        stat_y += sum((cy[i] - ey) ** 2 for i in range(1, N + 1)) / (ey * (1 - p) * N / (N - 1))
    df_y = (N - 1) * len(years)
    tests.append((f"B4 frequencies stable in every year ({len(years)} years)", f"chi2={stat_y:.1f}, df={df_y}",
                  s.chi2_sf(stat_y, df_y)))
    tbar = (n - 1) / 2
    stt = sum((t - tbar) ** 2 for t in range(n))
    trend = sum((sum(t - tbar for t, d in enumerate(res) if i in d) / math.sqrt(p * (1 - p) * stt)) ** 2
                for i in range(1, N + 1)) * (N - 1) / N
    tests.append(("B5 no number drifting up/down over time", f"chi2={trend:.1f}, df={N - 1}", s.chi2_sf(trend, N - 1)))
    var_overlap = K * p * (1 - p) * (N - K) / (N - 1)
    lag_z = []
    for lag in range(1, 11):
        ov = [len(set(res[t]) & set(res[t + lag])) for t in range(n - lag)]
        lag_z.append((sum(ov) / len(ov) - K * p) / math.sqrt(var_overlap / len(ov)))
    tests.append(("B6 no memory: overlap with draws 1..10 back", f"chi2={sum(z * z for z in lag_z):.1f}, df=10",
                  s.chi2_sf(sum(z * z for z in lag_z), 10)))
    follow = sum(draws[t]["bonus"] in draws[t + 1]["result"] for t in range(n - 1))
    ef = (n - 1) * p
    zf = (follow - ef) / math.sqrt((n - 1) * p * (1 - p))
    tests.append(("B7 bonus number does not predict the next draw", f"{follow} vs {ef:.1f} expected, z={zf:+.2f}",
                  2 * s.norm_sf(abs(zf))))
    sums = [sum(d) for d in res]
    stat_s, df_s = s.binned_chi2(sums, s.sum_distribution())
    tests.append(("B8 sum of the 6 numbers", f"chi2={stat_s:.1f}, df={df_s}, mean={sum(sums) / n:.1f}"
                  f" (exp {K * (N + 1) / 2:.1f})", s.chi2_sf(stat_s, df_s)))
    odd_n = (N + 1) // 2
    odd_pmf = {k: math.comb(odd_n, k) * math.comb(N - odd_n, K - k) / C_TOTAL for k in range(K + 1)}
    stat_o, df_o = s.binned_chi2([sum(x % 2 for x in d) for d in res], odd_pmf)
    tests.append(("B9 odd/even split", f"chi2={stat_o:.1f}, df={df_o}", s.chi2_sf(stat_o, df_o)))
    low_n = N // 2
    low_pmf = {k: math.comb(low_n, k) * math.comb(N - low_n, K - k) / C_TOTAL for k in range(K + 1)}
    stat_l, df_l = s.binned_chi2([sum(x <= low_n for x in d) for d in res], low_pmf)
    tests.append((f"B10 low (1-{low_n}) / high split", f"chi2={stat_l:.1f}, df={df_l}", s.chi2_sf(stat_l, df_l)))
    adj_pmf = {m: math.comb(K - 1, m) * math.comb(N - K + 1, K - m) / C_TOTAL for m in range(K)}
    stat_a, df_a = s.binned_chi2([s.adjacent_pairs(d) for d in res], adj_pmf)
    tests.append(("B11 consecutive numbers", f"chi2={stat_a:.1f}, df={df_a}", s.chi2_sf(stat_a, df_a)))
    obs_pt, obs_gr = s.pair_triple_stats(res), s.gap_and_runs_stats(res)
    sim_pt, sim_gr = [], []
    for _ in range(sims):
        fake = s.fair_lottery(n, rng)
        sim_pt.append(s.pair_triple_stats(fake))
        sim_gr.append(s.gap_and_runs_stats(fake))
    tests.append(("B12 gaps between appearances are geometric", f"chi2={obs_gr['gap_chi']:.1f} (MC)",
                  s.mc_p_upper(obs_gr["gap_chi"], [x["gap_chi"] for x in sim_gr])))
    tests.append(("B13 runs of appear/miss per number", f"sum z2={obs_gr['runs_z2']:.1f} (MC)",
                  s.mc_p_upper(obs_gr["runs_z2"], [x["runs_z2"] for x in sim_gr])))
    tests.append(("B14 pairs appear together evenly", f"chi2={obs_pt['pair_chi']:.0f} (MC)",
                  s.mc_p_upper(obs_pt["pair_chi"], [x["pair_chi"] for x in sim_pt])))
    tests.append(("B15 most frequent pair", f"{obs_pt['pair_top'][0][0]} x{obs_pt['pair_max']} (MC)",
                  s.mc_p_upper(obs_pt["pair_max"], [x["pair_max"] for x in sim_pt])))
    tests.append(("B16 most frequent triple", f"{obs_pt['triple_top'][0][0]} x{obs_pt['triple_max']} (MC)",
                  s.mc_p_upper(obs_pt["triple_max"], [x["triple_max"] for x in sim_pt])))
    adj_p = s.holm([t[2] for t in tests])
    return [(name, detail, pv, pa) for (name, detail, pv), pa in zip(tests, adj_p)], counts


def reconstruct_sales(draws):
    out, debt, prev = [], float(J1_MIN + J2_MIN), None
    for d in draws:
        fixed = FIXED[5] * d["first_winners"] + FIXED[4] * d["second_winners"] + FIXED[3] * d["third_winners"]
        if prev is None:
            base1, base2 = J1_MIN, J2_MIN
        else:
            if prev["j1_winners"] > 0:
                base1 = J1_MIN
                debt += J1_MIN
            elif prev["pool1"] > J1_CAP and prev["j2_winners"] > 0:
                base1 = J1_CAP
            else:
                base1 = prev["pool1"]
            if prev["j2_winners"] > 0:
                base2 = J2_MIN
                debt += J2_MIN
            else:
                base2 = prev["pool2"]
        from_3match = d["third_winners"] / P_MATCH[3]
        overflow = d["j1_winners"] == 0 and d["j2_winners"] > 0 and d["j1_prize"] == J1_CAP
        if overflow:
            pool1, pool2 = J1_CAP, d["j2_prize"]
            accum = pool2 - base2 - base1 + J1_CAP
            accum2 = accum
        else:
            pools = [d["j1_prize"]] + ([d["j1_prize"] * d["j1_winners"]] if d["j1_winners"] >= 2 else [])
            pool1 = min(pools, key=lambda j: abs(s._revenue((j - base1) / J1_PART, fixed, debt)[0] / TICKET
                                                 - from_3match))
            accum = (pool1 - base1) / J1_PART
            pools2 = [d["j2_prize"]] + ([d["j2_prize"] * d["j2_winners"]] if d["j2_winners"] >= 2 else [])
            pool2 = min(pools2, key=lambda j: abs((j - base2) / (1 - J1_PART) - accum))
            accum2 = (pool2 - base2) / (1 - J1_PART)
        revenue, debt = s._revenue(accum, fixed, debt)
        row = {**d, "pool1": pool1, "pool2": pool2, "adv1": base1, "adv2": base2, "accum": accum, "accum_j2": accum2,
               "revenue": revenue, "tickets": revenue / TICKET, "tickets_from_3match": from_3match,
               "debt_after": debt, "jackpot_winners": d["j1_winners"], "overflow": overflow}
        out.append(row)
        prev = row
    return out


def sales_model(rows, window=300):
    xs, ys, after_win = [], [], []
    for i in range(max(2, len(rows) - window), len(rows)):
        if rows[i - 1]["j1_winners"] > 0:
            after_win.append(rows[i]["tickets"] / rows[i - 1]["tickets"])
            continue
        xs.append(math.log((rows[i]["adv1"] + rows[i]["adv2"]) / (rows[i - 1]["adv1"] + rows[i - 1]["adv2"])))
        ys.append(math.log(rows[i]["tickets"] / rows[i - 1]["tickets"]))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope, statistics.median(after_win)


def next_advertised(rows):
    last = rows[-1]
    if last["j1_winners"] > 0:
        adv1 = J1_MIN
    elif last["pool1"] > J1_CAP and last["j2_winners"] > 0:
        adv1 = J1_CAP
    else:
        adv1 = last["pool1"]
    adv2 = J2_MIN if last["j2_winners"] > 0 else last["pool2"]
    return adv1, adv2


def forecast(rows):
    alpha, elasticity, reset_ratio = sales_model(rows)
    last = rows[-1]
    adv1, adv2 = next_advertised(rows)
    if last["j1_winners"] > 0:
        tickets = last["tickets"] * reset_ratio
    else:
        tickets = last["tickets"] * math.exp(alpha + elasticity * math.log((adv1 + adv2) / (last["adv1"] + last["adv2"])))
    debt = last["debt_after"] + (J1_MIN if last["j1_winners"] > 0 else 0) + (J2_MIN if last["j2_winners"] > 0 else 0)
    revenue = tickets * TICKET
    accum = ACCUM * revenue - min(0.20 * revenue, debt)
    return {"adv1": adv1, "adv2": adv2, "tickets": tickets, "j1": adv1 + J1_PART * accum,
            "j2": adv2 + (1 - J1_PART) * accum, "elasticity": elasticity}


def forecast_backtest(sales, count=300):
    errs = []
    for t in range(len(sales) - count, len(sales)):
        fc = forecast(sales[:t])
        errs.append((abs(fc["tickets"] / sales[t]["tickets"] - 1), abs(fc["j1"] / sales[t]["pool1"] - 1),
                     abs(fc["j2"] / sales[t]["pool2"] - 1) if sales[t]["j2_winners"] == 0 else None))
    return (statistics.median(e[0] for e in errs), statistics.median(e[1] for e in errs),
            statistics.median(e[2] for e in errs if e[2] is not None))


def net_prize(j):
    return j - 0.10 * max(0, j - 10_000_000)


def expected_value(j1, j2, tickets, mult=1.0):
    def share(p_hit):
        lam = max(tickets - 1, 0) * p_hit * mult
        return (1 - math.exp(-lam)) / lam if lam > 0 else 1.0
    s1, s2 = share(P_J1), share(P_J2)
    part1, part2 = P_J1 * s1 * net_prize(j1), P_J2 * s2 * net_prize(j2)
    return {"ev": FIXED_EV + part1 + part2, "fixed": FIXED_EV, "j1_part": part1, "j2_part": part2,
            "share1": s1, "share2": s2}


class Scorer:
    def __init__(self, bundle, j1, j2, tickets):
        self.bundle, self.j1, self.j2, self.tickets = bundle, j1, j2, tickets
        self.cache = {}

    def co_pickers(self, t):
        return b.co_pickers(self.bundle, t)

    def ev(self, t):
        key = tuple(t)
        if key not in self.cache:
            self.cache[key] = expected_value(self.j1, self.j2, self.tickets, self.co_pickers(t))
        return self.cache[key]


def enumerate_main(tickets):
    inc = [0] * (N + 2)
    for j, t in enumerate(tickets):
        for x in t:
            inc[x] += 1 << (4 * j)
    cnt = Counter()
    for a in range(1, N - 4):
        sa = inc[a]
        for b_ in range(a + 1, N - 3):
            sb = sa + inc[b_]
            for c in range(b_ + 1, N - 2):
                sc = sb + inc[c]
                for d in range(c + 1, N - 1):
                    sd = sc + inc[d]
                    for e in range(d + 1, N):
                        cnt.update(map((sd + inc[e]).__add__, inc[e + 1:N + 1]))
    out = Counter()
    for key, w in cnt.items():
        out[tuple(sorted(((key >> (4 * j)) & 15 for j in range(len(tickets))), reverse=True))] += w
    assert sum(out.values()) == C_TOTAL
    return out


def exact_summary(tickets):
    dist = enumerate_main(tickets)
    total = C_TOTAL
    r = Counter()
    for hits, w in dist.items():
        r["any"] += w * (hits[0] >= 3)
        r["four"] += w * (hits[0] >= 4)
        r["five"] += w * (hits[0] >= 5)
        r["no_prize"] += w * (hits[0] < 3)
    out = {k: v / total for k, v in r.items()}
    out.update(jackpot_chances(tickets))
    return out, dist


def jackpot_chances(tickets):
    j2_outcomes = set()
    for t in tickets:
        rest = [y for y in range(1, N + 1) if y not in t]
        for x in t:
            core = tuple(v for v in t if v != x)
            for y in rest:
                j2_outcomes.add((tuple(sorted(core + (y,))), x))
    return {"j2_any": len(j2_outcomes) / (C_TOTAL * (N - K)), "j1_any": len({tuple(t) for t in tickets}) / C_TOTAL}


def simulate(designs, sims, rng, j1, j2):
    masks = {name: [sum(1 << x for x in t) for t in d] for name, d in designs.items()}
    stats = {name: Counter() for name in designs}
    balls = list(range(1, N + 1))
    for _ in range(sims):
        draw = rng.sample(balls, K + 1)
        dm = 0
        for x in draw[:K]:
            dm |= 1 << x
        bm = 1 << draw[K]
        for name, ms in masks.items():
            best, money, jack = 0, 0, 0
            for m in ms:
                h = bin(dm & m).count("1")
                if h == 6:
                    jack += net_prize(j1)
                elif h == 5 and m & bm:
                    jack += net_prize(j2)
                    h = 5.5
                elif h >= 3:
                    money += FIXED[h]
                best = max(best, h)
            st = stats[name]
            st["money"] += money
            st["jack"] += jack
            st["any"] += best >= 3
            st["four"] += best >= 4
            st["five"] += best >= 5
            st["j2"] += best >= 5.5
    return {name: {k: v / sims for k, v in st.items()} for name, st in stats.items()}


def money_distribution(dist):
    money = Counter()
    for hits, w in dist.items():
        money[sum(FIXED.get(h, 0) for h in hits)] += w
    total = sum(money.values())
    values = sorted(money)
    cum, acc = [], 0
    for v in values:
        acc += money[v]
        cum.append(acc / total)
    return values, cum


def long_run(values, cum, history, n_tickets, years, players, rng):
    draws = DRAWS_PER_YEAR * years
    cost = draws * n_tickets * TICKET
    fixed_mean = sum(v * (c - (cum[i - 1] if i else 0)) for i, (v, c) in enumerate(zip(values, cum)))
    jack_mean = statistics.fmean(sum(p_hit * value for p_hit, value in events) for events in history)
    results = []
    for _ in range(players):
        start = rng.randrange(len(history))
        won = 0.0
        for i in range(draws):
            won += values[bisect.bisect_left(cum, rng.random())]
            for p_hit, value in history[(start + i) % len(history)]:
                if rng.random() < p_hit * n_tickets:
                    won += value
        results.append(won - cost)
    results.sort()
    return {"cost": cost, "median": results[len(results) // 2], "ahead": sum(r >= 0 for r in results) / players,
            "p10": results[players // 10], "p90": results[players * 9 // 10],
            "expected": draws * (fixed_mean + n_tickets * jack_mean) - cost}


def jackpot_history(sales):
    out = []
    for r in sales:
        e = expected_value(r["pool1"], r["pool2"], r["tickets"])
        out.append([(P_J1, net_prize(r["pool1"]) * e["share1"]), (P_J2, net_prize(r["pool2"]) * e["share2"])])
    return out


def next_draw_date(last_date):
    d = dt.date.fromisoformat(last_date) + dt.timedelta(days=1)
    while d.weekday() not in DRAW_WEEKDAYS:
        d += dt.timedelta(days=1)
    return d.isoformat()


def fmt(t):
    return " ".join(f"{x:02d}" for x in sorted(t))


def render_snapshot(snap, digest):
    names = list(snap["models"])
    fc = snap["forecast"]
    lines = [f"Power 6/55 draw #{snap['target_draw']} on {snap['target_date']} at 18:00. Prediction frozen"
             f" {snap['created_at']} from {snap['data_through']['draws']:,} draws (through #{snap['data_through']['id']},"
             f" {snap['data_through']['date']}).", f"SHA-256 of the JSON file: {digest}", "",
             f"Chance that each number is among the 6 main numbers (fair = {snap['fair_p']*100:.2f}% for every number)",
             "  no" + "".join(f"{pr.SHORT[n]:>8s}" for n in names)]
    for i in range(1, N + 1):
        lines.append(f"  {i:02d}" + "".join(f"{snap['models'][n]['p'][i - 1]*100:7.2f}%" for n in names))
    lines += ["", "Each model's top 6:"]
    lines += [f"  {n:46s} {fmt(snap['models'][n]['top6'])}" for n in names]
    lines += ["", f"Forecast: {fc['tickets']:,.0f} tickets; Jackpot 1 at the draw {fc['j1']/1e9:.1f} ty, Jackpot 2"
              f" {fc['j2']/1e9:.2f} ty; chance someone wins Jackpot 1 {fc['p_won1']*100:.1f}%, Jackpot 2"
              f" {fc['p_won2']*100:.1f}%", "", "Tickets to check after the draw:"]
    for label, tickets in snap["tickets"].items():
        lines.append(f"  {label}:")
        lines += [f"      {fmt(t)}" for t in tickets]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--mc", type=int, default=1_000_000)
    ap.add_argument("--players", type=int, default=5000)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        print(f"data: {update.refresh('655')}")
    use_655_constants()
    rng = random.Random(args.seed)
    RESULTS.mkdir(exist_ok=True)
    report = {}

    draws, gaps = load_draws()
    n = len(draws)
    print(f"DATA: {n} Power 6/55 draws, #{draws[0]['id']} ({draws[0]['date']}) .. #{draws[-1]['id']}"
          f" ({draws[-1]['date']}); missing ids: {len(gaps)}")

    print(f"\nPART A - EXACT ODDS FOR ONE 10,000 VND TICKET (C(55,6) = {C_TOTAL:,}; bonus drawn from the other 49)")
    for name, ways, one_in in odds_table():
        print(f"  {name:32s} {ways:>9,} of {C_TOTAL:,} draws   1 in {one_in:>14,.1f}")
    p_any = (1 + K + K * (N - K - 1) + math.comb(K, 4) * math.comb(N - K, 2) + math.comb(K, 3) * math.comb(N - K, 3)) / C_TOTAL
    print(f"  any prize: {p_any*100:.3f}% (1 in {1/p_any:.1f});  6/45 for comparison: 2.383% (1 in 42.0)")
    print(f"  fixed prizes return {FIXED_EV:,.1f} VND per ticket ({FIXED_EV/TICKET*100:.2f}%); jackpots get the other"
          f" {ACCUM*100:.2f}% of sales once the guarantee fund is repaid: Jackpot 1 {ACCUM*J1_PART*100:.2f}%,"
          f" Jackpot 2 {ACCUM*(1-J1_PART)*100:.2f}% (rules: 37.47% and 4.16%)")
    report["odds"] = {"p_any": p_any, "fixed_ev": FIXED_EV, "accum": ACCUM}

    print(f"\nPART B - RANDOMNESS TESTS ON 6/55 (Holm-corrected, {args.sims} fake lotteries for the MC tests)")
    tests, counts = run_tests(draws, args.sims, rng)
    for name, detail, pv, pa in tests:
        print(f"  {name:48s} {detail:44s} p={s.fmt_p(pv):>6s}  Holm p={s.fmt_p(pa):>6s}"
              + ("  <-- signal" if pa < 0.05 else ""))
    hb = s.hierarchical_beta_binomial([counts[i] for i in range(1, N + 1)], n)
    print(f"  Bayes factor fair : biased = {hb['bf_fair_vs_biased']:.1f} : 1; spread of true per-number chances:"
          f" 95% upper bound SD {hb['sd_95']:.4f} vs p {K/N:.4f} (+-{hb['sd_95']/(K/N)*100:.1f}% relative)")
    report["tests"] = [{"name": a, "detail": d, "p": pv, "holm_p": pa} for a, d, pv, pa in tests]
    report["bayes"] = hb

    print(f"\nPART C - THE SAME 9 PREDICTION MODELS, {args.runs} WALK-FORWARD RUNS + {args.controls} FAKE LOTTERIES")
    wf = s.walk_forward(draws, args.runs, args.seed, "real")
    pooled = s.pool_runs(wf)
    controls = []
    for ci in range(args.controls):
        fake = [{"id": d["id"], "date": d["date"], "result": r, "jackpot_value": d["jackpot_value"]}
                for d, r in zip(draws, s.fair_lottery(n, random.Random(args.seed + 1 + ci)))]
        controls.append(s.pool_runs(s.walk_forward(fake, args.runs, args.seed + ci, f"fake{ci + 1}")))
    print(f"  runs cover #{wf['runs'][0]['draws'][0]}..#{wf['runs'][-1]['draws'][1]}; fair expectation: hits/draw"
          f" {K*K/N:.3f}, prize rate {sum(P_MATCH[3:])*100:.2f}%")
    print(f"  {'model':46s} {'log-score vs fair':>18s} {'z':>6s} {'runs better':>11s} {'runs >2SE':>9s}"
          f" {'fake runs better':>17s} {'hits/draw':>9s}")
    for name, st in sorted(pooled.items(), key=lambda kv: -kv[1]["ls_diff"]):
        fake_won = " ".join(str(c_[name]["runs_won"]) for c_ in controls)
        print(f"  {name:46s} {st['ls_diff']:+10.4f}+-{st['ls_se']:.4f} {st['z']:+6.2f} {st['runs_won']:>7d}/{args.runs}"
              f" {st['runs_sig']:>6d}/{args.runs} {fake_won:>17s} {st['hits']:9.3f}")
    report["walk_forward_real"] = pooled
    report["walk_forward_fake"] = controls

    print("\nPART D - SALES, JACKPOTS AND PLAYER HABITS (rebuilt from jackpot growth under the published rules)")
    sales = reconstruct_sales(draws)
    ratio_j2 = [r["accum_j2"] / r["accum"] for r in sales if r["accum"] > 0 and not r["overflow"]]
    ratio_3 = [r["tickets"] / r["tickets_from_3match"] for r in sales if r["third_winners"] > 0]
    print(f"  check 1, Jackpot 2 growth / Jackpot 1 growth matches the 10/90 rule: median ratio"
          f" {statistics.median(ratio_j2):.4f}, middle 90% {sorted(ratio_j2)[len(ratio_j2)//20]:.4f}.."
          f"{sorted(ratio_j2)[len(ratio_j2)*19//20]:.4f}")
    print(f"  check 2, tickets from jackpot growth vs tickets implied by 3-match winners: median ratio"
          f" {statistics.median(ratio_3):.3f}")
    j1_w = sum(r["j1_winners"] for r in sales)
    j2_w = sum(r["j2_winners"] for r in sales)
    e1 = sum(r["tickets"] * P_J1 for r in sales)
    e2 = sum(r["tickets"] * P_J2 for r in sales)
    print(f"  Jackpot 1 winners {j1_w} vs {e1:.1f} expected if players picked at random;"
          f" Jackpot 2 winners {j2_w} vs {e2:.1f}; total tickets sold {sum(r['tickets'] for r in sales)/1e6:,.0f} million")
    err_t, err_j1, err_j2 = forecast_backtest(sales)
    print(f"  next-draw forecast checked on the last 300 draws: median error tickets {err_t*100:.1f}%,"
          f" Jackpot 1 {err_j1*100:.2f}%, Jackpot 2 {err_j2*100:.2f}%")
    bundle = b.popularity_bundle(sales)
    beta = bundle["beta"]
    ranked = sorted(range(1, N + 1), key=lambda i: beta[i - 1])
    print(f"  least-picked numbers: {' '.join(f'{i:02d}' for i in ranked[:10])}")
    print(f"  most-picked numbers:  {' '.join(f'{i:02d}' for i in ranked[-10:][::-1])}")
    print(f"  5-correct winners follow the popularity score with exp({bundle['cal5']['a']:+.3f} +"
          f" {bundle['cal5']['c']:.2f} x score)")
    report["sales_checks"] = {"j2_ratio_median": statistics.median(ratio_j2), "t_vs_3match": statistics.median(ratio_3),
                              "j1_winners": j1_w, "j1_expected": e1, "j2_winners": j2_w, "j2_expected": e2,
                              "forecast_err": [err_t, err_j1, err_j2]}

    fc = forecast(sales)
    target = draws[-1]["id"] + 1
    target_date = next_draw_date(draws[-1]["date"])
    coverage = j1_w / e1
    p_won1 = 1 - math.exp(-fc["tickets"] * P_J1 * coverage)
    p_won2 = 1 - math.exp(-fc["tickets"] * P_J2 * (j2_w / e2))
    ev_r = expected_value(fc["j1"], fc["j2"], fc["tickets"])
    print(f"\nPART E - NEXT DRAW #{target} ({target_date}): forecast and expected value")
    print(f"  advertised Jackpot 1 {fc['adv1']/1e9:.1f} ty, Jackpot 2 {fc['adv2']/1e9:.2f} ty;"
          f" forecast {fc['tickets']:,.0f} tickets -> at the draw Jackpot 1 {fc['j1']/1e9:.1f} ty,"
          f" Jackpot 2 {fc['j2']/1e9:.2f} ty")
    print(f"  chance someone wins Jackpot 1: {p_won1*100:.1f}%; Jackpot 2: {p_won2*100:.1f}%")
    print(f"  random ticket: EV {ev_r['ev']:,.0f} VND per 10,000 VND = fixed {ev_r['fixed']:,.0f} + Jackpot 1"
          f" {ev_r['j1_part']:,.0f} + Jackpot 2 {ev_r['j2_part']:,.0f}  -> EV/price {ev_r['ev']/TICKET:.2f}")
    lo, hi = J1_MIN, 3e12
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if expected_value(mid, fc["j2"], fc["tickets"])["ev"] >= TICKET else (mid, hi)
    print(f"  Jackpot 1 needed for EV = price at these sales: {hi/1e9:,.0f} ty (overflow above 300 ty goes to"
          f" Jackpot 2, so the true break-even is higher still)")
    mega = RESULTS / "latest.json"
    if mega.exists():
        m = json.loads(mega.read_text())
        mf = m.get("ev", {}).get("scenarios", {}).get("model forecast")
        if mf:
            print(f"  Mega 6/45 next draw for comparison: EV {mf['ev']:,.0f} VND per 10,000 VND ticket"
                  f" (EV/price {mf['ev']/TICKET:.2f}), jackpot {mf['jackpot']/1e9:.1f} ty")
    hist_ev = [expected_value(r["pool1"], r["pool2"], r["tickets"])["ev"] for r in sales]
    print(f"  history: {sum(e >= TICKET for e in hist_ev)} of {len(hist_ev)} draws had EV >= price;"
          f" mean EV {statistics.fmean(hist_ev):,.0f} VND; best {max(hist_ev):,.0f} VND")
    report["next"] = {"draw": target, "date": target_date, **fc, "p_won1": p_won1, "p_won2": p_won2, "ev_random": ev_r}

    print("\nPART F - TICKETS FOR THE NEXT DRAW (spread for small prizes, unpopular numbers for a bigger jackpot share)")
    scorer = Scorer(bundle, fc["j1"], fc["j2"], fc["tickets"])
    trng = random.Random(args.seed)
    designs = {}
    for k in (5, 10):
        designs[f"{k} spread + unpopular"] = o.optimise(scorer, k, 0 if k <= 6 else 1, trng, max_uses=2)
        designs[f"{k} random quick picks"] = [sorted(trng.sample(range(1, N + 1), K)) for _ in range(k)]
    sim = simulate(designs, args.mc, random.Random(args.seed + 1), fc["j1"], fc["j2"])
    print(f"  {'option':28s} {'win something':>22s} {'4+ correct':>9s} {'5+ correct':>10s} {'Jackpot 2':>12s}"
          f" {'Jackpot 1':>12s} {'EV':>9s} {'EV/price':>8s}")
    exacts = {}
    for name, d in designs.items():
        ex, dist = exact_summary(d)
        exacts[name] = (ex, dist)
        ev = sum(scorer.ev(t)["ev"] for t in d)
        print(f"  {name:28s} {ex['any']*100:6.2f}% (sim {sim[name]['any']*100:5.2f}%) {ex['four']*100:8.3f}%"
              f" {ex['five']*100:9.4f}% 1 in {1/ex['j2_any']:>9,.0f} 1 in {1/ex['j1_any']:>9,.0f}"
              f" {ev:9,.0f} {ev/(len(d)*TICKET):8.2f}")
    for name, d in designs.items():
        if "spread" in name:
            print(f"  {name}:")
            for t in d:
                e = scorer.ev(t)
                print(f"      {fmt(t)}   co-pickers x{scorer.co_pickers(t):.2f}  EV {e['ev']:,.0f} VND")
    report["tickets"] = {name: {"tickets": d, "exact": exacts[name][0], "sim": sim[name],
                                "ev": sum(scorer.ev(t)["ev"] for t in d)} for name, d in designs.items()}

    print(f"\nPART G - LONG-RUN SIMULATOR: {args.players:,} players buy the same tickets every draw, starting at a"
          f" random past draw and living through the real history of jackpot sizes and sales")
    history = jackpot_history(sales)
    for name in ("10 spread + unpopular", "5 spread + unpopular"):
        values, cum = money_distribution(exacts[name][1])
        k = len(designs[name])
        for years in (1, 5):
            lr = long_run(values, cum, history, k, years, args.players, random.Random(args.seed + years))
            print(f"  {name}, {years} year(s) = {lr['cost']/1e6:,.2f} million VND spent: median result"
                  f" {lr['median']/1e6:+,.2f} million; middle 80% {lr['p10']/1e6:+,.2f} .. {lr['p90']/1e6:+,.2f} million;"
                  f" ended ahead {lr['ahead']*100:.2f}% of players; exact expected result {lr['expected']/1e6:+,.2f} million")
            report.setdefault("long_run", {})[f"{name}, {years}y"] = lr

    hist = s.History([d["result"] for d in draws])
    models, kappa = pr.model_probabilities(hist, args.seed)
    snap = {"game": "Power 6/55", "target_draw": target, "target_date": target_date,
            "draw_time": f"{target_date}T18:00:00+07:00", "sales_close": f"{target_date}T17:45:00+07:00",
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_through": {"id": draws[-1]["id"], "date": draws[-1]["date"], "draws": n},
            "history_sha256": pr.history_hash([{**d, "jackpot_value": d["j1_prize"], "jackpot_winners": d["j1_winners"]}
                                               for d in draws]),
            "seed": args.seed, "hierarchical_kappa": kappa, "fair_p": K / N, "models": models,
            "forecast": report["next"],
            "tickets": {name: d for name, d in designs.items() if "spread" in name}}
    print("\nPART H - EACH MODEL'S CHANCE FOR EVERY NUMBER IN THE NEXT DRAW (fair = 10.91% each)")
    for name, m in models.items():
        p = m["p"]
        print(f"  {name:46s} range {min(p)*100:5.2f}%..{max(p)*100:5.2f}%   top 6 {fmt(m['top6'])}")
    path = ROOT / "predictions" / f"power655_draw_{target:05d}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        print(f"  {path.name} already exists; the frozen prediction was not overwritten")
    else:
        path.write_text(json.dumps(snap, indent=1, default=str))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".txt").write_text(render_snapshot(snap, digest))
        print(f"  frozen to {path.name} and {path.with_suffix('.txt').name}; SHA-256 {digest}")
    (RESULTS / "power655_latest.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nsaved {RESULTS / 'power655_latest.json'}")


if __name__ == "__main__":
    main()
