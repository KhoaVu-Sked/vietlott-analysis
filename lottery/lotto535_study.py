"""Lotto 5/35 study: exact odds with the special number, randomness tests, sales rebuilt from jackpot growth, the jackpot
share-out rule ("Chia Giải Độc Đắc") and what it paid, when the next share-out is due, tickets and a long-run simulator.

Usage: python3 lottery/lotto535_study.py [--sims 200] [--seed 2026] [--no-update]
"""

import argparse
import bisect
import datetime as dt
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import coverage
import power645_study as s

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lotto535.jsonl"
RESULTS = ROOT / "results"
N, K, S = 35, 5, 12
MAIN_TOTAL = math.comb(N, K)
TOTAL = MAIN_TOTAL * S
TICKET = 10_000
J_MIN, SHARE_OUT_AT = 6_000_000_000, 12_000_000_000
GUARANTEE = 0.30
TIERS = ("first", "second", "third", "fourth", "fifth", "consolation")
BASE = {"first": 10_000_000, "second": 5_000_000, "third": 500_000, "fourth": 100_000, "fifth": 30_000,
        "consolation": 10_000}
MAIN_WAYS = [math.comb(K, h) * math.comb(N - K, K - h) for h in range(K + 1)]
P_MAIN = [w / MAIN_TOTAL for w in MAIN_WAYS]
P_JP = 1 / TOTAL
P_TIER = {"first": P_MAIN[5] * (S - 1) / S, "second": P_MAIN[4] / S, "third": P_MAIN[4] * (S - 1) / S,
          "fourth": P_MAIN[3] / S, "fifth": P_MAIN[3] * (S - 1) / S, "consolation": sum(P_MAIN[:3]) / S}
FIXED_EV = sum(P_TIER[t] * BASE[t] for t in TIERS)
ACCUM = 0.55 - FIXED_EV / TICKET


def use_535_constants():
    s.N_BALLS, s.K, s.P, s.C_TOTAL = N, K, K / N, MAIN_TOTAL
    s.P_MATCH = P_MAIN


def load_draws():
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["id"])
    gaps = [i for i in range(1, rows[-1]["id"] + 1) if i not in {r["id"] for r in rows}]
    return rows, gaps


def tier_of(hits, special_hit):
    if hits == 5:
        return "jackpot" if special_hit else "first"
    if hits == 4:
        return "second" if special_hit else "third"
    if hits == 3:
        return "fourth" if special_hit else "fifth"
    return "consolation" if special_hit else None


def net(v):
    return v - 0.10 * max(0, v - 10_000_000)


def is_share_out(r):
    return any(r[f"{t}_prize"] > BASE[t] for t in BASE)


def reconstruct_sales(rows):
    out, debt, prev, reset = [], float(J_MIN), None, True
    for r in rows:
        fixed = sum(r[f"{t}_winners"] * BASE[t] for t in TIERS)
        if reset and prev is not None:
            debt += J_MIN
        base = J_MIN if reset else prev
        accum = r["jackpot_prize"] - base
        if debt > 0:
            rev = (accum + fixed) / (0.55 - GUARANTEE)
            if GUARANTEE * rev > debt:
                rev, debt = (accum + fixed + debt) / 0.55, 0.0
            else:
                debt -= GUARANTEE * rev
        else:
            rev = (accum + fixed) / 0.55
        out.append({**r, "tickets": rev / TICKET, "debt_after": debt, "base": base, "split": is_share_out(r),
                    "t_fifth": r["fifth_winners"] / P_TIER["fifth"], "t_refund": r["consolation_winners"] / P_TIER["consolation"]})
        reset = out[-1]["split"] or r["jackpot_winners"] > 0
        prev = r["jackpot_prize"]
    return out


def designated_share_outs(rows):
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    found, pending = [], None
    for r in rows:
        if pending is not None and r["id"] == pending:
            found.append(r["id"])
            pending = None
            continue
        if r["jackpot_winners"] > 0 or is_share_out(r):
            pending = None
            continue
        if pending is None and r["jackpot_prize"] > SHARE_OUT_AT:
            nxt = (dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1)).isoformat()
            pending = max(x["id"] for x in by_date[nxt]) if nxt in by_date else ("next-day", nxt)
    return found, pending


def paid(r, after_tax):
    f = net if after_tax else (lambda v: v)
    return sum(r[f"{t}_winners"] * f(r[f"{t}_prize"]) for t in TIERS) + r["jackpot_winners"] * f(r["jackpot_prize"])


def ols(xs, ys):
    k = len(xs[0])
    xtx = [[sum(x[a] * x[b] for x in xs) for b in range(k)] for a in range(k)]
    xty = [sum(x[a] * y for x, y in zip(xs, ys)) for a in range(k)]
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(k):
            if r != c:
                f = m[r][c] / m[c][c]
                for j in range(c, k + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][k] / m[i][i] for i in range(k)]


def share_out_crowd(sales, ids):
    by_id = {r["id"]: r for r in sales}
    xs, ys = [], []
    for i in ids:
        jp = by_id[i - 1]["jackpot_prize"]
        xs.append([1.0, (i - ids[-1]) / 100, math.log(jp / 20e9)])
        ys.append(math.log(by_id[i]["tickets"]))
    b = ols(xs, ys)
    resid = [y - sum(bi * xi for bi, xi in zip(b, x)) for x, y in zip(xs, ys)]
    return b, math.sqrt(sum(e * e for e in resid) / max(1, len(ys) - 3)), ids[-1]


def share_out_value(j_prev, draw_id, crowd, tax_share, c_share):
    b, sd, ref = crowd
    t = math.exp(b[0] + b[1] * (draw_id - ref) / 100 + b[2] * math.log(j_prev / 20e9))
    j = j_prev + ACCUM * t * TICKET
    return {"tickets": t, "jackpot": j, "ev": FIXED_EV + tax_share * j / t,
            "ev_low": FIXED_EV + tax_share * (j_prev * math.exp(-sd) / t + ACCUM * TICKET),
            "ev_high": FIXED_EV + tax_share * (j_prev * math.exp(sd) / t + ACCUM * TICKET),
            "p_share": math.exp(-c_share * t * P_JP)}


def estimated_share_out_prizes(j, t):
    out = {tier: BASE[tier] + j * part / max(1.0, t * P_TIER[tier]) for tier, part in
           (("first", 1 / 3), ("second", 1 / 6), ("third", 1 / 6), ("fourth", 1 / 6), ("fifth", 1 / 6))}
    out["consolation"] = BASE["consolation"]
    return out


def money_table(tickets, prizes):
    money, p_real = exact_money(tickets, {t: net(v) for t, v in prizes.items()})
    total = sum(money.values())
    values = sorted(money)
    cum, acc = [], 0
    for v in values:
        acc += money[v]
        cum.append(acc / total)
    return {"values": values, "cum": cum, "p_real": p_real,
            "mean": sum(v * w for v, w in money.items()) / total,
            "any": sum(w for v, w in money.items() if v > 0) / total,
            "ahead": lambda cost: sum(w for v, w in money.items() if v >= cost) / total}


def sample(table, rng):
    return table["values"][bisect.bisect_left(table["cum"], rng.random())]


def normal_value(jackpot, tickets, coverage_factor):
    lam = max(tickets - 1, 0) * P_JP * coverage_factor
    share = (1 - math.exp(-lam)) / lam if lam > 0 else 1.0
    return FIXED_EV + P_JP * share * net(jackpot)


def next_share_out(sales, coverage_factor, rng, runs=4000, horizon=400):
    last = sales[-1]
    typical = {}
    for r in sales[-120:]:
        if not r["split"] and r["tickets"] > 0:
            typical.setdefault(r["id"] % 2, []).append(r["tickets"])
    fixed_rate = FIXED_EV / TICKET
    first_draw = last["id"] + 1
    reset = last["split"] or last["jackpot_winners"] > 0
    results = Counter()
    for _ in range(runs):
        j = J_MIN if reset else last["jackpot_prize"]
        debt = last["debt_after"] + (J_MIN if reset else 0.0)
        pending_day, over_draw = None, None
        found = None
        for step in range(horizon):
            draw_id = first_draw + step
            t = rng.choice(typical[draw_id % 2])
            rev = t * TICKET
            g = min(GUARANTEE * rev, debt)
            debt -= g
            j += 0.55 * rev - fixed_rate * rev - g
            day = step // 2 if first_draw % 2 == 1 else (step + 1) // 2
            if rng.random() < 1 - math.exp(-t * P_JP * coverage_factor):
                if pending_day is not None and day == pending_day and draw_id % 2 == 0:
                    found = ("won", step)
                    break
                j, debt, pending_day = J_MIN, debt + J_MIN, None
                continue
            if pending_day is not None and day == pending_day and draw_id % 2 == 0:
                found = ("share-out", step)
                break
            if pending_day is None and j > SHARE_OUT_AT:
                pending_day = day + 1
        if found:
            results[(found[0], found[1])] += 1
    return results, first_draw


def draw_time(draw_id, rows):
    first = rows[0]
    day = dt.date.fromisoformat(first["date"]) + dt.timedelta(days=(draw_id - 1) // 2)
    return day, "13:00" if draw_id % 2 == 1 else "21:00"


def run_tests(rows, sims, rng):
    use_535_constants()
    res = [r["result"] for r in rows]
    n = len(res)
    p = K / N
    tests = []
    counts = Counter(x for d in res for x in d)
    e = n * p
    stat = sum((counts[i] - e) ** 2 / e for i in range(1, N + 1)) * (N - 1) / (N - K)
    tests.append(("C1 each main number equally likely", f"chi2={stat:.1f}, df={N - 1}", s.chi2_sf(stat, N - 1)))
    sp_counts = Counter(r["special"] for r in rows)
    es = n / S
    stat_s = sum((sp_counts[i] - es) ** 2 / es for i in range(1, S + 1))
    tests.append(("C2 each special number equally likely", f"chi2={stat_s:.1f}, df={S - 1}", s.chi2_sf(stat_s, S - 1)))
    zs = {i: (counts[i] - e) / math.sqrt(n * p * (1 - p)) for i in range(1, N + 1)}
    zi = max(zs, key=lambda i: abs(zs[i]))
    tests.append(("C3 most extreme main number", f"#{zi} z={zs[zi]:+.2f}", min(1.0, 1 - (1 - 2 * s.norm_sf(abs(zs[zi]))) ** N)))
    halves = [[r["result"] for r in rows if r["id"] % 2 == h] for h in (1, 0)]
    stat_h = 0.0
    for i in range(1, N + 1):
        a, b_ = sum(i in d for d in halves[0]), sum(i in d for d in halves[1])
        na, nb = len(halves[0]), len(halves[1])
        pooled = (a + b_) / (na + nb)
        var = pooled * (1 - pooled) * (1 / na + 1 / nb)
        stat_h += ((a / na - b_ / nb) ** 2 / var) if var > 0 else 0
    stat_h *= (N - 1) / N
    tests.append(("C4 13:00 and 21:00 draws behave the same", f"chi2={stat_h:.1f}, df={N - 1}", s.chi2_sf(stat_h, N - 1)))
    tbar = (n - 1) / 2
    stt = sum((t - tbar) ** 2 for t in range(n))
    trend = sum((sum(t - tbar for t, d in enumerate(res) if i in d) / math.sqrt(p * (1 - p) * stt)) ** 2
                for i in range(1, N + 1)) * (N - 1) / N
    tests.append(("C5 no number drifting up/down over time", f"chi2={trend:.1f}, df={N - 1}", s.chi2_sf(trend, N - 1)))
    var_overlap = K * p * (1 - p) * (N - K) / (N - 1)
    lag_z = []
    for lag in range(1, 11):
        ov = [len(set(res[t]) & set(res[t + lag])) for t in range(n - lag)]
        lag_z.append((sum(ov) / len(ov) - K * p) / math.sqrt(var_overlap / len(ov)))
    tests.append(("C6 no memory: overlap with draws 1..10 back", f"chi2={sum(z * z for z in lag_z):.1f}, df=10",
                  s.chi2_sf(sum(z * z for z in lag_z), 10)))
    rep = sum(rows[t]["special"] == rows[t + 1]["special"] for t in range(n - 1))
    zr = (rep - (n - 1) / S) / math.sqrt((n - 1) / S * (1 - 1 / S))
    tests.append(("C7 special number does not repeat more than chance", f"{rep} vs {(n - 1) / S:.1f}, z={zr:+.2f}",
                  2 * s.norm_sf(abs(zr))))
    stat_s2, df_s2 = s.binned_chi2([sum(d) for d in res], s.sum_distribution())
    tests.append(("C8 sum of the 5 main numbers", f"chi2={stat_s2:.1f}, df={df_s2}, mean={sum(map(sum, res)) / n:.1f}"
                  f" (exp {K * (N + 1) / 2:.1f})", s.chi2_sf(stat_s2, df_s2)))
    odd_pmf = {k: math.comb(18, k) * math.comb(17, K - k) / MAIN_TOTAL for k in range(K + 1)}
    so, do = s.binned_chi2([sum(x % 2 for x in d) for d in res], odd_pmf)
    tests.append(("C9 odd/even split", f"chi2={so:.1f}, df={do}", s.chi2_sf(so, do)))
    low_pmf = {k: math.comb(17, k) * math.comb(18, K - k) / MAIN_TOTAL for k in range(K + 1)}
    sl, dl = s.binned_chi2([sum(x <= 17 for x in d) for d in res], low_pmf)
    tests.append(("C10 low (1-17) / high split", f"chi2={sl:.1f}, df={dl}", s.chi2_sf(sl, dl)))
    adj_pmf = {m: math.comb(K - 1, m) * math.comb(N - K + 1, K - m) / MAIN_TOTAL for m in range(K)}
    sa, da = s.binned_chi2([s.adjacent_pairs(d) for d in res], adj_pmf)
    tests.append(("C11 consecutive numbers", f"chi2={sa:.1f}, df={da}", s.chi2_sf(sa, da)))
    obs_pt, obs_gr = s.pair_triple_stats(res), s.gap_and_runs_stats(res)
    sim_pt, sim_gr = [], []
    for _ in range(sims):
        fake = s.fair_lottery(n, rng)
        sim_pt.append(s.pair_triple_stats(fake))
        sim_gr.append(s.gap_and_runs_stats(fake))
    tests.append(("C12 gaps between appearances are geometric", f"chi2={obs_gr['gap_chi']:.1f} (MC)",
                  s.mc_p_upper(obs_gr["gap_chi"], [x["gap_chi"] for x in sim_gr])))
    tests.append(("C13 runs of appear/miss per number", f"sum z2={obs_gr['runs_z2']:.1f} (MC)",
                  s.mc_p_upper(obs_gr["runs_z2"], [x["runs_z2"] for x in sim_gr])))
    tests.append(("C14 pairs appear together evenly", f"chi2={obs_pt['pair_chi']:.0f} (MC)",
                  s.mc_p_upper(obs_pt["pair_chi"], [x["pair_chi"] for x in sim_pt])))
    tests.append(("C15 most frequent pair", f"{obs_pt['pair_top'][0][0]} x{obs_pt['pair_max']} (MC)",
                  s.mc_p_upper(obs_pt["pair_max"], [x["pair_max"] for x in sim_pt])))
    tests.append(("C16 most frequent triple", f"{obs_pt['triple_top'][0][0]} x{obs_pt['triple_max']} (MC)",
                  s.mc_p_upper(obs_pt["triple_max"], [x["triple_max"] for x in sim_pt])))
    adj = s.holm([t[2] for t in tests])
    hb = s.hierarchical_beta_binomial([counts[i] for i in range(1, N + 1)], n)
    return [(a, b_, c, d) for (a, b_, c), d in zip(tests, adj)], hb


def with_specials(mains):
    return [(sorted(m), 1 + j % S) for j, m in enumerate(mains)]


def exact_money(tickets, prizes):
    inc = [0] * (N + 2)
    for j, (m, _) in enumerate(tickets):
        for x in m:
            inc[x] += 1 << (4 * j)
    cnt = Counter()
    for a in range(1, N - 3):
        sa = inc[a]
        for b in range(a + 1, N - 2):
            sb = sa + inc[b]
            for c in range(b + 1, N - 1):
                sc = sb + inc[c]
                for d in range(c + 1, N):
                    cnt.update(map((sc + inc[d]).__add__, inc[d + 1:N + 1]))
    assert sum(cnt.values()) == MAIN_TOTAL
    money = Counter()
    real = 0
    for key, w in cnt.items():
        hits = [(key >> (4 * j)) & 15 for j in range(len(tickets))]
        real += w * S * (max(hits) >= 3)
        for special in range(1, S + 1):
            total = 0
            for h, (_, sp) in zip(hits, tickets):
                tier = tier_of(h, sp == special)
                total += prizes.get(tier, 0) if tier else 0
            money[total] += w
    return money, real / TOTAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        print(f"data: {update.refresh('535')}")
    rng = random.Random(args.seed)
    rows, gaps = load_draws()
    report = {}
    print(f"DATA: {len(rows)} Lotto 5/35 draws, #{rows[0]['id']} ({rows[0]['date']}) .. #{rows[-1]['id']}"
          f" ({rows[-1]['date']}); missing ids: {len(gaps)}; two draws a day, 13:00 and 21:00")

    print(f"\nPART A - EXACT ODDS FOR ONE 10,000 VND TICKET ({MAIN_TOTAL:,} main-number results x {S} specials = {TOTAL:,})")
    labels = {"jackpot": "jackpot: 5 + special", "first": "first: 5", "second": "second: 4 + special",
              "third": "third: 4", "fourth": "fourth: 3 + special", "fifth": "fifth: 3",
              "consolation": "consolation: 0-2 + special"}
    print(f"  {labels['jackpot']:28s} 1 in {TOTAL:>12,.0f}   jackpot, from 6 ty")
    for t in TIERS:
        print(f"  {labels[t]:28s} 1 in {1 / P_TIER[t]:>12,.1f}   pays {BASE[t]:>10,} VND")
    p_real = sum(P_TIER[t] for t in TIERS if t != "consolation") + P_JP
    print(f"  a real prize (3+ main numbers): {p_real*100:.3f}% (1 in {1/p_real:.1f}); the consolation only refunds the"
          f" ticket; fixed prizes return {FIXED_EV:,.1f} VND per ticket ({FIXED_EV/100:.2f}%), the jackpot gets"
          f" {ACCUM*100:.2f}% of sales once the guarantee fund is repaid")
    report["odds"] = {"p_real": p_real, "fixed_ev": FIXED_EV, "accum": ACCUM}

    print(f"\nPART B - RANDOMNESS TESTS (Holm-corrected, {args.sims} fake lotteries for the MC tests)")
    tests, hb = run_tests(rows, args.sims, rng)
    for name, detail, pv, pa in tests:
        print(f"  {name:52s} {detail:40s} p={s.fmt_p(pv):>6s}  Holm p={s.fmt_p(pa):>6s}" + ("  <-- signal" if pa < 0.05 else ""))
    print(f"  Bayes factor fair : biased = {hb['bf_fair_vs_biased']:.1f} : 1; 95% upper bound on the spread of true"
          f" per-number chances: SD {hb['sd_95']:.4f} vs p {K/N:.4f} (+-{hb['sd_95']/(K/N)*100:.1f}% relative)")
    report["tests"] = [{"name": a, "detail": d, "p": pv, "holm_p": pa} for a, d, pv, pa in tests]

    sales = reconstruct_sales(rows)
    r5 = [r["t_fifth"] / r["tickets"] for r in sales if r["tickets"] > 0]
    rk = [r["t_refund"] / r["tickets"] for r in sales if r["tickets"] > 0]
    jp_w = sum(r["jackpot_winners"] for r in sales)
    jp_e = sum(max(r["tickets"], 0) * P_JP for r in sales)
    cov = jp_w / jp_e
    print("\nPART C - SALES REBUILT FROM JACKPOT GROWTH (guarantee fund 30% of sales until each 6 ty advance is repaid)")
    print(f"  check: tickets implied by 3-correct winners / by jackpot growth: median {statistics.median(r5):.3f};"
          f" by refund winners: median {statistics.median(rk):.3f}")
    ordinary = [r["tickets"] for r in sales[-200:] if not r["split"] and r["tickets"] > 0]
    print(f"  ordinary draws sell about {statistics.median(ordinary):,.0f} tickets; jackpot winners {jp_w} vs {jp_e:.1f}"
          f" expected if players picked at random ({cov:.2f})")

    ids, pending = designated_share_outs(rows)
    by_id = {r["id"]: r for r in sales}
    print(f"\nPART D - JACKPOT SHARE-OUT DRAWS: when the jackpot passes 12 ty unwon, the last draw of the next day shares it"
          f" out to the 3, 4 and 5-correct winners unless someone wins it first ({len(ids)} designated so far)")
    print(f"  {'draw':>5s} {'date':10s} {'jackpot':>8s} {'tickets':>10s} {'x usual':>7s} {'3 correct paid':>15s}"
          f" {'4 correct paid':>15s} {'back per ticket':>16s} {'after tax':>10s}")
    usual = statistics.median(ordinary)
    rows_out = []
    for i in ids:
        r = by_id[i]
        g, a = paid(r, False) / r["tickets"], paid(r, True) / r["tickets"]
        note = "" if r["split"] else "  jackpot won instead"
        print(f"  {i:>5d} {r['date']:10s} {r['jackpot_prize']/1e9:6.2f}ty {r['tickets']:>10,.0f} {r['tickets']/usual:>7.1f}"
              f" {r['fifth_prize']:>15,} {r['third_prize']:>15,} {g/1e4:>15.2f}x {a/1e4:>9.2f}x{note}")
        rows_out.append({"id": i, "date": r["date"], "jackpot": r["jackpot_prize"], "tickets": r["tickets"],
                         "gross": g, "after_tax": a, "split": r["split"]})
    tg = sum(paid(by_id[i], False) for i in ids)
    ta = sum(paid(by_id[i], True) for i in ids)
    tt = sum(by_id[i]["tickets"] for i in ids)
    last6 = ids[-6:]
    a6 = sum(paid(by_id[i], True) for i in last6) / sum(by_id[i]["tickets"] for i in last6)
    tax_share = (ta / tt - FIXED_EV) / (tg / tt - FIXED_EV)
    ordinary_ev = statistics.fmean(normal_value(r["jackpot_prize"], r["tickets"], cov) for r in sales
                                   if r["id"] not in set(ids) and r["tickets"] > 0)
    won_there = sum(by_id[i]["jackpot_winners"] for i in ids)
    exposure = sum(by_id[i]["tickets"] * P_JP for i in ids)
    c_share = won_there / exposure
    print(f"  all {len(ids)}: {tg/tt/1e4:.2f}x the price before tax, {ta/tt/1e4:.2f}x after tax; the last 6: {a6/1e4:.2f}x"
          f" after tax. An ordinary draw returns {ordinary_ev/1e4:.2f}x on average")
    print(f"  the jackpot was won first on {sum(not by_id[i]['split'] for i in ids)} of {len(ids)} designated draws"
          f" ({won_there} winners vs {exposure:.1f} expected if every ticket were different: the crowd's tickets overlap,"
          f" x{c_share:.2f})")
    crowd = share_out_crowd(sales, ids)
    b = crowd[0]
    print(f"  the crowd: log(tickets) = {b[0]:.2f} + {b[1]:.3f} x (draws since #{crowd[2]})/100 + {b[2]:.2f} x log(jackpot"
          f" before / 20 ty), residual SD {crowd[1]:.3f}: x{math.exp(b[1]):.2f} per 100 draws, and in step with the jackpot")
    report["share_outs"] = {"draws": rows_out, "gross": tg / tt, "after_tax": ta / tt, "last6_after_tax": a6,
                            "tax_share": tax_share, "c_share": c_share, "ordinary_ev": ordinary_ev,
                            "crowd": {"b": b, "sd": crowd[1], "ref": crowd[2]}}

    last = sales[-1]
    nxt = last["id"] + 1
    day, hour = draw_time(nxt, rows)
    reset = last["split"] or last["jackpot_winners"] > 0
    jackpot_now = J_MIN if reset else last["jackpot_prize"]
    fc_t = statistics.median([r["tickets"] for r in sales[-60:] if not r["split"] and r["id"] % 2 == nxt % 2])
    fc_j = jackpot_now + (ACCUM - (GUARANTEE if last["debt_after"] > 0 or reset else 0)) * fc_t * TICKET
    ev_normal = normal_value(fc_j, fc_t, cov)
    print(f"\nPART E - NEXT DRAW #{nxt} ({day} {hour}) AND THE NEXT SHARE-OUT")
    print(f"  jackpot now {jackpot_now/1e9:.2f} ty (it resets after a win or a share-out); next draw ≈ {fc_j/1e9:.2f} ty"
          f" with about {fc_t:,.0f} tickets; a ticket is worth about {ev_normal:,.0f} VND ({ev_normal/1e4:.2f}x the price)")
    if pending is not None:
        print(f"  a share-out draw is already scheduled: {pending}")
    sim, first = next_share_out(sales, cov, random.Random(args.seed))
    share = sorted((step, n) for (kind, step), n in sim.items() if kind == "share-out")
    j_prev = statistics.median(by_id[i - 1]["jackpot_prize"] for i in ids)
    v = None
    if share:
        steps = []
        for step, n in share:
            steps += [step] * n
        med = steps[len(steps) // 2]
        p14 = sum(n for step, n in share if step < 28) / 4000
        md, mh = draw_time(first + med, rows)
        print(f"  simulating the jackpot forward 4,000 times: the next share-out draw most likely comes around {md} {mh}"
              f" (draw #{first + med}); chance it comes within 14 days: {p14*100:.0f}%")
        v = share_out_value(j_prev, first + med, crowd, tax_share, c_share)
        print(f"  if the crowd keeps growing, it sells about {v['tickets']:,.0f} tickets and the jackpot reaches"
              f" {v['jackpot']/1e9:.1f} ty; the average ticket is worth {v['ev']:,.0f} VND after tax ({v['ev']/1e4:.2f}x;"
              f" range {v['ev_low']/1e4:.2f}x..{v['ev_high']/1e4:.2f}x); chance the jackpot is not won first, so the"
              f" share-out really happens: {v['p_share']*100:.0f}%")
        report["next_share_out"] = {"draw": first + med, "date": str(md), "time": mh, "p14": p14, "j_prev": j_prev, **v}
    report["next"] = {"draw": nxt, "date": str(day), "time": hour, "jackpot": fc_j, "tickets": fc_t, "ev": ev_normal}

    print("\nPART F - TICKETS (most chance to win: every main number used, least overlap, a different special on each)")
    designs = {}
    for k in (5, 10):
        mains = coverage.design(k, N, args.seed, size=K)
        tickets = with_specials(mains)
        designs[k] = tickets
        cost = k * TICKET
        normal = money_table(tickets, dict(BASE, jackpot=fc_j))
        print(f"  {k:>2d} tickets ({cost:,} VND): real prize (3+ main numbers) {normal['p_real']*100:.2f}%; anything"
              f" including refunds {normal['any']*100:.1f}%")
        print(f"      ordinary draw: end up ahead {normal['ahead'](cost)*100:.2f}%, average back {normal['mean']:,.0f} VND")
        if v:
            shared = money_table(tickets, dict(estimated_share_out_prizes(v["jackpot"], v["tickets"]), jackpot=v["jackpot"]))
            won = money_table(tickets, dict(BASE, jackpot=v["jackpot"] / 2))
            ahead = v["p_share"] * shared["ahead"](cost) + (1 - v["p_share"]) * won["ahead"](cost)
            print(f"      next share-out draw: end up ahead {ahead*100:.2f}% ({shared['ahead'](cost)*100:.1f}% if the"
                  f" share-out happens), average back ≈ {k * v['ev']:,.0f} VND")
        print("      " + "   ".join(f"{' '.join(f'{x:02d}' for x in m)} | {sp:02d}" for m, sp in tickets))
    report["tickets"] = {str(k): [[m, sp] for m, sp in t] for k, t in designs.items()}

    print("\nPART G - LONG-RUN, AS IT HAPPENED: 10 tickets on every draw vs only on the designated share-out draws (after tax)")
    tickets = designs[10]
    base_table = money_table(tickets, dict(BASE, jackpot=0))
    per_draw = {}
    for i in ids:
        r = by_id[i]
        prizes = {t: r[f"{t}_prize"] for t in TIERS}
        prizes["jackpot"] = r["jackpot_prize"] / (r["jackpot_winners"] + 1)
        per_draw[i] = money_table(tickets, prizes)
    prng = random.Random(args.seed)
    for label, chosen in (("every draw", [r["id"] for r in sales]), ("share-out draws only", ids)):
        spent = len(chosen) * 10 * TICKET
        average = sum(10 * paid(by_id[i], True) / by_id[i]["tickets"] for i in chosen) - spent
        results = []
        for _ in range(4000 if len(chosen) > 100 else 20_000):
            won = sum(sample(per_draw.get(i, base_table), prng) for i in chosen)
            results.append(won - spent)
        results.sort()
        print(f"  {label:22s} {len(chosen):>4d} draws, {spent/1e6:,.1f} million VND spent: the average ticket's result"
              f" {average/1e6:+,.2f} million; typical player {results[len(results)//2]/1e6:+,.2f} million;"
              f" ended ahead {sum(x >= 0 for x in results)/len(results)*100:.1f}%")
    (RESULTS / "lotto535_latest.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nsaved {RESULTS / 'lotto535_latest.json'}")


if __name__ == "__main__":
    main()


def forecast_next(seed=2026, runs=1000):
    rows, _ = load_draws()
    sales = reconstruct_sales(rows)
    ids, pending = designated_share_outs(rows)
    by_id = {r["id"]: r for r in sales}
    jp_w = sum(r["jackpot_winners"] for r in sales)
    cov = jp_w / sum(max(r["tickets"], 0) * P_JP for r in sales)
    tg = sum(paid(by_id[i], False) for i in ids) / sum(by_id[i]["tickets"] for i in ids)
    ta = sum(paid(by_id[i], True) for i in ids) / sum(by_id[i]["tickets"] for i in ids)
    tax_share = (ta - FIXED_EV) / (tg - FIXED_EV)
    c_share = sum(by_id[i]["jackpot_winners"] for i in ids) / sum(by_id[i]["tickets"] * P_JP for i in ids)
    crowd = share_out_crowd(sales, ids)
    last = sales[-1]
    nxt = last["id"] + 1
    day, hour = draw_time(nxt, rows)
    reset = last["split"] or last["jackpot_winners"] > 0
    jackpot_now = J_MIN if reset else last["jackpot_prize"]
    fc_t = statistics.median([r["tickets"] for r in sales[-60:] if not r["split"] and r["id"] % 2 == nxt % 2])
    fc_j = jackpot_now + (ACCUM - (GUARANTEE if last["debt_after"] > 0 or reset else 0)) * fc_t * TICKET
    scheduled = pending if isinstance(pending, int) else None
    if isinstance(pending, tuple):
        scheduled = next((i for i in (nxt, nxt + 1, nxt + 2) if str(draw_time(i, rows)[0]) == pending[1]
                          and draw_time(i, rows)[1] == "21:00"), None)
    out = {"draw": nxt, "date": str(day), "time": hour, "n_draws": len(rows), "jackpot_now": jackpot_now,
           "jackpot": fc_j, "tickets": fc_t, "coverage": cov, "tax_share": tax_share, "c_share": c_share,
           "share_out_now": scheduled == nxt, "scheduled": scheduled, "history_after_tax": ta,
           "last6_after_tax": sum(paid(by_id[i], True) for i in ids[-6:]) / sum(by_id[i]["tickets"] for i in ids[-6:]),
           "n_share_outs": len(ids)}
    if scheduled == nxt:
        v = share_out_value(jackpot_now, nxt, crowd, tax_share, c_share)
        out.update(share_out=v, ev=v["ev"], share_date=str(day), share_time=hour, share_draw=nxt, p14=1.0)
        return out
    out["ev"] = normal_value(fc_j, fc_t, cov)
    if scheduled:
        sd, sh = draw_time(scheduled, rows)
        v = share_out_value(max(jackpot_now, SHARE_OUT_AT), scheduled, crowd, tax_share, c_share)
        out.update(share_out=v, share_date=str(sd), share_time=sh, share_draw=scheduled, p14=1.0)
        return out
    sim, first = next_share_out(sales, cov, random.Random(seed), runs=runs)
    steps = []
    for (kind, step), n in sorted(sim.items(), key=lambda kv: kv[0][1]):
        if kind == "share-out":
            steps += [step] * n
    if steps:
        med = steps[len(steps) // 2]
        sd, sh = draw_time(first + med, rows)
        j_prev = statistics.median(by_id[i - 1]["jackpot_prize"] for i in ids)
        out.update(share_out=share_out_value(j_prev, first + med, crowd, tax_share, c_share), share_date=str(sd),
                   share_time=sh, share_draw=first + med, p14=sum(1 for x in steps if x < 28) / runs)
    return out
