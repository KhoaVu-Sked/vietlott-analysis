"""Mega 6/45 study: randomness tests, prediction models, 10-run walk-forward comparison, payout analysis.

Usage: python3 lottery/power645_study.py [--sims 400] [--runs 10] [--seed 2026]
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DATA = ROOT / "data" / "mega645.jsonl"
RESULTS = ROOT / "results"

N_BALLS, K = 45, 6
P = K / N_BALLS
C_TOTAL = math.comb(N_BALLS, K)
P_MATCH = [math.comb(K, k) * math.comb(N_BALLS - K, K - k) / C_TOTAL for k in range(K + 1)]
TICKET = 10_000
FIXED_PRIZE = {3: 30_000, 4: 300_000, 5: 10_000_000}
JACKPOT_MIN = 12_000_000_000
PRACTICAL_GAIN = 1e-4
FIXED_EV_RATE = sum(P_MATCH[k] * FIXED_PRIZE[k] for k in (3, 4, 5)) / TICKET
ACCUM_RATE = 0.55 - FIXED_EV_RATE
ACCUM_RATE_REPAYING = ACCUM_RATE - 0.20


def load_draws():
    by_id = {}
    for line in DATA.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            i = int(r["id"])
            by_id[i] = {"id": i, "date": r["date"], "result": sorted(r["result"]),
                        **{k: v for k, v in r.items() if k not in ("id", "date", "result")}}
    ids = sorted(by_id)
    gaps = [i for i in range(ids[0], ids[-1] + 1) if i not in by_id]
    return [by_id[i] for i in ids], gaps, []


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def _gammainc_lower_reg(a, x):
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    return max(0.0, 1 - _gammainc_lower_reg(df / 2, x / 2))


def mc_p_upper(observed, simulated):
    return (1 + sum(s >= observed for s in simulated)) / (1 + len(simulated))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def fair_lottery(n, rng):
    return [sorted(rng.sample(range(1, N_BALLS + 1), K)) for _ in range(n)]


def sum_distribution():
    ways = [[0] * (K * N_BALLS + 1) for _ in range(K + 1)]
    ways[0][0] = 1
    for x in range(1, N_BALLS + 1):
        for k in range(K, 0, -1):
            row, prev = ways[k], ways[k - 1]
            for s in range(len(row) - 1, x - 1, -1):
                if prev[s - x]:
                    row[s] += prev[s - x]
    return {s: c / C_TOTAL for s, c in enumerate(ways[K]) if c}


def binned_chi2(observed_values, pmf, min_expected=5.0):
    n = len(observed_values)
    counts = Counter(observed_values)
    keys = sorted(pmf)
    bins, cur_e, cur_o = [], 0.0, 0
    for k in keys:
        cur_e += n * pmf[k]
        cur_o += counts.get(k, 0)
        if cur_e >= min_expected:
            bins.append((cur_o, cur_e))
            cur_e, cur_o = 0.0, 0
    if cur_e > 0:
        o, e = bins.pop()
        bins.append((o + cur_o, e + cur_e))
    stat = sum((o - e) ** 2 / e for o, e in bins)
    return stat, len(bins) - 1


def pair_triple_stats(draws):
    pairs, triples = Counter(), Counter()
    for d in draws:
        pairs.update(combinations(d, 2))
        triples.update(combinations(d, 3))
    n = len(draws)
    e2 = n * math.comb(K, 2) / math.comb(N_BALLS, 2)
    e3 = n * math.comb(K, 3) / math.comb(N_BALLS, 3)
    n2, n3 = math.comb(N_BALLS, 2), math.comb(N_BALLS, 3)
    chi_pairs = sum((c - e2) ** 2 for c in pairs.values()) / e2 + (n2 - len(pairs)) * e2
    chi_triples = sum((c - e3) ** 2 for c in triples.values()) / e3 + (n3 - len(triples)) * e3
    return {"pair_chi": chi_pairs, "pair_max": max(pairs.values()),
            "triple_chi": chi_triples, "triple_max": max(triples.values()),
            "pair_top": pairs.most_common(3), "triple_top": triples.most_common(3)}


def gap_and_runs_stats(draws):
    n = len(draws)
    seqs = [[0] * n for _ in range(N_BALLS)]
    for t, d in enumerate(draws):
        for x in d:
            seqs[x - 1][t] = 1
    gaps = Counter()
    runs_z2 = 0.0
    for s in seqs:
        last = None
        for t, v in enumerate(s):
            if v:
                if last is not None:
                    gaps[min(t - last, 26)] += 1
                last = t
        n1 = sum(s)
        n0 = n - n1
        runs = 1 + sum(1 for t in range(1, n) if s[t] != s[t - 1])
        mu = 1 + 2 * n1 * n0 / n
        var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
        runs_z2 += (runs - mu) ** 2 / var
    total = sum(gaps.values())
    gap_chi = 0.0
    for g in range(1, 27):
        pg = P * (1 - P) ** (g - 1) if g < 26 else (1 - P) ** 25
        e = total * pg
        gap_chi += (gaps.get(g, 0) - e) ** 2 / e
    return {"gap_chi": gap_chi, "runs_z2": runs_z2}


def run_tests(draw_rows, sims, rng):
    draws = [d["result"] for d in draw_rows]
    n = len(draws)
    tests = []

    counts = Counter(x for d in draws for x in d)
    e = n * P
    pearson = sum((counts[i] - e) ** 2 / e for i in range(1, N_BALLS + 1))
    stat = pearson * 44 / 39
    tests.append(("A1 each number equally likely (frequency)", f"chi2={stat:.1f}, df=44", chi2_sf(stat, 44)))

    zs = {i: (counts[i] - e) / math.sqrt(n * P * (1 - P)) for i in range(1, N_BALLS + 1)}
    zmax_i = max(zs, key=lambda i: abs(zs[i]))
    p_single = 2 * norm_sf(abs(zs[zmax_i]))
    tests.append(("A2 most extreme single number", f"#{zmax_i} z={zs[zmax_i]:+.2f}",
                  min(1.0, 1 - (1 - p_single) ** N_BALLS)))

    years = defaultdict(list)
    for row in draw_rows:
        years[int(row["date"][:4])].append(row["result"])
    stat_y = 0.0
    for yd in years.values():
        cy = Counter(x for d in yd for x in d)
        ey = len(yd) * P
        stat_y += sum((cy[i] - ey) ** 2 for i in range(1, N_BALLS + 1)) / (ey * (1 - P) * 45 / 44)
    df_y = 44 * len(years)
    tests.append((f"A3 frequencies stable in every year ({len(years)} years)", f"chi2={stat_y:.1f}, df={df_y}",
                  chi2_sf(stat_y, df_y)))

    tbar = (n - 1) / 2
    stt = sum((t - tbar) ** 2 for t in range(n))
    trend_z2 = 0.0
    for i in range(1, N_BALLS + 1):
        s = sum((t - tbar) for t, d in enumerate(draws) if i in d)
        trend_z2 += (s / math.sqrt(P * (1 - P) * stt)) ** 2
    stat_tr = trend_z2 * 44 / 45
    tests.append(("A4 no number drifting up/down over time", f"chi2={stat_tr:.1f}, df=44", chi2_sf(stat_tr, 44)))

    var_overlap = K * P * (1 - P) * (N_BALLS - K) / (N_BALLS - 1)
    lag_z = []
    for lag in range(1, 11):
        ov = [len(set(draws[t]) & set(draws[t + lag])) for t in range(n - lag)]
        lag_z.append((sum(ov) / len(ov) - K * P) / math.sqrt(var_overlap / len(ov)))
    stat_lag = sum(z * z for z in lag_z)
    tests.append(("A5 no memory: overlap with draws 1..10 back", f"chi2={stat_lag:.1f}, df=10, lag1 z={lag_z[0]:+.2f}",
                  chi2_sf(stat_lag, 10)))

    sums = [sum(d) for d in draws]
    sum_pmf = sum_distribution()
    stat_s, df_s = binned_chi2(sums, sum_pmf)
    tests.append(("A6 sum of the 6 numbers", f"chi2={stat_s:.1f}, df={df_s}, mean={sum(sums)/n:.1f} (exp 138.0)",
                  chi2_sf(stat_s, df_s)))

    odd = [sum(x % 2 for x in d) for d in draws]
    odd_pmf = {k: math.comb(23, k) * math.comb(22, K - k) / C_TOTAL for k in range(K + 1)}
    stat_o, df_o = binned_chi2(odd, odd_pmf)
    tests.append(("A7 odd/even split", f"chi2={stat_o:.1f}, df={df_o}", chi2_sf(stat_o, df_o)))

    low = [sum(x <= 22 for x in d) for d in draws]
    low_pmf = {k: math.comb(22, k) * math.comb(23, K - k) / C_TOTAL for k in range(K + 1)}
    stat_l, df_l = binned_chi2(low, low_pmf)
    tests.append(("A8 low (1-22) / high (23-45) split", f"chi2={stat_l:.1f}, df={df_l}", chi2_sf(stat_l, df_l)))

    adj = [sum(1 for a, b in zip(d, d[1:]) if b == a + 1) for d in draws]
    adj_pmf = {m: math.comb(K - 1, m) * math.comb(N_BALLS - K + 1, K - m) / C_TOTAL for m in range(K)}
    stat_a, df_a = binned_chi2(adj, adj_pmf)
    tests.append(("A9 consecutive numbers (e.g. 25-26)", f"chi2={stat_a:.1f}, df={df_a}", chi2_sf(stat_a, df_a)))

    obs_pt = pair_triple_stats(draws)
    obs_gr = gap_and_runs_stats(draws)
    sim_pt, sim_gr = [], []
    for _ in range(sims):
        fake = fair_lottery(n, rng)
        sim_pt.append(pair_triple_stats(fake))
        sim_gr.append(gap_and_runs_stats(fake))
    tests.append(("A10 gaps between appearances are geometric", f"chi2={obs_gr['gap_chi']:.1f} (MC)",
                  mc_p_upper(obs_gr["gap_chi"], [s["gap_chi"] for s in sim_gr])))
    tests.append(("A11 runs of appear/miss per number", f"sum z2={obs_gr['runs_z2']:.1f} (MC)",
                  mc_p_upper(obs_gr["runs_z2"], [s["runs_z2"] for s in sim_gr])))
    tests.append(("A12 pairs appear together evenly", f"chi2={obs_pt['pair_chi']:.0f} (MC)",
                  mc_p_upper(obs_pt["pair_chi"], [s["pair_chi"] for s in sim_pt])))
    tests.append(("A13 most frequent pair", f"{obs_pt['pair_top'][0][0]} x{obs_pt['pair_max']} (MC)",
                  mc_p_upper(obs_pt["pair_max"], [s["pair_max"] for s in sim_pt])))
    tests.append(("A14 triples appear together evenly", f"chi2={obs_pt['triple_chi']:.0f} (MC)",
                  mc_p_upper(obs_pt["triple_chi"], [s["triple_chi"] for s in sim_pt])))
    tests.append(("A15 most frequent triple", f"{obs_pt['triple_top'][0][0]} x{obs_pt['triple_max']} (MC)",
                  mc_p_upper(obs_pt["triple_max"], [s["triple_max"] for s in sim_pt])))

    adj_p = holm([t[2] for t in tests])
    sim_ref = {"pair_chi_mean": sum(s["pair_chi"] for s in sim_pt) / sims,
               "triple_chi_mean": sum(s["triple_chi"] for s in sim_pt) / sims,
               "triple_max_mean": sum(s["triple_max"] for s in sim_pt) / sims}
    return [(name, detail, p, pa) for (name, detail, p), pa in zip(tests, adj_p)], counts, sim_ref


def hierarchical_beta_binomial(counts, n):
    grid = [10 ** (-3.3 + i * 0.01) for i in range(231)]
    log_m0 = sum(c * math.log(P) + (n - c) * math.log(1 - P) for c in counts)
    post = []
    for sd in grid:
        kappa = P * (1 - P) / (sd * sd) - 1
        if kappa <= 0:
            continue
        a, b = P * kappa, (1 - P) * kappa
        lm = sum(math.lgamma(c + a) + math.lgamma(n - c + b) - math.lgamma(n + a + b)
                 - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b) for c in counts)
        post.append((sd, lm))
    lmax = max(lm for _, lm in post)
    w = [math.exp(lm - lmax) for _, lm in post]
    marg1 = math.log(sum(w) / len(w)) + lmax
    total = sum(w)
    cdf, acc = [], 0.0
    for (sd, _), wi in zip(post, w):
        acc += wi / total
        cdf.append((sd, acc))
    sd_median = next(sd for sd, c in cdf if c >= 0.5)
    sd_95 = next(sd for sd, c in cdf if c >= 0.95)
    return {"bf_fair_vs_biased": math.exp(log_m0 - marg1), "sd_median": sd_median, "sd_95": sd_95}


class History:
    def __init__(self, draws):
        self.draws = draws
        n = len(draws)
        self.cum = [[0] * (N_BALLS + 1)]
        self.last = [[-1] * (N_BALLS + 1)]
        for t, d in enumerate(draws):
            c = self.cum[-1][:]
            ls = self.last[-1][:]
            for x in d:
                c[x] += 1
                ls[x] = t
            self.cum.append(c)
            self.last.append(ls)
        self.n = n

    def window(self, s, i, w):
        lo = max(0, s - w)
        return (self.cum[s][i] - self.cum[lo][i]) / max(1, s - lo)

    def features(self, s, i):
        gap = s - self.last[s][i] if self.last[s][i] >= 0 else s + 1
        return [self.window(s, i, 10) - P, self.window(s, i, 50) - P, self.window(s, i, 200) - P,
                self.cum[s][i] / max(1, s) - P, math.log(gap), 1.0 if self.last[s][i] == s - 1 else 0.0]


def normalise(q):
    q = [max(v, 1e-6) for v in q]
    total = sum(q)
    q = [v * K / total for v in q]
    return [min(max(v, 1e-4), 0.999) for v in q]


def log_score(q, drawn):
    return sum(math.log(q[i - 1]) if i in drawn else math.log(1 - q[i - 1]) for i in range(1, N_BALLS + 1))


def fit_logistic(rows, ys, l2=1.0, iters=6):
    dim = len(rows[0])
    beta = [0.0] * dim
    beta[0] = math.log(P / (1 - P))
    for _ in range(iters):
        grad = [0.0] * dim
        hess = [[0.0] * dim for _ in range(dim)]
        for x, y in zip(rows, ys):
            eta = sum(b * v for b, v in zip(beta, x))
            mu = 1 / (1 + math.exp(-eta))
            r = y - mu
            w = mu * (1 - mu)
            for a in range(dim):
                grad[a] += r * x[a]
                wa = w * x[a]
                ha = hess[a]
                for b in range(a, dim):
                    ha[b] += wa * x[b]
        for a in range(1, dim):
            grad[a] -= l2 * beta[a]
            hess[a][a] += l2
        for a in range(dim):
            for b in range(a):
                hess[a][b] = hess[b][a]
        step = solve(hess, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                if f:
                    for k in range(c, n + 1):
                        m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


class Standardiser:
    def __init__(self, rows):
        dim = len(rows[0])
        self.mu = [sum(r[a] for r in rows) / len(rows) for a in range(dim)]
        self.sd = [math.sqrt(sum((r[a] - self.mu[a]) ** 2 for r in rows) / len(rows)) or 1.0 for a in range(dim)]

    def __call__(self, x):
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)]


def training_set(hist, t, span=400, start=50):
    rows, ys = [], []
    for s in range(max(start, t - span), t):
        drawn = set(hist.draws[s])
        for i in range(1, N_BALLS + 1):
            rows.append(hist.features(s, i))
            ys.append(1 if i in drawn else 0)
    return rows, ys


class LogisticModel:
    name = "logistic regression"

    def fit(self, hist, t):
        rows, ys = training_set(hist, t)
        self.std = Standardiser(rows)
        self.beta = fit_logistic([[1.0] + self.std(r) for r in rows], ys)

    def predict(self, hist, t):
        out = []
        for i in range(1, N_BALLS + 1):
            x = [1.0] + self.std(hist.features(t, i))
            out.append(1 / (1 + math.exp(-sum(b * v for b, v in zip(self.beta, x)))))
        return out


class NaiveBayesModel:
    name = "naive Bayes"

    def fit(self, hist, t):
        rows, ys = training_set(hist, t)
        self.prior1 = sum(ys) / len(ys)
        self.stats = {}
        for cls in (0, 1):
            sub = [r for r, y in zip(rows, ys) if y == cls]
            mus = [sum(r[a] for r in sub) / len(sub) for a in range(len(sub[0]))]
            sds = [math.sqrt(sum((r[a] - mus[a]) ** 2 for r in sub) / len(sub)) or 1e-3 for a in range(len(sub[0]))]
            last_rate = sum(r[5] for r in sub) / len(sub)
            self.stats[cls] = (mus, sds, last_rate)

    def predict(self, hist, t):
        out = []
        for i in range(1, N_BALLS + 1):
            x = hist.features(t, i)
            logp = {}
            for cls, prior in ((1, self.prior1), (0, 1 - self.prior1)):
                mus, sds, last_rate = self.stats[cls]
                lp = math.log(prior)
                for a in range(5):
                    lp += -0.5 * ((x[a] - mus[a]) / sds[a]) ** 2 - math.log(sds[a])
                lp += math.log(last_rate if x[5] else 1 - last_rate)
                logp[cls] = lp
            out.append(1 / (1 + math.exp(logp[0] - logp[1])))
        return out


def eb_kappa(counts, n):
    best, best_lm = None, -math.inf
    for e in range(20, 81):
        kappa = 10 ** (e / 10)
        a, b = P * kappa, (1 - P) * kappa
        lm = sum(math.lgamma(c + a) + math.lgamma(n - c + b) - math.lgamma(n + a + b)
                 - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b) for c in counts)
        if lm > best_lm:
            best, best_lm = kappa, lm
    return best


def simple_models(hist, t, state):
    cum = hist.cum[t]
    last = hist.last[t]
    q = {}
    q["uniform (fair lottery)"] = [P] * N_BALLS
    q["frequency / hot all-time (no pooling)"] = [cum[i] + 1 for i in range(1, N_BALLS + 1)]
    a = P * state["kappa"]
    b = (1 - P) * state["kappa"]
    q["hierarchical Beta-Binomial (partial pooling)"] = [(a + cum[i]) / (a + b + t) for i in range(1, N_BALLS + 1)]
    q["hot last 30 draws"] = [hist.window(t, i, 30) * 30 + 1 for i in range(1, N_BALLS + 1)]
    q["exponentially weighted hot"] = [v + 0.05 for v in state["ewma"][1:]]
    q["overdue / cold (gambler's fallacy)"] = [(t - last[i]) if last[i] >= 0 else t + 1 for i in range(1, N_BALLS + 1)]
    prev = hist.draws[t - 1]
    trans = state["trans"]
    nprev = state["nprev"]
    alpha = 10.0
    q["Markov: follows last draw"] = [
        sum((trans[j][i] + alpha * P) / (nprev[j] + alpha) for j in prev) / K for i in range(1, N_BALLS + 1)]
    return q


def update_state(state, hist, t):
    lam = 0.97
    drawn = set(hist.draws[t])
    state["ewma"] = [lam * v + (1 if i in drawn else 0) for i, v in enumerate(state["ewma"])]
    if t >= 1:
        for j in hist.draws[t - 1]:
            state["nprev"][j] += 1
            for i in hist.draws[t]:
                state["trans"][j][i] += 1


def top6(q, rng):
    order = sorted(range(1, N_BALLS + 1), key=lambda i: (-q[i - 1], rng.random()))
    return set(order[:K])


def ticket_return(hits, jackpot):
    if hits == 6:
        return jackpot
    return FIXED_PRIZE.get(hits, 0)


def walk_forward(draws, runs, seed, label):
    hist = History([d["result"] for d in draws])
    n = len(draws)
    first_test = n - runs * ((n - 266) // runs)
    block = (n - first_test) // runs
    state = {"ewma": [0.0] * (N_BALLS + 1), "trans": [[0] * (N_BALLS + 1) for _ in range(N_BALLS + 1)],
             "nprev": [0] * (N_BALLS + 1), "kappa": 1e6}
    for t in range(first_test):
        update_state(state, hist, t)
    learned = [LogisticModel(), NaiveBayesModel()]
    results = []
    rng = random.Random(seed)
    for r in range(runs):
        lo, hi = first_test + r * block, first_test + (r + 1) * block
        state["kappa"] = eb_kappa([hist.cum[lo][i] for i in range(1, N_BALLS + 1)], lo)
        per_model = defaultdict(lambda: {"ls": [], "hits": [], "money": []})
        for t in range(lo, hi):
            if (t - lo) % 26 == 0:
                for m in learned:
                    m.fit(hist, t)
            qs = simple_models(hist, t, state)
            for m in learned:
                qs[m.name] = m.predict(hist, t)
            drawn = set(draws[t]["result"])
            base = log_score([P] * N_BALLS, drawn)
            jackpot = draws[t].get("jackpot_value") or JACKPOT_MIN
            for name, qraw in qs.items():
                qn = normalise(qraw)
                pick = set(rng.sample(range(1, N_BALLS + 1), K)) if name.startswith("uniform") else top6(qn, rng)
                hits = len(pick & drawn)
                rec = per_model[name]
                rec["ls"].append(log_score(qn, drawn) - base)
                rec["hits"].append(hits)
                rec["money"].append(ticket_return(hits, jackpot) - TICKET)
            update_state(state, hist, t)
        summary = {}
        for name, rec in per_model.items():
            m = len(rec["ls"])
            mean = sum(rec["ls"]) / m
            sd = math.sqrt(sum((v - mean) ** 2 for v in rec["ls"]) / (m - 1)) if m > 1 else 0
            summary[name] = {"ls_diff": mean, "ls_se": sd / math.sqrt(m), "hits": sum(rec["hits"]) / m,
                             "prize_rate": sum(h >= 3 for h in rec["hits"]) / m,
                             "jackpots": sum(h == 6 for h in rec["hits"]),
                             "money": sum(rec["money"]), "n": m, "raw_ls": rec["ls"], "raw_hits": rec["hits"]}
        results.append({"run": r + 1, "draws": (draws[lo]["id"], draws[hi - 1]["id"]), "summary": summary})
    return {"label": label, "runs": results, "block": block, "first_test": first_test}


def beats_fair(s):
    return s["ls_diff"] - 2 * s["ls_se"] > 0 and s["ls_diff"] > PRACTICAL_GAIN


def pool_runs(wf):
    names = list(wf["runs"][0]["summary"])
    pooled = {}
    for name in names:
        ls = [v for r in wf["runs"] for v in r["summary"][name]["raw_ls"]]
        hits = [v for r in wf["runs"] for v in r["summary"][name]["raw_hits"]]
        m = len(ls)
        mean = sum(ls) / m
        sd = math.sqrt(sum((v - mean) ** 2 for v in ls) / (m - 1))
        hmean = sum(hits) / m
        hsd = math.sqrt(sum((h - hmean) ** 2 for h in hits) / (m - 1))
        pooled[name] = {
            "ls_diff": mean, "ls_se": sd / math.sqrt(m), "z": mean / (sd / math.sqrt(m)) if sd > 1e-9 else 0.0,
            "runs_won": sum(r["summary"][name]["ls_diff"] > 1e-12 for r in wf["runs"]),
            "runs_sig": sum(beats_fair(r["summary"][name]) for r in wf["runs"]),
            "hits": hmean, "hits_se": hsd / math.sqrt(m),
            "prize_rate": sum(h >= 3 for h in hits) / m,
            "money": sum(r["summary"][name]["money"] for r in wf["runs"]), "n": m}
    return pooled


def reconstruct_sales(draws):
    rows = []
    for d in sorted(draws, key=lambda d: d["id"]):
        if d.get("jackpot_value") is None or "third_winners" not in d:
            break
        if rows and d["id"] != rows[-1]["id"] + 1:
            break
        rows.append(d)
    if len(rows) < 50 or rows[0]["id"] != 1:
        return None
    out = []
    debt = float(JACKPOT_MIN)
    prev_j, prev_won = None, True
    for d in rows:
        fixed = (FIXED_PRIZE[5] * d["first_winners"] + FIXED_PRIZE[4] * d["second_winners"]
                 + FIXED_PRIZE[3] * d["third_winners"])
        if prev_won and prev_j is not None:
            debt += JACKPOT_MIN
        base = JACKPOT_MIN if prev_won else prev_j
        from_3match = d["third_winners"] / P_MATCH[3]
        pools = [d["jackpot_value"]]
        if d["jackpot_winners"] >= 2 and d.get("jackpot_prize"):
            pools.append(d["jackpot_prize"])
        pool = min(pools, key=lambda j: abs(_revenue(j - base, fixed, debt)[0] / TICKET - from_3match))
        revenue, debt = _revenue(pool - base, fixed, debt)
        out.append({**d, "pool": pool, "revenue": revenue, "tickets": revenue / TICKET, "fixed_paid": fixed,
                    "tickets_from_3match": from_3match, "debt_after": debt})
        prev_won = d["jackpot_winners"] > 0
        prev_j = pool
    return out


def _revenue(accum, fixed, debt):
    if debt > 0:
        revenue = (accum + fixed) / 0.35
        if 0.2 * revenue > debt:
            return (accum + fixed + debt) / 0.55, 0.0
        return revenue, debt - 0.2 * revenue
    return (accum + fixed) / 0.55, debt


def popularity_model(sales):
    rows = [s for s in sales if s["tickets"] > 0 and s["third_winners"] > 0]
    ys, xs = [], []
    for s in rows:
        ys.append(math.log(s["third_winners"] / (s["tickets"] * P_MATCH[3])))
        xs.append([1 if i in s["result"] else 0 for i in range(1, N_BALLS + 1)])
    dim = N_BALLS
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for x, y in zip(xs, ys):
        idx = [i for i, v in enumerate(x) if v]
        for a in idx:
            xty[a] += y
            for b in idx:
                xtx[a][b] += 1
    ridge = 1.0
    for a in range(dim):
        xtx[a][a] += ridge
    beta = solve(xtx, xty)
    mean_beta = sum(beta) / dim
    beta = [b - mean_beta for b in beta]
    resid = [y - sum(beta[i] for i, v in enumerate(x) if v) for x, y in zip(xs, ys)]
    birthday = [sum(1 for v in s["result"] if v <= 31) for s in rows]
    by_bday = defaultdict(list)
    for b, y in zip(birthday, ys):
        by_bday[b].append(y)
    return {"beta": beta, "resid_sd": math.sqrt(sum(r * r for r in resid) / len(resid)), "n": len(rows),
            "by_birthday_count": {k: (len(v), sum(v) / len(v)) for k, v in sorted(by_bday.items())}}


def jackpot_share_check(sales):
    expected_winners = sum(s["tickets"] * P_MATCH[6] for s in sales)
    expected_won_draws = sum(1 - math.exp(-s["tickets"] * P_MATCH[6]) for s in sales)
    return {"winners": sum(s["jackpot_winners"] for s in sales), "expected_winners": expected_winners,
            "won_draws": sum(s["jackpot_winners"] > 0 for s in sales), "expected_won_draws": expected_won_draws,
            "multi_winner_draws": sum(s["jackpot_winners"] > 1 for s in sales)}


def popularity_calibration(sales, beta, tier=5):
    rows = [(s["tickets"] * P_MATCH[tier], sum(beta[i - 1] for i in s["result"]), s[
        {5: "first_winners", 4: "second_winners"}[tier]]) for s in sales if s["tickets"] > 0]
    a = c = 0.0
    for _ in range(25):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for off, x, y in rows:
            mu = off * math.exp(a + c * x)
            g0 += y - mu
            g1 += (y - mu) * x
            h00 += mu
            h01 += mu * x
            h11 += mu * x * x
        det = h00 * h11 - h01 * h01
        da = (h11 * g0 - h01 * g1) / det
        dc = (h00 * g1 - h01 * g0) / det
        a += da
        c += dc
        if abs(da) + abs(dc) < 1e-10:
            break
    return {"a": a, "c": c}


def adjacent_pairs(d):
    return sum(1 for a, b in zip(d, d[1:]) if b == a + 1)


def pattern_factors(sales, beta, c):
    acc = defaultdict(lambda: [0.0, 0.0])
    for d in sales:
        g = min(adjacent_pairs(d["result"]), 3)
        acc[g][0] += d["first_winners"]
        acc[g][1] += d["tickets"] * P_MATCH[5] * math.exp(c * sum(beta[i - 1] for i in d["result"]))
    return {g: (o / e, math.sqrt(o) / e) for g, (o, e) in sorted(acc.items())}


def kelly_bankroll(ev_ratio, jackpot_net_share, price=TICKET):
    if ev_ratio <= 1:
        return None
    second_moment = P_MATCH[6] * (jackpot_net_share / price) ** 2
    fraction = (ev_ratio - 1) / second_moment
    return price / fraction


def expected_value(jackpot, tickets, popularity_mult=1.0, tax_rate=0.10, tax_free=10_000_000):
    lam = max(tickets - 1, 0) * P_MATCH[6] * popularity_mult
    share = (1 - math.exp(-lam)) / lam if lam > 0 else 1.0
    jackpot_net = jackpot - tax_rate * max(0, jackpot - tax_free)
    fixed = sum(P_MATCH[k] * FIXED_PRIZE[k] for k in (3, 4, 5))
    return {"ev": fixed + P_MATCH[6] * share * jackpot_net, "fixed": fixed, "share": share, "lambda": lam}


def advertised(rows, i):
    if i == 0 or rows[i - 1]["jackpot_winners"] > 0:
        return JACKPOT_MIN
    return rows[i - 1]["pool"]


def sales_model(rows, window=300):
    xs, ys, after_win = [], [], []
    for i in range(max(2, len(rows) - window), len(rows)):
        if rows[i - 1]["jackpot_winners"] > 0:
            after_win.append(rows[i]["tickets"] / rows[i - 1]["tickets"])
            continue
        xs.append(math.log(advertised(rows, i) / advertised(rows, i - 1)))
        ys.append(math.log(rows[i]["tickets"] / rows[i - 1]["tickets"]))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope, statistics.median(after_win)


def forecast_sales(rows):
    alpha, elasticity, reset_ratio = sales_model(rows)
    last = rows[-1]
    adv = advertised(rows, len(rows))
    if last["jackpot_winners"] > 0:
        tickets = last["tickets"] * reset_ratio
    else:
        tickets = last["tickets"] * math.exp(alpha + elasticity * math.log(adv / advertised(rows, len(rows) - 1)))
    repaying = last["jackpot_winners"] > 0 or last["debt_after"] > 0
    jackpot = adv + (ACCUM_RATE_REPAYING if repaying else ACCUM_RATE) * tickets * TICKET
    return {"advertised": adv, "tickets": tickets, "jackpot": jackpot, "elasticity": elasticity,
            "rule_1_2": last["tickets"] * 1.2}


def sales_curve(sales, recent=400):
    pts = [(math.log(s["jackpot_value"]), math.log(s["tickets"])) for s in sales[-recent:] if s["tickets"] > 0]
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return lambda j: math.exp(my + slope * (math.log(j) - mx)), slope


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_run_matrix(wf, pooled):
    names = sorted(pooled, key=lambda k: -pooled[k]["ls_diff"])
    runs = wf["runs"]
    print("  log-score gain vs fair lottery per run, in milli-nats per draw (* = more than 2 SE above 0):")
    print("  " + " " * 46 + "".join(f"{'run'+str(r['run']):>9s}" for r in runs))
    for name in names:
        cells = []
        for r in runs:
            s = r["summary"][name]
            star = "*" if beats_fair(s) else " "
            cells.append(f"{s['ls_diff']*1000:+8.1f}{star}")
        print(f"  {name:46s}" + "".join(cells))
    print("  hits per draw (top-6 ticket) per run:")
    for name in names:
        print(f"  {name:46s}" + "".join(f"{r['summary'][name]['hits']:9.3f}" for r in runs))


def next_draw_picks(draws, seed):
    hist = History([d["result"] for d in draws])
    n = len(draws)
    state = {"ewma": [0.0] * (N_BALLS + 1), "trans": [[0] * (N_BALLS + 1) for _ in range(N_BALLS + 1)],
             "nprev": [0] * (N_BALLS + 1)}
    for t in range(n):
        update_state(state, hist, t)
    state["kappa"] = eb_kappa([hist.cum[n][i] for i in range(1, N_BALLS + 1)], n)
    qs = simple_models(hist, n, state)
    for m in (LogisticModel(), NaiveBayesModel()):
        m.fit(hist, n)
        qs[m.name] = m.predict(hist, n)
    rng = random.Random(seed)
    picks = {}
    for name, q in qs.items():
        if name.startswith("uniform"):
            picks[name + " = random ticket"] = sorted(rng.sample(range(1, N_BALLS + 1), K))
            continue
        qn = normalise(q)
        label = name + (" (no real preference)" if max(qn) - min(qn) < 1e-3 else "")
        picks[label] = sorted(top6(qn, rng))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        print(f"data: {update.refresh('645')}")
    rng = random.Random(args.seed)
    RESULTS.mkdir(exist_ok=True)

    draws, missing_ids, conflicts = load_draws()
    n = len(draws)
    print(f"DATA: {n} draws, #{draws[0]['id']} ({draws[0]['date']}) .. #{draws[-1]['id']} ({draws[-1]['date']})"
          f"; missing ids: {len(missing_ids)}")
    report = {"n_draws": n, "first": draws[0]["id"], "last": draws[-1]["id"], "missing": missing_ids}

    print("\nPART A - PRE-REGISTERED RANDOMNESS TESTS (Holm-corrected for 15 tests)")
    tests, counts, sim_ref = run_tests(draws, args.sims, rng)
    for name, detail, p, pa in tests:
        flag = "  <-- signal" if pa < 0.05 else ""
        print(f"  {name:48s} {detail:42s} p={fmt_p(p):>6s}  Holm p={fmt_p(pa):>6s}{flag}")
    print(f"  (MC reference from {args.sims} fair lotteries of {n} draws: pair chi2 mean {sim_ref['pair_chi_mean']:.0f},"
          f" triple chi2 mean {sim_ref['triple_chi_mean']:.0f}, best triple count mean {sim_ref['triple_max_mean']:.1f})")
    report["tests"] = [{"name": a, "detail": b, "p": c, "holm_p": d} for a, b, c, d in tests]

    hb = hierarchical_beta_binomial([counts[i] for i in range(1, N_BALLS + 1)], n)
    sd_detect = 2 * math.sqrt(P * (1 - P) / n)
    print("\nPART A2 - BAYESIAN VIEW (hierarchical Beta-Binomial over the 45 numbers)")
    print(f"  Bayes factor fair : biased = {hb['bf_fair_vs_biased']:.1f} : 1")
    print(f"  spread of the TRUE per-number probabilities: posterior median SD {hb['sd_median']:.4f},"
          f" 95% upper bound {hb['sd_95']:.4f} (vs p = {P:.4f}; i.e. at most +-{hb['sd_95']/P*100:.1f}% relative)")
    print(f"  smallest single-number bias this history can detect (2 SE): +-{sd_detect:.4f} ({sd_detect/P*100:.0f}% relative)")
    report["bayes"] = hb

    print(f"\nPART B - {args.runs} WALK-FORWARD RUNS ON THE REAL DRAWS (train on all earlier draws, predict each next draw)")
    wf = walk_forward(draws, args.runs, args.seed, "real")
    pooled = pool_runs(wf)
    print(f"  each run tests {wf['block']} consecutive draws; runs cover #{wf['runs'][0]['draws'][0]}..#{wf['runs'][-1]['draws'][1]}")
    print(f"  {'model':46s} {'log-score vs fair':>18s} {'z':>6s} {'runs better':>11s} {'runs >2SE':>9s}"
          f" {'hits/draw':>9s} {'prize %':>7s} {'net VND (1 ticket/draw)':>24s}")
    for name, s in sorted(pooled.items(), key=lambda kv: -kv[1]["ls_diff"]):
        print(f"  {name:46s} {s['ls_diff']:+10.4f}+-{s['ls_se']:.4f} {s['z']:+6.2f} {s['runs_won']:>7d}/{args.runs}"
              f" {s['runs_sig']:>6d}/{args.runs} {s['hits']:9.3f} {s['prize_rate']*100:6.2f}% {s['money']:>24,.0f}")
    print(f"  fair-lottery expectation: hits/draw 0.800, prize rate {sum(P_MATCH[3:])*100:.2f}%")
    print_run_matrix(wf, pooled)
    report["walk_forward_real"] = pooled
    report["walk_forward_real_runs"] = [
        {"run": r["run"], "draws": r["draws"],
         "models": {k: {kk: v[kk] for kk in ("ls_diff", "ls_se", "hits", "prize_rate", "money")}
                    for k, v in r["summary"].items()}} for r in wf["runs"]]

    print(f"\nPART C - NEGATIVE CONTROL: SAME {args.runs} RUNS ON {args.controls} FAKE LOTTERIES FROM A FAIR RANDOM GENERATOR")
    controls = []
    for c in range(args.controls):
        fake = [{"id": d["id"], "date": d["date"], "result": r}
                for d, r in zip(draws, fair_lottery(n, random.Random(args.seed + 1 + c)))]
        controls.append(pool_runs(walk_forward(fake, args.runs, args.seed + c, f"fake{c + 1}")))
    print(f"  {'model':46s} {'REAL runs better':>16s} {'FAKE runs better (each fake)':>30s}"
          f" {'REAL hits':>9s} {'FAKE hits (mean)':>16s}")
    for name in sorted(pooled, key=lambda k: -pooled[k]["ls_diff"]):
        fake_won = [c[name]["runs_won"] for c in controls]
        fake_hits = sum(c[name]["hits"] for c in controls) / len(controls)
        print(f"  {name:46s} {pooled[name]['runs_won']:>12d}/{args.runs}   "
              f"{' '.join(str(v) for v in fake_won):>26s}   {pooled[name]['hits']:9.3f} {fake_hits:16.3f}")
    report["walk_forward_fake"] = controls

    print("\nPART D - SALES, PLAYER HABITS AND EXPECTED VALUE")
    sales = reconstruct_sales(draws)
    if not sales:
        print("  prize data not collected yet - run lottery/collect_prizes.py first")
    else:
        ratio = sorted(s["tickets_from_3match"] / s["tickets"] for s in sales if s["tickets"] > 0)
        print(f"  {len(sales)} draws with prize data; tickets from jackpot growth vs from 3-match winners:"
              f" median ratio {ratio[len(ratio)//2]:.3f}, IQR {ratio[len(ratio)//4]:.3f}-{ratio[3*len(ratio)//4]:.3f}")
        last = sales[-1]
        print(f"  last draw #{last['id']} {last['date']}: jackpot {last['jackpot_value']:,} VND, "
              f"est. {last['tickets']:,.0f} tickets ({last['revenue']/1e9:.1f} ty VND)")
        pop = popularity_model(sales)
        ranked = sorted(range(1, N_BALLS + 1), key=lambda i: pop["beta"][i - 1])
        print(f"  player popularity (log winners-vs-expected, {pop['n']} draws, resid SD {pop['resid_sd']:.3f}):")
        print("    least picked: " + ", ".join(f"{i} ({pop['beta'][i-1]:+.3f})" for i in ranked[:8]))
        print("    most picked : " + ", ".join(f"{i} ({pop['beta'][i-1]:+.3f})" for i in ranked[-8:][::-1]))
        print("    3-match winners vs expected, by how many of the 6 drawn numbers are 1-31 (birthdays):")
        for k, (cnt, mean) in pop["by_birthday_count"].items():
            print(f"      {k} birthday numbers: {cnt:4d} draws, winners x{math.exp(mean):.2f} of expected")
        share_chk = jackpot_share_check(sales)
        coverage = share_chk["winners"] / share_chk["expected_winners"]
        print(f"  jackpot winners 2016-2026: {share_chk['winners']} (if everyone picked at random: {share_chk['expected_winners']:.1f});"
              f" draws won {share_chk['won_draws']} (random: {share_chk['expected_won_draws']:.1f});"
              f" draws with 2+ winners: {share_chk['multi_winner_draws']}")
        cal = popularity_calibration(sales, pop["beta"], 5)
        cal4 = popularity_calibration(sales, pop["beta"], 4)
        print(f"  popularity score predicts 5-match winners: co-pickers = exp({cal['a']:+.3f} {cal['c']:+.2f} x score);"
              f" 4-match: exp({cal4['a']:+.3f} {cal4['c']:+.2f} x score)")
        pattern = pattern_factors(sales, pop["beta"], cal["c"])
        print("  5-match winners vs expected, by consecutive pairs in the draw (below 1 = players avoid that shape):")
        print("    " + " | ".join(f"{g}{'+' if g == 3 else ''} pairs x{r:.3f} +-{se:.3f}" for g, (r, se) in pattern.items()))

        def co_pickers(ticket):
            base_mult = math.exp(cal["c"] * sum(pop["beta"][i - 1] for i in ticket))
            return base_mult * pattern.get(min(adjacent_pairs(ticket), 3), (1.0, 0.0))[0]

        curve, _ = sales_curve(sales)
        least6 = sorted(ranked[:6])
        most6 = sorted(ranked[-6:])
        mult_least = co_pickers(least6)
        fc = forecast_sales(sales)
        print(f"  NEXT DRAW #{last['id'] + 1}: advertised jackpot {fc['advertised']/1e9:.1f} ty VND;"
              f" last draw sold {last['tickets']:,.0f} tickets; sales model (median error 3.4% in a 300-draw backtest)"
              f" forecasts {fc['tickets']:,.0f}")
        print(f"    {'scenario':>18s} {'tickets':>10s} {'jackpot at draw':>15s} {'P(jackpot won)':>14s}"
              f" {'your share if you win':>21s} {'EV random pick':>14s} {'EV least-picked':>15s}")
        scenarios = {}
        for label, t in (("model forecast", fc["tickets"]), ("x1.5 last draw", last["tickets"] * 1.5),
                         ("x2.0 last draw", last["tickets"] * 2.0)):
            j = fc["advertised"] + ACCUM_RATE * t * TICKET
            e = expected_value(j, t)
            e_least = expected_value(j, t, mult_least)
            p_won = 1 - math.exp(-t * P_MATCH[6] * coverage)
            scenarios[label] = {"tickets": t, "jackpot": j, "p_won": p_won, **e, "ev_least": e_least["ev"]}
            print(f"    {label:>18s} {t:10,.0f} {j/1e9:12.1f} ty {p_won*100:13.1f}% {e['share']:21.2f}"
                  f" {e['ev']:14,.0f} {e_least['ev']:15,.0f}")
        base = scenarios["model forecast"]
        print("    tickets, using the model forecast:")
        quick = sorted(random.Random(args.seed + 7).sample(range(1, N_BALLS + 1), K))
        report["tickets"] = {}
        for tag, ticket in (("6 least-picked numbers", least6), ("one random quick pick", quick),
                            ("neatly spread (1 per band)", [3, 11, 19, 27, 35, 43]), ("6 most-picked numbers", most6)):
            mult = co_pickers(ticket)
            e2 = expected_value(base["jackpot"], base["tickets"], mult)
            print(f"      {tag:24s} {' '.join(f'{x:02d}' for x in ticket)}: co-pickers x{mult:.2f},"
                  f" share if you win {e2['share']:.2f}, EV {e2['ev']:,.0f} VND per 10,000 VND ticket")
            report["tickets"][tag] = {"numbers": ticket, "mult": mult, **e2}
        ratio = base["ev"] / TICKET
        bank = kelly_bankroll(ratio, (base["jackpot"] - 0.1 * (base["jackpot"] - 10_000_000)) * base["share"])
        print(f"    EV/price = {ratio:.2f}; Kelly criterion: bankroll needed to justify ONE ticket:"
              f" {bank/1e9:,.0f} ty VND" if bank else f"    EV/price = {ratio:.2f} < 1: Kelly says buy nothing")
        for per_draw, years in ((1, 10), (1, 50), (10, 50)):
            n_tickets = per_draw * 156 * years
            p_any = 1 - (1 - P_MATCH[6]) ** n_tickets
            print(f"    {per_draw} ticket(s) every draw for {years} years = {n_tickets:,} tickets"
                  f" ({n_tickets * TICKET / 1e6:,.0f} million VND): P(at least one jackpot) = {p_any*100:.3f}%")
        be = next((j for j in range(12, 1000) if expected_value(j * 1e9, curve(j * 1e9))["ev"] >= TICKET), None)
        print(f"    break-even jackpot (EV = ticket price, typical sales) ~ {be} ty VND")
        hist_ev = [expected_value(s["pool"], s["tickets"])["ev"] for s in sales]
        print(f"    history: {sum(e >= TICKET for e in hist_ev)} of {len(hist_ev)} draws had EV >= ticket price;"
              f" mean EV {sum(hist_ev)/len(hist_ev):,.0f} VND per ticket")
        report["ev"] = {"scenarios": scenarios, "break_even_ty": be, "coverage": coverage,
                        "popularity_beta": pop["beta"], "kelly_bankroll": bank}

    print("\nPART E - PROBABILITY LIST FOR THE NEXT DRAW")
    hist = History([d["result"] for d in draws])
    counts_now = [hist.cum[n][i] for i in range(1, N_BALLS + 1)]
    kappa = eb_kappa(counts_now, n)
    a, b = P * kappa, (1 - P) * kappa
    post = [(i, (a + counts_now[i - 1]) / (a + b + n)) for i in range(1, N_BALLS + 1)]
    post.sort(key=lambda x: -x[1])
    raw = sorted((counts_now[i - 1] / n, i) for i in range(1, N_BALLS + 1))
    print(f"  raw frequencies (no pooling): {raw[0][0]:.4f} (#{raw[0][1]}) .. {raw[-1][0]:.4f} (#{raw[-1][1]})")
    print(f"  hierarchical posterior (partial pooling, kappa={kappa:.3g}{' = grid max, data prefer zero spread' if kappa >= 1e8 else ''}):"
          f" {post[-1][1]:.5f} .. {post[0][1]:.5f} for every number (fair = {P:.5f})")
    report["next_draw_probs"] = post
    picks = next_draw_picks(draws, args.seed)
    print("  each model's 6 numbers for the next draw (for comparison only):")
    for name, pick in picks.items():
        print(f"    {name:46s} {' '.join(f'{x:02d}' for x in pick)}")
    report["next_draw_picks"] = picks
    (RESULTS / "latest.json").write_text(json.dumps(report, default=str, indent=1))
    print(f"\nsaved {RESULTS / 'latest.json'}")


if __name__ == "__main__":
    main()
