"""Shared building blocks for the models in models/. One pass over the draws keeps every running statistic, so a model
answers each call in constant time while the backtest walks a history forward one draw at a time."""

import math

K = 6
STRENGTH_GRID = [10 ** (e / 10) for e in range(10, 81)]
MAX_GAP = 31


class Tracker:
    def __init__(self, n_balls):
        self.n, self.p, self.t = n_balls, K / n_balls, 0
        self.draws = []
        self.cum = [[0] * (n_balls + 1)]
        self.last = [[-1] * (n_balls + 1)]
        self.ewma = [0.0] * (n_balls + 1)
        self.trans = [[0] * (n_balls + 1) for _ in range(n_balls + 1)]
        self.nprev = [0] * (n_balls + 1)
        self.gap_risk = [0] * (MAX_GAP + 1)
        self.gap_hit = [0] * (MAX_GAP + 1)

    def add(self, draw):
        drawn = set(draw)
        for i in range(1, self.n + 1):
            g = min(self.gap(self.t, i), MAX_GAP)
            self.gap_risk[g] += 1
            self.gap_hit[g] += i in drawn
        if self.draws:
            for j in self.draws[-1]:
                self.nprev[j] += 1
                for i in draw:
                    self.trans[j][i] += 1
        self.ewma = [0.97 * v + (1 if i in drawn else 0) for i, v in enumerate(self.ewma)]
        cum, last = self.cum[-1][:], self.last[-1][:]
        for x in draw:
            cum[x] += 1
            last[x] = self.t
        self.cum.append(cum)
        self.last.append(last)
        self.draws.append(list(draw))
        self.t += 1

    def gap(self, s, i):
        seen = self.last[s][i]
        return s - seen if seen >= 0 else s + 1

    def window(self, s, i, w):
        lo = max(0, s - w)
        return (self.cum[s][i] - self.cum[lo][i]) / max(1, s - lo)

    def features(self, s, i):
        p = self.p
        return [self.window(s, i, 10) - p, self.window(s, i, 50) - p, self.window(s, i, 200) - p,
                self.cum[s][i] / max(1, s) - p, math.log(self.gap(s, i)), 1.0 if self.last[s][i] == s - 1 else 0.0]

    def recent(self, i, w):
        return self.cum[self.t][i] - self.cum[max(0, self.t - w)][i]


class Follower:
    def __init__(self):
        self.tracker, self.generation, self.cache, self.rows = None, 0, {}, {}

    def __call__(self, past, n_balls):
        tr = self.tracker
        if (tr is None or tr.n != n_balls or tr.t > len(past)
                or (tr.t and (past[0] != tr.draws[0] or past[tr.t - 1] != tr.draws[tr.t - 1]))):
            tr = self.tracker = Tracker(n_balls)
            self.generation += 1
            self.cache, self.rows = {}, {}
        for draw in past[tr.t:]:
            tr.add(draw)
        return tr

    def training(self, span=400, start=50):
        tr = self.tracker
        rows, ys = [], []
        for t in range(max(start, tr.t - span), tr.t):
            if t not in self.rows:
                drawn = set(tr.draws[t])
                self.rows[t] = [(tr.features(t, i), 1 if i in drawn else 0) for i in range(1, tr.n + 1)]
            for x, y in self.rows[t]:
                rows.append(x)
                ys.append(y)
        return rows, ys

    def cached(self, name, every, make):
        key = (name, self.tracker.t // every)
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]


def prior_strength(pairs, p):
    best, best_lm = STRENGTH_GRID[-1], -math.inf
    for kappa in STRENGTH_GRID:
        a, b = p * kappa, (1 - p) * kappa
        base = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        lm = sum(math.lgamma(h + a) + math.lgamma(n - h + b) - math.lgamma(n + a + b) + base for h, n in pairs if n)
        if lm > best_lm:
            best, best_lm = kappa, lm
    return best


def chances(scores):
    total = sum(scores)
    return [min(max(v * K / total, 1e-4), 0.999) for v in scores]


def log_score(ch, drawn):
    return sum(math.log(c) if i in drawn else math.log(1 - c) for i, c in enumerate(ch, 1))


def fit_logistic(rows, ys, start, l2=1.0, iters=8):
    dim = len(rows[0])
    beta = list(start)
    for _ in range(iters):
        grad = [0.0] * dim
        hess = [[0.0] * dim for _ in range(dim)]
        for x, y in zip(rows, ys):
            mu = 1 / (1 + math.exp(-sum(b * v for b, v in zip(beta, x))))
            r, w = y - mu, mu * (1 - mu)
            for a in range(dim):
                grad[a] += r * x[a]
                wa, ha = w * x[a], hess[a]
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
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
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


FRAMES = ("fair", "frequency", "hot30", "ewma", "overdue", "markov", "gap", "repeat_last")


class FrameSet:
    def __init__(self, n_balls):
        self.tr = Tracker(n_balls)
        self.kappas = {}
        self.hits, self.scores = [], []

    def gap_strength(self):
        key = self.tr.t // 26
        if key not in self.kappas:
            self.kappas[key] = prior_strength(list(zip(self.tr.gap_hit, self.tr.gap_risk)), self.tr.p)
        return self.kappas[key]

    def frames(self):
        tr, n, p = self.tr, self.tr.n, self.tr.p
        nums = range(1, n + 1)
        prev = set(tr.draws[-1]) if tr.draws else set()
        kappa = self.gap_strength()
        hazard = [(p * kappa + h) / (kappa + r) for h, r in zip(tr.gap_hit, tr.gap_risk)]
        raw = [[1.0] * n,
               [tr.cum[tr.t][i] + 1.0 for i in nums],
               [tr.recent(i, 30) + 1.0 for i in nums],
               [tr.ewma[i] + 0.05 for i in nums],
               [float(tr.gap(tr.t, i)) for i in nums],
               ([sum((tr.trans[j][i] + 10 * p) / (tr.nprev[j] + 10) for j in prev) / K for i in nums]
                if prev else [1.0] * n),
               [hazard[min(tr.gap(tr.t, i), MAX_GAP)] for i in nums],
               [2.0 if i in prev else 1.0 for i in nums]]
        return [chances(r) for r in raw]

    def picks(self, ch, frame):
        import random
        rng = random.Random(self.tr.t * 100 + frame)
        order = sorted(range(1, self.tr.n + 1), key=lambda i: (-ch[i - 1], rng.random()))
        return set(order[:K])

    def add(self, draw):
        drawn = set(draw)
        frames = self.frames()
        self.hits.append([len(self.picks(ch, f) & drawn) for f, ch in enumerate(frames)])
        self.scores.append([log_score(ch, drawn) for ch in frames])
        self.tr.add(draw)


def follow_frames(state, past, n_balls):
    fs = state.get("fs")
    if (fs is None or fs.tr.n != n_balls or fs.tr.t > len(past)
            or (fs.tr.t and (past[0] != fs.tr.draws[0] or past[fs.tr.t - 1] != fs.tr.draws[fs.tr.t - 1]))):
        fs = state["fs"] = FrameSet(n_balls)
    for draw in past[fs.tr.t:]:
        fs.add(draw)
    return fs


def frame_totals(fs, window):
    recent_hits, recent_scores = fs.hits[-window:], fs.scores[-window:]
    return [(sum(h[f] for h in recent_hits), sum(s[f] for s in recent_scores)) for f in range(len(FRAMES))]
