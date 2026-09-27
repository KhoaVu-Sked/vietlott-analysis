"""Tickets chosen only for the chance of winning a prize. Two tickets that share numbers tend to win on the same draws,
which wastes one of them, so the tickets are spread over every number with as little overlap as possible. The exact
chance that two tickets sharing o numbers both win (inclusion-exclusion, second order) weights each overlap."""

import math
import random

K = 6


def both_win(n_balls, overlap, need=3, size=K):
    only = size - overlap
    rest = n_balls - 2 * size + overlap
    count = 0
    for s in range(overlap + 1):
        for a in range(only + 1):
            for b in range(only + 1):
                r = size - s - a - b
                if 0 <= r <= rest and s + a >= need and s + b >= need:
                    count += math.comb(overlap, s) * math.comb(only, a) * math.comb(only, b) * math.comb(rest, r)
    return count / math.comb(n_balls, size)


def design(k, n_balls, seed=2026, iters=40_000, size=K):
    rng = random.Random(seed)
    weight = [both_win(n_balls, o, size=size) for o in range(size + 1)]
    numbers = list(range(1, n_balls + 1))
    rng.shuffle(numbers)
    base, extra = divmod(size * k, n_balls)
    pool = numbers * base + numbers[:extra]
    tickets = [[pool[j + i * k] for i in range(size)] for j in range(k)]

    def cost_of(j, ticket):
        mine = set(ticket)
        dup = (size - len(mine)) * 1e3
        return dup + sum(weight[len(mine & set(tickets[m]))] for m in range(k) if m != j)

    for _ in range(iters):
        j, m = rng.sample(range(k), 2)
        a, b = rng.randrange(size), rng.randrange(size)
        x, y = tickets[j][a], tickets[m][b]
        if x == y:
            continue
        before = cost_of(j, tickets[j]) + cost_of(m, tickets[m])
        tickets[j][a], tickets[m][b] = y, x
        after = cost_of(j, tickets[j]) + cost_of(m, tickets[m])
        if after > before:
            tickets[j][a], tickets[m][b] = x, y
    return sorted(sorted(t) for t in tickets)
