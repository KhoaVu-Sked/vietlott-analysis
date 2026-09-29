"""Candidate ticket matrices: M sorted rows of k distinct numbers from 1..N as int16, either sampled uniformly from
all $\\binom{N}{k}$ tickets (the k smallest of N random keys is a uniform k-subset) or every ticket in lexicographic
order, built level by level."""

import math

import numpy as np

MAX_ALL = 10_000_000


def sample(n_balls, k, m, rng):
    rows = np.empty((m, k), dtype=np.int16)
    block = 1 << 16
    for start in range(0, m, block):
        keys = rng.random((min(m, start + block) - start, n_balls))
        rows[start:start + len(keys)] = np.sort(np.argpartition(keys, k - 1, axis=1)[:, :k], axis=1) + 1
    return rows


def all_tickets(n_balls, k):
    total = math.comb(n_balls, k)
    if total > MAX_ALL:
        raise ValueError(f"C({n_balls}, {k}) = {total:,} tickets is more than the {MAX_ALL:,} limit; use sample()")
    rows = np.arange(1, n_balls + 1, dtype=np.int16).reshape(-1, 1)
    for _ in range(1, k):
        last = rows[:, -1].astype(np.int64)
        counts = n_balls - last
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        step = np.arange(int(counts.sum())) - offsets
        rows = np.column_stack([np.repeat(rows, counts, axis=0), (np.repeat(last, counts) + 1 + step).astype(np.int16)])
    return rows
