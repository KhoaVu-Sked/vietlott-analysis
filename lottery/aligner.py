"""Aligned vote over every model. Each model's ticket votes for its numbers with the model's weight, and the k numbers
with the most weight make the aligned ticket. After each draw every weight moves a little toward the models that
caught more: w <- w * exp(STEP * (caught - luck)), with luck = k * k / N the numbers a random ticket catches. The ticket
for a draw only uses weights learned from earlier draws, so the whole run is walk-forward."""

import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path

STEP = 0.1
PRED_DIR = Path(__file__).resolve().parent.parent / "predictions"


def vote(tickets, weights, n_balls, k, rng):
    score = [0.0] * (n_balls + 1)
    for ticket, w in zip(tickets, weights):
        for x in ticket:
            score[x] += w
    return sorted(sorted(range(1, n_balls + 1), key=lambda i: (-score[i], rng.random()))[:k])


def shares(names, log_w):
    top = max(log_w)
    raw = [math.exp(v - top) for v in log_w]
    total = sum(raw)
    return {name: w / total for name, w in zip(names, raw)}


def align(model_tickets, outcomes, n_balls, k, next_tickets=None, step=STEP, seed=2026):
    names = sorted(model_tickets)
    luck = k * k / n_balls
    log_w = [0.0] * len(names)
    rng = random.Random(seed)
    steps = []
    for j, drawn in enumerate(outcomes):
        weights = shares(names, log_w)
        ticket = vote([model_tickets[n][j] for n in names], [weights[n] for n in names], n_balls, k, rng)
        drawn = set(drawn)
        steps.append({"ticket": ticket, "hits": len(set(ticket) & drawn),
                      "leaders": [[n, weights[n]] for n in sorted(names, key=lambda n: -weights[n])[:3]]})
        log_w = [v + step * (len(set(model_tickets[n][j]) & drawn) - luck) for v, n in zip(log_w, names)]
    final = shares(names, log_w)
    nxt = None
    if next_tickets:
        nxt = vote([next_tickets[n] for n in names], [final[n] for n in names], n_balls, k, rng)
    return {"steps": steps, "hits": [s["hits"] for s in steps], "weights": final, "next": nxt}


def fingerprint(record):
    body = {key: value for key, value in record.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def freeze(code, draw_id, ticket, weights, last_draw, name=None, directory=PRED_DIR):
    path = Path(directory) / f"aligned_{code}_draw_{draw_id:05d}.json"
    if path.exists():
        return path, False
    leaders = sorted(weights.items(), key=lambda kv: -kv[1])
    record = {"game": name or code, "draw": draw_id,
              "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "history_last_draw": last_draw, "step": STEP, "ticket": ticket,
              "weights": {n: round(w, 6) for n, w in leaders}}
    record["sha256"] = fingerprint(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    top = ", ".join(f"{n} {w * 100:.1f}%" for n, w in leaders[:3])
    note = " (main numbers; the special number is not voted)" if code == "535" else ""
    path.with_suffix(".txt").write_text(
        f"Aligned vote for {record['game']} draw #{draw_id}, frozen {record['frozen_at']} from draws up to"
        f" #{last_draw}\n  ticket: {' '.join(f'{x:02d}' for x in ticket)}{note}\n  most trusted now: {top}\n"
        f"  sha256: {record['sha256']}\n")
    return path, True
