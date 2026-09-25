#!/usr/bin/env python3
"""Pick Vietlott tickets for the next draw in a Claude Code style terminal session.

Usage: python3 prediction.py                                 # interactive: arrow keys, type, Enter
       python3 prediction.py --game 645 --tickets 10         # no questions, plain text output
"""

import argparse
import contextlib
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lottery"))

import tui

MAX_TICKETS = 50
EXACT_LIMIT = 10
SIMS = 200_000
TICKET = 10_000
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GAME_NAME = {"645": "Mega 6/45", "655": "Power 6/55"}


@contextlib.contextmanager
def power655_constants():
    import power645_study as s
    import power655_study as sp
    names = ("N_BALLS", "K", "P", "C_TOTAL", "P_MATCH", "FIXED_PRIZE", "JACKPOT_MIN", "TICKET")
    saved = {name: getattr(s, name) for name in names}
    sp.use_655_constants()
    try:
        yield sp
    finally:
        for name, value in saved.items():
            setattr(s, name, value)


def last_row(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]), len(lines)


def game_overview():
    import power655_study as sp
    import predict_draw as pr
    mega, n645 = last_row(ROOT / "data" / "mega645.jsonl")
    power, n655 = last_row(ROOT / "data" / "power655.jsonl")
    adv645 = 12e9 if mega["jackpot_winners"] else mega["jackpot_value"]
    adv655 = 30e9 if power["j1_winners"] else min(power["j1_prize"], 300e9)
    return {"645": (pr.next_draw_date(mega["date"]), adv645, n645),
            "655": (sp.next_draw_date(power["date"]), adv655, n655)}


def refresh(game):
    import collect_prizes as c
    try:
        if game == "645":
            out, fetch, latest = c.OUT, c.fetch, c.latest_id
        else:
            import collect_power655 as cp
            out, fetch, latest = cp.OUT, cp.fetch, cp.latest_id
        rows = c.load(out)
        missing = list(range(max(rows) + 1, latest() + 1))
        for i in missing:
            rows[i] = fetch(i)
        if missing:
            c.save(rows, out)
        return f"up to date with vietlott.vn ({len(missing)} new draw{'s' if len(missing) != 1 else ''} added)"
    except Exception as e:
        return f"could not reach vietlott.vn ({type(e).__name__}), using the saved data"


def simulated_any(tickets, n_balls, seed):
    rng = random.Random(seed)
    masks = [sum(1 << x for x in t) for t in tickets]
    balls = range(1, n_balls + 1)
    hit = 0
    for _ in range(SIMS):
        drawn = 0
        for x in rng.sample(balls, 6):
            drawn |= 1 << x
        hit += any(bin(drawn & m).count("1") >= 3 for m in masks)
    return hit / SIMS


def spread_design(optimise, scorer, k, n_balls, seed):
    base_uses = max(2, math.ceil(6 * k / n_balls))
    for overlap in range(6):
        if overlap == 0 and (k > 6 or 6 * k > n_balls):
            continue
        for uses in (base_uses, base_uses + 1, base_uses + 2, None):
            try:
                return optimise(scorer, k, overlap, random.Random(seed), max_uses=uses)
            except ValueError:
                pass
    raise RuntimeError("could not build a ticket set")


def mega645(k, seed, status):
    import backtest_checkpoints as b
    import optimise_tickets as o
    import power645_study as s
    import predict_draw as pr
    import verify_tickets as v
    status("Reading 1,500+ past draws")
    draws, _, _ = s.load_draws()
    sales = s.reconstruct_sales(draws)
    bundle = b.popularity_bundle(sales)
    g = o.hot_coefficient(sales, bundle)
    recent = [0] * (s.N_BALLS + 1)
    for d in sales[-o.HOT_WINDOW:]:
        for x in d["result"]:
            recent[x] += 1
    fc = s.forecast_sales(sales)
    scorer = o.Scorer(bundle, g, recent, fc["jackpot"], fc["tickets"])
    status("Choosing your tickets")
    tickets = spread_design(o.optimise, scorer, k, s.N_BALLS, seed)
    status("Working out the exact odds" if k <= EXACT_LIMIT else f"Simulating {SIMS:,} draws")
    p_any = float(v.summarise(v.enumerate_all(tickets), k)["any"]) if k <= EXACT_LIMIT else simulated_any(
        tickets, s.N_BALLS, seed)
    share = s.jackpot_share_check(sales)
    p_won = 1 - math.exp(-fc["tickets"] * s.P_MATCH[6] * share["winners"] / share["expected_winners"])
    return {"game": "Mega 6/45", "draw": draws[-1]["id"] + 1, "date": pr.next_draw_date(draws[-1]["date"]),
            "n_draws": len(draws), "tickets": tickets, "p_any": p_any, "p_jackpot": k / s.C_TOTAL,
            "ev": sum(scorer.ev(t)["ev"] for t in tickets),
            "forecast": [f"jackpot at the draw ≈ {fc['jackpot']/1e9:.1f} tỷ (now {fc['advertised']/1e9:.1f} tỷ)"
                         f" · someone wins it: {p_won*100:.0f}%"]}


def power655(k, seed, status):
    import backtest_checkpoints as b
    import optimise_tickets as o
    with power655_constants() as sp:
        status("Reading 1,400+ past draws")
        draws, _ = sp.load_draws()
        sales = sp.reconstruct_sales(draws)
        bundle = b.popularity_bundle(sales)
        fc = sp.forecast(sales)
        scorer = sp.Scorer(bundle, fc["j1"], fc["j2"], fc["tickets"])
        status("Choosing your tickets")
        tickets = spread_design(o.optimise, scorer, k, sp.N, seed)
        status("Working out the exact odds" if k <= EXACT_LIMIT else f"Simulating {SIMS:,} draws")
        exact = sp.exact_summary(tickets)[0] if k <= EXACT_LIMIT else {
            "any": simulated_any(tickets, sp.N, seed), **sp.jackpot_chances(tickets)}
        j1_w = sum(r["j1_winners"] for r in sales)
        e1 = sum(r["tickets"] * sp.P_J1 for r in sales)
        p_won1 = 1 - math.exp(-fc["tickets"] * sp.P_J1 * j1_w / e1)
        return {"game": "Power 6/55", "draw": draws[-1]["id"] + 1, "date": sp.next_draw_date(draws[-1]["date"]),
                "n_draws": len(draws), "tickets": tickets, "p_any": exact["any"], "p_jackpot": exact["j1_any"],
                "p_jackpot2": exact["j2_any"], "ev": sum(scorer.ev(t)["ev"] for t in tickets),
                "forecast": [f"Jackpot 1 at the draw ≈ {fc['j1']/1e9:.1f} tỷ · someone wins it: {p_won1*100:.0f}%",
                             f"Jackpot 2 ≈ {fc['j2']/1e9:.2f} tỷ · needs 5 numbers plus the bonus ball"]}


def compute(game, k, seed, update, status):
    note = "not updated (--no-update)"
    if update:
        status("Checking vietlott.vn for new draws")
        note = refresh(game)
    return (mega645 if game == "645" else power655)(k, seed, status), note


def when(r):
    day = WEEKDAY[dt.date.fromisoformat(r["date"]).weekday()]
    return f"{day} {r['date']} at 18:00, sales close 17:45"


def plain_report(r, k, note):
    cost = k * TICKET
    pct = f"{r['p_any']*100:.1f}%" + ("" if k <= EXACT_LIMIT else f" (estimated from {SIMS:,} simulated draws)")
    lines = [f"{r['game']}  draw #{r['draw']}  {when(r)}", f"data: {r['n_draws']:,} past draws, {note}"]
    lines += [f"forecast: {line}" for line in r["forecast"]]
    lines += ["", f"Your {k} ticket{'s' if k > 1 else ''} ({cost:,} VND):"]
    lines += [f"  {i:>2d})  " + "  ".join(f"{x:02d}" for x in t) for i, t in enumerate(r["tickets"], 1)]
    lines += ["", f"chance of winning any prize: {pct}   (chance of winning nothing: {(1-r['p_any'])*100:.1f}%)",
              f"chance of the jackpot: 1 in {1/r['p_jackpot']:,.0f}"]
    if "p_jackpot2" in r:
        lines.append(f"chance of Jackpot 2: 1 in {1/r['p_jackpot2']:,.0f}")
    lines.append(f"average value back: {r['ev']:,.0f} VND for {cost:,} VND ({r['ev']/cost:.2f}x the price)")
    lines += ["", "These numbers are not a prediction: every combination is equally likely. They are spread out for",
              "more small prizes and avoid numbers other players like, so a jackpot would be shared with fewer people."]
    return "\n".join(lines)


def styled_report(term, r, k, note, seconds):
    s = term.style
    cost = k * TICKET
    ratio = r["ev"] / cost
    lines = [s("⏺ ", s.accent) + s(f"{r['game']} · draw #{r['draw']}", s.bold) + s(f" · {when(r)}", s.grey)]
    lines += [s("  ⎿  " if i == 0 else "     ", s.grey) + line for i, line in enumerate(r["forecast"] + [note])]
    columns = 1 if k <= 6 else 2 if k <= 20 else 3
    gap = "   " if columns == 1 else " "
    cells = [s(f"{i:>2d}", s.grey) + ("   " if columns == 1 else "  ") + gap.join(s(f"{x:02d}", s.bold) for x in t)
             for i, t in enumerate(r["tickets"], 1)]
    height = math.ceil(len(cells) / columns)
    rows = ["    ".join(cells[c * height + row] for c in range(columns) if c * height + row < len(cells))
            for row in range(height)]
    title = f"Your {k} ticket{'s' if k > 1 else ''} · {cost:,} VND · spread out, avoiding popular numbers"
    lines += [""] + term.box(rows, title=title, colour=s.accent)
    filled = round(r["p_any"] * 24)
    bar = s("█" * filled, s.green) + s("░" * (24 - filled), s.grey)
    about = "" if k <= EXACT_LIMIT else s(f"  estimated from {SIMS:,} simulated draws", s.dim)
    chance = s(f"{r['p_any'] * 100:.1f}%", s.bold)
    jackpot = f"1 in {1/r['p_jackpot']:,.0f}" + (f"   Jackpot 2: 1 in {1/r['p_jackpot2']:,.0f}" if "p_jackpot2" in r else "")
    verdict = (s("✓ average beats the price, but most draws still pay nothing", s.green) if ratio >= 1
               else s("! below the price: on average this draw loses money", s.yellow))
    lines += ["", f"  {'Chance of any prize':21s}{chance}  {bar}{about}",
              f"  {'Chance of nothing':21s}{(1 - r['p_any'])*100:.1f}%",
              f"  {'Jackpot':21s}{jackpot}",
              f"  {'Average value back':21s}{r['ev']:,.0f} VND for {cost:,} VND · {ratio:.2f}× the price",
              f"  {verdict}", "",
              s("  Not a prediction: every combination is equally likely.", s.dim),
              s(f"  saved to results/last_prediction.txt · {seconds:.1f}s", s.dim), ""]
    return lines


def header(term, overview, full):
    s = term.style
    jackpots = f"Mega 6/45 {overview['645'][1]/1e9:.1f} tỷ · Power 6/55 {overview['655'][1]/1e9:.1f} tỷ"
    if not full:
        return term.box([s("✻ ", s.accent) + s("Vietlott Predictor", s.bold) + s(f" · jackpots {jackpots}", s.grey)],
                        colour=s.accent) + [""]
    rows = [s("✻ ", s.accent) + s("Welcome to Vietlott Predictor", s.bold), "",
            s(f"  Mega 6/45   {overview['645'][2]:,} draws · jackpot now {overview['645'][1]/1e9:.1f} tỷ", s.grey),
            s(f"  Power 6/55  {overview['655'][2]:,} draws · Jackpot 1 now {overview['655'][1]/1e9:.1f} tỷ", s.grey), "",
            s(f"  cwd: {ROOT}", s.dim)]
    return term.box(rows, colour=s.accent) + [""]


def save(text):
    out = ROOT / "results" / "last_prediction.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text.strip() + "\n")


def session(seed, update):
    term = tui.Terminal()
    overview = game_overview()
    options = []
    for code in ("645", "655"):
        date, adv, _ = overview[code]
        day = WEEKDAY[dt.date.fromisoformat(date).weekday()][:3]
        options.append((GAME_NAME[code], f"next draw {day} {date[8:]}/{date[5:7]} · jackpot {adv/1e9:.1f} tỷ"))
    refreshed = set()
    game, trail = None, []
    first = True

    def fresh(*extra):
        nonlocal first
        term.clear()
        term.print(*header(term, overview, first), *trail, *extra)
        first = False

    with term:
        while True:
            if game is None:
                trail = []
                fresh()
                idx = term.menu("Which lottery?", options, "↑/↓ to move · Enter to choose · Esc to quit")
                if idx is None:
                    break
                game = ("645", "655")[idx]
                trail = [term.style("❯ ", term.style.accent) + term.style("Lottery ", term.style.grey) + GAME_NAME[game]]
            fresh()
            k = term.ask_number("How many tickets will you buy?", 10, 1, MAX_TICKETS,
                                f"1-{MAX_TICKETS} tickets, {TICKET:,} VND each · Enter to confirm · Esc to go back")
            if k is None:
                game = None
                continue
            fresh(term.style("❯ ", term.style.accent) + term.style("Tickets ", term.style.grey) + str(k))
            (result, note), seconds = term.spin(
                lambda status: compute(game, k, seed, update and game not in refreshed, status), "Starting")
            if update:
                refreshed.add(game)
            term.clear()
            term.print(*styled_report(term, result, k, note, seconds))
            save(plain_report(result, k, note))
            nxt = term.menu("What next?", [("Pick again", "same lottery, new number of tickets"),
                                           ("Switch lottery", ""), ("Quit", "")],
                            "↑/↓ to move · Enter to choose · Esc to quit")
            if nxt == 0:
                continue
            if nxt == 1:
                game = None
                continue
            break
    term.print(term.style("  Good luck, and play within your budget.", term.style.dim), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(GAME_NAME))
    ap.add_argument("--tickets", type=int)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if args.game and args.tickets:
        if not 1 <= args.tickets <= MAX_TICKETS:
            raise SystemExit(f"--tickets must be 1 to {MAX_TICKETS}")
        result, note = compute(args.game, args.tickets, args.seed, not args.no_update, lambda text: None)
        text = plain_report(result, args.tickets, note)
        print(text)
        save(text)
        return
    try:
        session(args.seed, not args.no_update)
    except KeyboardInterrupt:
        print("\n  cancelled")


if __name__ == "__main__":
    main()
