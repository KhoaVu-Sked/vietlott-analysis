#!/usr/bin/env python3
"""Vietlott in a Claude Code style terminal session: get tickets for the next draw, or test the prediction models.

Usage: python3 prediction.py                                 # interactive: arrow keys, type, Enter
       python3 prediction.py --game 645 --tickets 10         # tickets with no questions, plain text output
       python3 lottery/backtest_models.py                    # model tests with no questions
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

import backtest_models as bm
import coverage
import tui
import update

MAX_TICKETS = 50
EXACT_LIMIT = 10
SIMS = 200_000
TICKET = 10_000
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GAME_NAME = {"645": "Mega 6/45", "655": "Power 6/55", "535": "Lotto 5/35"}
GAMES = ("645", "655", "535")
GOALS = {"win": ("Most chance to win", "every number used, least overlap"),
         "value": ("Best value", "bigger jackpot share, avoids popular numbers"),
         "cold": ("Cold-number model", "least-drawn numbers first, typical shapes"),
         "engine": ("Engine", "five-term scorer with fitted weights, greedy portfolio")}
GAME_GOALS = {"645": ("win", "value", "cold", "engine"), "655": ("win", "value", "cold", "engine"),
              "535": ("win", "cold", "engine")}
HOW = {"win": "most chance to win, least overlap", "value": "best value, avoiding popular numbers",
       "cold": "cold-number model, typical shapes", "engine": "engine: fitted five-term scorer"}
WHY = {"win": "They are spread over every number with the least overlap, for the most chance of a prize.",
       "value": "They avoid numbers other players like, so a jackpot is shared with fewer people.",
       "cold": "They use the least-drawn numbers first, in typical shapes, laid out for the most chance of a prize.",
       "engine": "They are the engine's top-scored tickets, at most k-2 numbers shared, weighted as the fitted terms say."}


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
    import lotto535_study as ls
    import power655_study as sp
    import predict_draw as pr
    mega, n645 = last_row(ROOT / "data" / "mega645.jsonl")
    power, n655 = last_row(ROOT / "data" / "power655.jsonl")
    lotto, n535 = last_row(ROOT / "data" / "lotto535.jsonl")
    adv645 = 12e9 if mega["jackpot_winners"] else mega["jackpot_value"]
    adv655 = 30e9 if power["j1_winners"] else min(power["j1_prize"], 300e9)
    adv535 = ls.J_MIN if lotto["jackpot_winners"] or ls.is_share_out(lotto) else lotto["jackpot_prize"]
    rows, _ = ls.load_draws()
    day535, hour535 = ls.draw_time(lotto["id"] + 1, rows)
    return {"645": {"date": pr.next_draw_date(mega["date"]), "time": "18:00", "jackpot": adv645, "draws": n645},
            "655": {"date": sp.next_draw_date(power["date"]), "time": "18:00", "jackpot": adv655, "draws": n655},
            "535": {"date": str(day535), "time": hour535, "jackpot": adv535, "draws": n535}}


def simulated_any(tickets, n_balls, seed, size=6):
    rng = random.Random(seed)
    masks = [sum(1 << x for x in t) for t in tickets]
    balls = range(1, n_balls + 1)
    hit = 0
    for _ in range(SIMS):
        drawn = 0
        for x in rng.sample(balls, size):
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


def mega645(k, seed, status, goal):
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
    extra = []
    if goal == "cold":
        tickets, cold_line = cold_model([d["result"] for d in draws], s.N_BALLS, 6, k, seed)
        extra = [cold_line]
    elif goal == "engine":
        tickets, extra = engine_model([d["result"] for d in draws], s.N_BALLS, 6, k, seed, status)
        if tickets is None:
            tickets, goal = coverage.design(k, s.N_BALLS, seed), "win"
    elif goal == "win":
        tickets = coverage.design(k, s.N_BALLS, seed)
    else:
        tickets = spread_design(o.optimise, scorer, k, s.N_BALLS, seed)
    status("Working out the exact odds" if k <= EXACT_LIMIT else f"Simulating {SIMS:,} draws")

    def any_prize(ts):
        if k <= EXACT_LIMIT:
            return float(v.summarise(v.enumerate_all(ts), k)["any"])
        return simulated_any(ts, s.N_BALLS, seed)

    p_any = any_prize(tickets)
    if goal == "engine":
        extra.append(f"any prize: this set {p_any * 100:.2f}%, most-chance design"
                     f" {any_prize(coverage.design(k, s.N_BALLS, seed)) * 100:.2f}%")
    share = s.jackpot_share_check(sales)
    p_won = 1 - math.exp(-fc["tickets"] * s.P_MATCH[6] * share["winners"] / share["expected_winners"])
    return {"game": "Mega 6/45", "goal": goal, "p_one": sum(s.P_MATCH[3:]),
            "draw": draws[-1]["id"] + 1, "date": pr.next_draw_date(draws[-1]["date"]),
            "n_draws": len(draws), "tickets": tickets, "p_any": p_any, "p_jackpot": k / s.C_TOTAL,
            "ev": sum(scorer.ev(t)["ev"] for t in tickets),
            "forecast": [f"jackpot at the draw ≈ {fc['jackpot']/1e9:.1f} tỷ (now {fc['advertised']/1e9:.1f} tỷ)"
                         f" · someone wins it: {p_won*100:.0f}%"] + extra}


def power655(k, seed, status, goal):
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
        extra = []
        if goal == "cold":
            tickets, cold_line = cold_model([d["result"] for d in draws], sp.N, 6, k, seed)
            extra = [cold_line]
        elif goal == "engine":
            tickets, extra = engine_model([d["result"] for d in draws], sp.N, 6, k, seed, status)
            if tickets is None:
                tickets, goal = coverage.design(k, sp.N, seed), "win"
        elif goal == "win":
            tickets = coverage.design(k, sp.N, seed)
        else:
            tickets = spread_design(o.optimise, scorer, k, sp.N, seed)
        status("Working out the exact odds" if k <= EXACT_LIMIT else f"Simulating {SIMS:,} draws")

        def any_prize(ts):
            if k <= EXACT_LIMIT:
                return sp.exact_summary(ts)[0]
            return {"any": simulated_any(ts, sp.N, seed), **sp.jackpot_chances(ts)}

        exact = any_prize(tickets)
        if goal == "engine":
            extra.append(f"any prize: this set {exact['any'] * 100:.2f}%, most-chance design"
                         f" {any_prize(coverage.design(k, sp.N, seed))['any'] * 100:.2f}%")
        j1_w = sum(r["j1_winners"] for r in sales)
        e1 = sum(r["tickets"] * sp.P_J1 for r in sales)
        p_won1 = 1 - math.exp(-fc["tickets"] * sp.P_J1 * j1_w / e1)
        return {"game": "Power 6/55", "goal": goal, "p_one": sum(sp.P_MATCH[3:]),
                "draw": draws[-1]["id"] + 1, "date": sp.next_draw_date(draws[-1]["date"]),
                "n_draws": len(draws), "tickets": tickets, "p_any": exact["any"], "p_jackpot": exact["j1_any"],
                "p_jackpot2": exact["j2_any"], "ev": sum(scorer.ev(t)["ev"] for t in tickets),
                "forecast": [f"Jackpot 1 at the draw ≈ {fc['j1']/1e9:.1f} tỷ · someone wins it: {p_won1*100:.0f}%",
                             f"Jackpot 2 ≈ {fc['j2']/1e9:.2f} tỷ · needs 5 numbers plus the bonus ball"]
                            + extra}


def cold_model(results, n_balls, size, k, seed):
    import model_tools as mt
    ln = mt.ColdLearner(n_balls, size)
    for d in results:
        ln.add(d)
    caught = sum(ln.list_hits) / len(ln.list_hits)
    line = (f"cold-number model weight now {ln.weight:.0f}/100 · its {ln.m} coldest numbers caught {caught:.2f} per draw,"
            f" luck {ln.luck():.2f}")
    return ln.tickets(results, count=k, seed=seed), line


def engine_model(results, n_balls, size, k, seed, status):
    try:
        import numpy as np
        from engine import Engine, candidates
    except ImportError:
        return None, ["the Engine goal needs numpy: pip3 install numpy"]
    status(f"Fitting the engine on {len(results):,} draws")
    eng = Engine(n_balls, size, seed=seed)
    eng.extend(results)
    fit = eng.fit()
    if math.comb(n_balls, size) <= candidates.MAX_ALL:
        status(f"Scoring all {math.comb(n_balls, size):,} tickets")
        pool = candidates.all_tickets(n_balls, size)
    else:
        status("Scoring 1,000,000 sampled tickets")
        pool = candidates.sample(n_balls, size, 1_000_000, np.random.default_rng((seed, 1)))
    tickets = eng.portfolio(k, pool)
    terms = fit.as_dict()
    parts = [f"{name} {v['w']:+.2f}±{v['se']:.2f}" for name, v in terms.items()]
    flagged = [name for name, v in terms.items() if abs(v["z"]) > 3]
    lines = ["engine weights ± error: " + " · ".join(parts[:3]), "                        " + " · ".join(parts[3:]),
             "all within error bars: nothing beyond a fair lottery" if not flagged
             else "beyond 3 errors: " + ", ".join(flagged) + " (check the study before trusting it)"]
    return tickets, lines


def lotto535(k, seed, status, goal):
    import lotto535_study as ls
    status("Reading 900+ past draws and the share-out history")
    f = ls.forecast_next(seed)
    status("Choosing your tickets")
    extra = []
    rows, _ = ls.load_draws()
    if goal == "cold":
        mains, cold_line = cold_model([r["result"] for r in rows], ls.N, ls.K, k, seed)
        extra = [cold_line]
    elif goal == "engine":
        mains, extra = engine_model([r["result"] for r in rows], ls.N, ls.K, k, seed, status)
        if mains is None:
            mains, goal = coverage.design(k, ls.N, seed, size=ls.K), "win"
    else:
        mains = coverage.design(k, ls.N, seed, size=ls.K)
    tickets = ls.with_specials(mains)
    status("Working out the exact odds" if k <= 2 * EXACT_LIMIT else f"Simulating {SIMS:,} draws")

    def any_prize(ts):
        if k <= 2 * EXACT_LIMIT:
            return ls.exact_money(ts, dict(ls.BASE, jackpot=f["jackpot"]))
        return None, simulated_any([m for m, _ in ts], ls.N, seed, size=ls.K)

    money, p_real = any_prize(tickets)
    if money is not None:
        total = sum(money.values())
        p_refund = sum(w for v, w in money.items() if v > 0) / total
    else:
        p_refund = 1 - (1 - p_real) * (1 - min(k, ls.S) / ls.S)
    if goal == "engine":
        p_design = any_prize(ls.with_specials(coverage.design(k, ls.N, seed, size=ls.K)))[1]
        extra.append(f"any prize: this set {p_real * 100:.2f}%, most-chance design {p_design * 100:.2f}%")
    p_won = 1 - math.exp(-f["tickets"] * ls.P_JP * f["coverage"])
    lines = [f"jackpot at the draw ≈ {f['jackpot']/1e9:.2f} tỷ (now {f['jackpot_now']/1e9:.2f} tỷ) · someone wins it:"
             f" {p_won*100:.0f}%"]
    lines += extra
    so = f.get("share_out")
    if f["share_out_now"]:
        lines.append(f"SHARE-OUT DRAW: if nobody wins the jackpot ({so['p_share']*100:.0f}% likely), it is shared out to the"
                     f" 3, 4 and 5-correct winners; worth about {so['ev']/1e4:.2f}× the price after tax")
    elif so:
        day = f"{f['share_date'][8:]}/{f['share_date'][5:7]} {f['share_time']}"
        when_so = f"scheduled for {day}" if f["scheduled"] else (
            f"most likely {day}, {f['p14']*100:.0f}% chance within 14 days")
        lines.append(f"next jackpot share-out: {when_so}")
        lines.append(f"share-out forecast {so['ev']/1e4:.2f}× the price after tax · past {f['history_after_tax']/1e4:.2f}×,"
                     f" the last 6 {f['last6_after_tax']/1e4:.2f}×")
    return {"game": "Lotto 5/35", "goal": goal, "p_one": ls.P_TIER["first"] + ls.P_TIER["second"] + ls.P_TIER["third"]
            + ls.P_TIER["fourth"] + ls.P_TIER["fifth"] + ls.P_JP, "draw": f["draw"], "date": f["date"], "time": f["time"],
            "close": f"sales close {'12:30' if f['time'] == '13:00' else '20:30'}", "n_draws": f["n_draws"], "tickets": tickets, "p_any": p_real,
            "p_refund": p_refund, "p_jackpot": k / ls.TOTAL, "ev": k * f["ev"], "forecast": lines,
            "share_out_now": f["share_out_now"]}


def compute(game, k, seed, fetch, status, goal="win"):
    note = "not updated (--no-update)"
    if fetch:
        status("Checking vietlott.vn for new draws")
        note = update.refresh(game)
    return {"645": mega645, "655": power655, "535": lotto535}[game](k, seed, status, goal), note


def when(r):
    day = WEEKDAY[dt.date.fromisoformat(r["date"]).weekday()]
    return f"{day} {r['date']} at {r.get('time', '18:00')}, {r.get('close', 'sales close 17:45')}"


def ticket_text(t, sep=" "):
    if isinstance(t, tuple):
        mains, special = t
        return sep.join(f"{x:02d}" for x in mains) + f" | {special:02d}"
    return sep.join(f"{x:02d}" for x in t)


def plain_report(r, k, note):
    cost = k * TICKET
    pct = f"{r['p_any']*100:.1f}%" + ("" if k <= EXACT_LIMIT else f" (estimated from {SIMS:,} simulated draws)")
    lines = [f"{r['game']}  draw #{r['draw']}  {when(r)}", f"data: {r['n_draws']:,} past draws, {note}",
             f"goal: {GOALS[r['goal']][0].lower()} ({GOALS[r['goal']][1]})"]
    lines += [f"forecast: {line}" for line in r["forecast"]]
    lines += ["", f"Your {k} ticket{'s' if k > 1 else ''} ({cost:,} VND):"]
    lines += [f"  {i:>2d})  " + ticket_text(t, "  ") for i, t in enumerate(r["tickets"], 1)]
    label = "a real prize (3+ numbers)" if "p_refund" in r else "any prize"
    lines += ["", f"chance of {label}: {pct}   (chance of winning nothing: {(1-r['p_any'])*100:.1f}%)",
              f"upper limit for {k} tickets: {min(1, k * r['p_one'])*100:.1f}% ({k} x one ticket's {r['p_one']*100:.2f}%)",
              f"chance of the jackpot: 1 in {1/r['p_jackpot']:,.0f}"]
    if "p_jackpot2" in r:
        lines.append(f"chance of Jackpot 2: 1 in {1/r['p_jackpot2']:,.0f}")
    if "p_refund" in r:
        lines.append(f"chance of at least the 10,000 VND refund: {r['p_refund']*100:.1f}%")
    lines.append(f"average value back: {r['ev']:,.0f} VND for {cost:,} VND ({r['ev']/cost:.2f}x the price)")
    lines += ["", "These numbers are not a prediction: every combination is equally likely.", WHY[r["goal"]]]
    return "\n".join(lines)


def styled_report(term, r, k, note, seconds):
    s = term.style
    cost = k * TICKET
    ratio = r["ev"] / cost
    lines = [s("⏺ ", s.accent) + s(f"{r['game']} · draw #{r['draw']}", s.bold) + s(f" · {when(r)}", s.grey)]
    lines += [s("  ⎿  " if i == 0 else "     ", s.grey) + line for i, line in enumerate(r["forecast"] + [note])]
    columns = 1 if k <= 6 else 2 if k <= 20 else 3
    gap = "   " if columns == 1 else " "
    def cell(t):
        if isinstance(t, tuple):
            return gap.join(s(f"{x:02d}", s.bold) for x in t[0]) + s(" | ", s.grey) + s(f"{t[1]:02d}", s.accent)
        return gap.join(s(f"{x:02d}", s.bold) for x in t)

    cells = [s(f"{i:>2d}", s.grey) + ("   " if columns == 1 else "  ") + cell(t) for i, t in enumerate(r["tickets"], 1)]
    height = math.ceil(len(cells) / columns)
    rows = ["    ".join(cells[c * height + row] for c in range(columns) if c * height + row < len(cells))
            for row in range(height)]
    title = f"Your {k} ticket{'s' if k > 1 else ''} · {cost:,} VND · {HOW[r['goal']]}"
    lines += [""] + term.box(rows, title=title, colour=s.accent)
    filled = round(r["p_any"] * 24)
    bar = s("█" * filled, s.green) + s("░" * (24 - filled), s.grey)
    about = "" if k <= EXACT_LIMIT else s(f"  estimated from {SIMS:,} simulated draws", s.dim)
    chance = s(f"{r['p_any'] * 100:.1f}%", s.bold)
    jackpot = f"1 in {1/r['p_jackpot']:,.0f}" + (f"   Jackpot 2: 1 in {1/r['p_jackpot2']:,.0f}" if "p_jackpot2" in r else "")
    verdict = (s("✓ average beats the price, but most draws still pay nothing", s.green) if ratio >= 1
               else s("! below the price: on average this draw loses money", s.yellow))
    label = "Chance of real prize" if "p_refund" in r else "Chance of any prize"
    second = (f"  {'Refund or better':21s}{r['p_refund']*100:.1f}%" + s("  a refund needs only the special number", s.dim)
              if "p_refund" in r else f"  {'Chance of nothing':21s}{(1 - r['p_any'])*100:.1f}%")
    lines += ["", f"  {label:21s}{chance}  {bar}{about}", second,
              f"  {'Upper limit':21s}{min(1, k * r['p_one'])*100:.1f}%" + s(f"  ({k} × {r['p_one']*100:.2f}% for one ticket)",
                                                                           s.dim),
              f"  {'Jackpot':21s}{jackpot}",
              f"  {'Average value back':21s}{r['ev']:,.0f} VND for {cost:,} VND · {ratio:.2f}× the price",
              f"  {verdict}", "",
              s("  Not a prediction: every combination is equally likely.", s.dim),
              s(f"  saved to results/last_prediction.txt · {seconds:.1f}s", s.dim), ""]
    return lines


def header(term, overview, full):
    s = term.style
    jackpots = " · ".join(f"{GAME_NAME[g].split()[1]} {overview[g]['jackpot']/1e9:.1f}" for g in GAMES)
    if not full:
        return term.box([s("✻ ", s.accent) + s("Vietlott Predictor", s.bold) + s(f" · jackpots in tỷ: {jackpots}", s.grey)],
                        colour=s.accent) + [""]
    rows = [s("✻ ", s.accent) + s("Welcome to Vietlott Predictor", s.bold), ""]
    rows += [s(f"  {GAME_NAME[g]:11s} {overview[g]['draws']:,} draws · jackpot now {overview[g]['jackpot']/1e9:.1f} tỷ",
               s.grey) for g in GAMES]
    rows += ["", s(f"  cwd: {ROOT}", s.dim)]
    return term.box(rows, colour=s.accent) + [""]


def save(text):
    out = ROOT / "results" / "last_prediction.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text.strip() + "\n")


def tests_report(term, result, seconds):
    s = term.style
    lines = []
    for out in result["games"].values():
        lines.append(s("⏺ ", s.accent) + s(out["game"], s.bold)
                     + s(f" · tested {out['tested_draws']} · hold-out {out['holdout_draws']}", s.grey))
        rows = [s(f"{'model':23s}{'hit rate':>9s}{'hold-out':>9s}{'score':>9s}   verdict", s.grey)]
        for name, r in bm.ranked(out):
            short = ("beats fair: test on live draws" if r["verdict"].startswith("beats") else
                     "within luck" if r["z_vs_fair"] > 0 else "no better than random")
            colour = s.green if short.startswith("beats") else s.grey if short == "within luck" else ""
            score = "" if r["log_score_vs_fair"] is None else f"{r['log_score_vs_fair'] * 1000:+.1f}"
            rows.append(f"{name[:23]:23s}{r['hit_rate_pct']:8.2f}%{r['holdout']['hit_rate_pct']:8.2f}%{score:>9s}   "
                        + s(short, colour))
        lines += term.box(rows, title=f"luck alone catches {out['fair_hit_rate_pct']:.2f}% of the numbers",
                          colour=s.accent) + [""]
    notes = {out["data"] for out in result["games"].values()}
    tried = sum(out["model_versions_tried_on_this_history"] for out in result["games"].values())
    lines += [s("  score = log-score per draw vs a random ticket (0 = as good, below 0 = worse)", s.dim),
              s(f"  data: {' / '.join(sorted(notes))} · {seconds:.1f}s", s.dim),
              s(f"  {tried} model versions tried so far: about 1 in 20 looks good by luck alone", s.dim),
              s("  add a model: copy models/_template.py · details: results/result.json", s.dim),
              ""]
    return lines


def session(seed, fetch):
    term = tui.Terminal()
    overview = game_overview()
    lottery_options = []
    for code in GAMES:
        o = overview[code]
        day = WEEKDAY[dt.date.fromisoformat(o["date"]).weekday()][:3]
        lottery_options.append((GAME_NAME[code], f"next draw {day} {o['date'][8:]}/{o['date'][5:7]} {o['time']}"
                                                 f" · jackpot {o['jackpot']/1e9:.1f} tỷ"))
    notes = {}
    state, game, goal, trail, first = "home", None, "win", [], True

    def fresh(*extra):
        nonlocal first
        term.clear()
        term.print(*header(term, overview, first), *trail, *extra)
        first = False

    def answer(label, value):
        return term.style("❯ ", term.style.accent) + term.style(label + " ", term.style.grey) + value

    def next_step(options):
        choice = term.menu("What next?", [(label, detail) for label, detail, _ in options],
                           "↑/↓ to move · Enter to choose · Esc for the main menu")
        return "home" if choice is None else options[choice][2]

    with term:
        while state != "quit":
            if state == "home":
                trail, game = [], None
                fresh()
                choice = term.menu("What would you like to do?",
                                   [("Get tickets", "numbers for the next draw, with their odds and value"),
                                    ("Test models", "every model on every past draw, all three games"),
                                    ("Quit", "")], "↑/↓ to move · Enter to choose · Esc to quit")
                state = {0: "lottery", 1: "tests"}.get(choice, "quit")
            elif state == "lottery":
                trail = [answer("Get", "tickets")]
                fresh()
                idx = term.menu("Which lottery?", lottery_options, "↑/↓ to move · Enter to choose · Esc to go back")
                if idx is None:
                    state = "home"
                    continue
                game = GAMES[idx]
                trail.append(answer("Lottery", GAME_NAME[game]))
                state = "goal"
            elif state == "goal":
                fresh()
                choices = GAME_GOALS[game]
                idx = term.menu("How should the tickets be picked?", [GOALS[g] for g in choices],
                                "↑/↓ to move · Enter to choose · Esc to go back")
                if idx is None:
                    trail = trail[:1]
                    state = "lottery"
                    continue
                goal = choices[idx]
                trail.append(answer("Goal", GOALS[goal][0]))
                state = "tickets"
            elif state == "tickets":
                fresh()
                k = term.ask_number("How many tickets will you buy?", 10, 1, MAX_TICKETS,
                                    f"1-{MAX_TICKETS} tickets, {TICKET:,} VND each · Enter to confirm · Esc to go back")
                if k is None:
                    trail = trail[:2]
                    state = "goal"
                    continue
                fresh(answer("Tickets", str(k)))
                fetch_now = fetch and game not in notes
                (result, note), seconds = term.spin(
                    lambda status: compute(game, k, seed, fetch_now, status, goal), "Starting")
                if fetch_now:
                    notes[game] = note
                elif fetch:
                    note = notes[game]
                term.clear()
                term.print(*styled_report(term, result, k, note, seconds))
                save(plain_report(result, k, note))
                state = next_step([("Pick again", "same lottery, new number of tickets", "tickets"),
                                   ("Switch lottery", "", "lottery"), ("Main menu", "", "home"),
                                   ("Quit", "", "quit")])
                if state == "lottery":
                    trail = trail[:1]
            elif state == "tests":
                trail = [answer("Test", "models")]
                fresh()
                games = GAMES
                todo = tuple(g for g in games if g not in notes) if fetch else ()
                result, seconds = term.spin(lambda status: bm.run(
                    games, seed=seed, fetch=bool(todo), status=status), "Starting")
                for code, out in result["games"].items():
                    if todo:
                        notes[code] = out["data"]
                    elif fetch:
                        out["data"] = notes[code]
                term.clear()
                term.print(*tests_report(term, result, seconds))
                state = next_step([("Run the tests again", "after changing a model in models/", "tests"),
                                   ("Get tickets", "", "lottery"), ("Main menu", "", "home"), ("Quit", "", "quit")])
    term.print(term.style("  Good luck, and play within your budget.", term.style.dim), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(GAME_NAME))
    ap.add_argument("--tickets", type=int)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--goal", choices=sorted(GOALS), default="win")
    args = ap.parse_args()
    if args.game and args.goal not in GAME_GOALS[args.game]:
        ap.error(f"--goal {args.goal} is not offered for {GAME_NAME[args.game]}; choose from "
                 f"{', '.join(GAME_GOALS[args.game])}")
    if args.game and args.tickets:
        if not 1 <= args.tickets <= MAX_TICKETS:
            raise SystemExit(f"--tickets must be 1 to {MAX_TICKETS}")
        result, note = compute(args.game, args.tickets, args.seed, not args.no_update, lambda text: None, args.goal)
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
