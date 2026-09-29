#!/usr/bin/env python3
"""Bring everything up to date in one go: fetch new draws for all three games, rerun every study and backtest, choose
the tickets and freeze the next Mega 6/45 and Power 6/55 predictions. Each step's output is saved under results/.

Usage: python3 update_all.py
       python3 update_all.py --quick      # skip the two full studies, which take most of the time
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lottery"))

import tui
import update

FIRST_WAVE = [
    ("Mega 6/45 full study", ["power645_study.py", "--no-update"], "latest_run.txt", True),
    ("Lotto 5/35 study and share-out forecast", ["lotto535_study.py", "--no-update"], "lotto535_run.txt", False),
    ("Engine study: diagnostics and fitted weights", ["engine_study.py", "--no-update"], "engine_study.txt", True),
    ("Model tests, all three games", ["backtest_models.py", "--no-update"], "backtest_models.txt", False),
    ("Forecast backtest, last 10 draws", ["backtest_checkpoints.py", "--count", "10"], "checkpoints_10x1.txt", False),
    ("Forecast backtest, last 300 draws", ["backtest_checkpoints.py", "--count", "300"], "checkpoints_300x1.txt", False),
    ("Mega 6/45 tickets, 5", ["optimise_tickets.py", "--tickets", "5", "--sims", "200000", "--no-update"],
     "tickets_5.txt", False),
    ("Mega 6/45 tickets, 7", ["optimise_tickets.py", "--tickets", "7", "--sims", "200000", "--no-update"],
     "tickets_7.txt", False),
    ("Mega 6/45 tickets, 10", ["optimise_tickets.py", "--tickets", "10", "--sims", "200000", "--no-update"],
     "tickets_10.txt", False),
]
SECOND_WAVE = [
    ("Power 6/55 full study, freezes its next prediction", ["power655_study.py", "--no-update"], "power655_run.txt",
     True),
    ("Freeze the next Mega 6/45 prediction", ["predict_draw.py", "--no-update"], "predict_draw.txt", False),
]
FINE_EXITS = ("already exists", "already happened", "needs numpy")


def run_step(step):
    label, argv, out_name, _ = step
    start = time.time()
    out = ROOT / "results" / out_name
    with out.open("w") as fh:
        code = subprocess.run([sys.executable, str(ROOT / "lottery" / argv[0]), *argv[1:]], cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT).returncode
    text = out.read_text()
    skipped = code != 0 and any(t in text for t in FINE_EXITS)
    return {"label": label, "code": code, "ok": code == 0 or skipped, "skipped": skipped,
            "seconds": time.time() - start, "file": f"results/{out_name}", "tail": text.strip().splitlines()[-1:]}


def run_wave(term, steps, begin):
    s = term.style
    results = []
    with ThreadPoolExecutor(max_workers=len(steps)) as pool:
        pending = {pool.submit(run_step, step) for step in steps}
        frame = 0
        while pending:
            done, pending = wait(pending, timeout=0.12)
            if term.interactive:
                term.write("\r\x1b[2K")
            for fut in done:
                r = fut.result()
                results.append(r)
                mark = s("✓", s.green) if r["ok"] else s("✗", s.red)
                extra = " (already frozen, kept as it was)" if r["skipped"] else ""
                term.print(f"  {mark} {r['label']}{extra} " + s(f"{r['seconds']:.0f}s → {r['file']}", s.dim))
                if not r["ok"]:
                    term.print(s(f"      exit {r['code']}: {' '.join(r['tail'])}", s.red))
            if pending and term.interactive:
                glyph = tui.FRAMES[frame % len(tui.FRAMES)]
                term.write(f"{s(glyph, s.accent)} {s(f'Running {len(pending)} step(s)…', s.accent)} "
                           f"{s(f'({time.time() - begin:.0f}s)', s.dim)}")
            frame += 1
    return results


def headlines():
    lines = []
    mega = ROOT / "results" / "latest.json"
    power = ROOT / "results" / "power655_latest.json"
    tests = ROOT / "results" / "result.json"
    if mega.exists():
        m = json.loads(mega.read_text())
        failed = sum(t["holm_p"] < 0.05 for t in m.get("tests", []))
        mf = m.get("ev", {}).get("scenarios", {}).get("model forecast")
        lines.append(f"Mega 6/45: {len(m.get('tests', [])) - failed} of {len(m.get('tests', []))} randomness tests pass"
                     + (f"; next jackpot ≈ {mf['jackpot']/1e9:.1f} tỷ, a ticket is worth {mf['ev']/10_000:.2f}× its price"
                        if mf else ""))
    if power.exists():
        p = json.loads(power.read_text())
        failed = sum(t["holm_p"] < 0.05 for t in p.get("tests", []))
        nxt = p.get("next", {})
        lines.append(f"Power 6/55: {len(p.get('tests', [])) - failed} of {len(p.get('tests', []))} randomness tests pass"
                     + (f"; next Jackpot 1 ≈ {nxt['j1']/1e9:.1f} tỷ, a ticket is worth"
                        f" {nxt['ev_random']['ev']/10_000:.2f}× its price" if nxt else ""))
    lotto = ROOT / "results" / "lotto535_latest.json"
    if lotto.exists():
        q = json.loads(lotto.read_text())
        failed = sum(t["holm_p"] < 0.05 for t in q.get("tests", []))
        nso = q.get("next_share_out")
        lines.append(f"Lotto 5/35: {len(q.get('tests', [])) - failed} of {len(q.get('tests', []))} randomness tests pass"
                     + (f"; next jackpot share-out most likely {nso['date']} {nso['time']}, forecast {nso['ev']/10_000:.2f}×"
                        f" the price after tax (past share-outs {q['share_outs']['after_tax']/10_000:.2f}×)" if nso else ""))
    if tests.exists():
        r = json.loads(tests.read_text())
        beat = [f"{name} ({out['game']})" for out in r.get("games", {}).values()
                for name, m in out["models"].items() if m["verdict"].startswith("beats")]
        lines.append("Model tests: " + (", ".join(beat) + " beat luck; freeze and test on live draws" if beat
                                        else "no model beats a random ticket"))
    return lines


def frozen():
    lines = []
    for game, name, pattern in (("645", "Mega 6/45", "draw_{:05d}.json"), ("655", "Power 6/55", "power655_draw_{:05d}.json")):
        data = ROOT / "data" / ("mega645.jsonl" if game == "645" else "power655.jsonl")
        last = json.loads([l for l in data.read_text().splitlines() if l.strip()][-1])
        path = ROOT / "predictions" / pattern.format(last["id"] + 1)
        if path.exists():
            lines.append(f"{name} #{last['id'] + 1}: {hashlib.sha256(path.read_bytes()).hexdigest()}")
        else:
            lines.append(f"{name} #{last['id'] + 1}: not frozen yet")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    term = tui.Terminal()
    s = term.style
    (ROOT / "results").mkdir(exist_ok=True)
    begin = time.time()
    term.print(s("✻ ", s.accent) + s("Updating everything", s.bold), "")
    for game, name in (("645", "Mega 6/45"), ("655", "Power 6/55"), ("535", "Lotto 5/35")):
        note = "not updated (--no-update)" if args.no_update else update.refresh(game)
        term.print(f"  {s('✓', s.green) if 'could not' not in note else s('!', s.yellow)} {name} data: {note}")
    results = []
    for wave in (FIRST_WAVE, SECOND_WAVE):
        steps = [step for step in wave if not (args.quick and step[3])]
        if steps:
            results += run_wave(term, steps, begin)
    term.print("", s("⏺ ", s.accent) + s("Headlines", s.bold))
    term.print(*[s("  ⎿  " if i == 0 else "     ", s.grey) + line for i, line in enumerate(headlines())])
    term.print("", s("⏺ ", s.accent) + s("Frozen predictions for the next draws (SHA-256)", s.bold))
    term.print(*[s("  ⎿  " if i == 0 else "     ", s.grey) + line for i, line in enumerate(frozen())])
    failed = [r for r in results if not r["ok"]]
    term.print("", s(f"  {len(results) - len(failed)} of {len(results)} steps finished in {time.time() - begin:.0f}s"
                     + (f"; {len(failed)} failed, see the files above" if failed else ""), s.red if failed else s.dim), "")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
