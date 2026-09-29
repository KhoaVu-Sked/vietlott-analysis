"""Test the idea that every universe has its own hash: its draws never repeat, so each combination already drawn is
gone and the rest share its chance equally. Looks for exact repeats in the history, weighs the idea against a fair
machine with a Bayes factor, and works out how much it would raise the jackpot chance if it were true.

Usage: python3 lottery/unique_hash_study.py [--no-update]
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "unique_hash_study.json"
GAMES = {"645": {"name": "Mega 6/45", "file": "mega645.jsonl", "balls": 45, "k": 6, "extra": None, "ways": 1,
                 "jackpot": math.comb(45, 6), "per_year": 156},
         "655": {"name": "Power 6/55", "file": "power655.jsonl", "balls": 55, "k": 6, "extra": "bonus", "ways": 49,
                 "jackpot": math.comb(55, 6), "per_year": 156},
         "535": {"name": "Lotto 5/35", "file": "lotto535.jsonl", "balls": 35, "k": 5, "extra": "special", "ways": 12,
                 "jackpot": math.comb(35, 5) * 12, "per_year": 730}}
KEEPS = (0.0, 0.1, 0.5)


def history(game):
    rows = [json.loads(line) for line in (ROOT / "data" / game["file"]).read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["id"])


def log_bayes_factor(keys, space, keep):
    seen, total = set(), 0.0
    for key in keys:
        if key in seen:
            if keep == 0:
                return -math.inf
            total += math.log(keep)
        total += math.log(space) - math.log(space - len(seen) * (1 - keep))
        seen.add(key)
    return total


def draws_for_factor(space, factor):
    return math.ceil((1 + math.sqrt(1 + 8 * space * math.log(factor))) / 2)


def check(keys, ids, space, game):
    first, repeats, expected = {}, [], 0.0
    for key, draw_id in zip(keys, ids):
        expected += len(first) / space
        if key in first:
            repeats.append((first[key], draw_id))
        else:
            first[key] = draw_id
    factor = space / (space - len(first))
    later = space / (space - len(first) - 10 * game["per_year"])
    return {"possible": space, "distinct_draws": len(first), "repeats": repeats, "fair_expects": expected,
            "bayes_factor": {str(w): math.exp(log_bayes_factor(keys, space, w)) for w in KEEPS},
            "jackpot_1_in_fair": game["jackpot"], "jackpot_1_in_if_true": game["jackpot"] / factor,
            "gain_pct_now": 100 * (factor - 1), "gain_pct_in_10_years": 100 * (later - 1),
            "no_repeat_evidence_2_at_draw": draws_for_factor(space, 2),
            "no_repeat_evidence_10_at_draw": draws_for_factor(space, 10)}


def number_chances(mains, n, k):
    distinct = set(mains)
    inside = Counter(i for combo in distinct for i in combo)
    unused = math.comb(n, k) - len(distinct)
    holding = math.comb(n - 1, k - 1)
    return {i: (holding - inside[i]) / unused for i in range(1, n + 1)}, inside


def factor_text(x):
    return "0 (ruled out)" if x == 0 else f"{x:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if not args.no_update:
        import update
        for code in GAMES:
            update.refresh(code)
    report = {}
    for code, game in GAMES.items():
        rows = history(game)
        n, k = game["balls"], game["k"]
        ids = [r["id"] for r in rows]
        mains = [tuple(sorted(r["result"])) for r in rows]
        versions = {"main numbers": check(mains, ids, math.comb(n, k), game)}
        if game["extra"]:
            fulls = [m + (r[game["extra"]],) for m, r in zip(mains, rows)]
            versions[f"main numbers + {game['extra']}"] = check(fulls, ids, math.comb(n, k) * game["ways"], game)
        print(f"{game['name']}: {len(rows):,} draws so far")
        for label, v in versions.items():
            if v["repeats"]:
                found = ", ".join(f"#{a} and #{b}" for a, b in v["repeats"])
                verdict = f"{len(v['repeats'])} exact repeat{'s' if len(v['repeats']) > 1 else ''} ({found}), the hash is broken"
            else:
                verdict = "no exact repeat, the hash holds"
            print(f"  {label}: {v['possible']:,} possible, {verdict}; a fair machine expects {v['fair_expects']:.3g} by now")
            bf = v["bayes_factor"]
            print(f"    Bayes factor, hash vs fair: never repeats {factor_text(bf['0.0'])},"
                  f" keeps 10% of its chance {factor_text(bf['0.1'])}, keeps half {factor_text(bf['0.5'])}")
            print(f"    if true, one jackpot ticket: 1 in {v['jackpot_1_in_if_true']:,.0f} instead of 1 in"
                  f" {v['jackpot_1_in_fair']:,} ({v['gain_pct_now']:+.4f}%); after 10 more years"
                  f" {v['gain_pct_in_10_years']:+.4f}%")
            if not v["repeats"]:
                two, ten = v["no_repeat_evidence_2_at_draw"], v["no_repeat_evidence_10_at_draw"]
                print(f"    if no repeat comes, the evidence reaches 2 at draw #{two:,}"
                      f" ({(two - len(rows)) / game['per_year']:.1f} years from now) and 10 at draw #{ten:,}"
                      f" ({(ten - len(rows)) / game['per_year']:.1f} years); one repeat rules it out")
        chances, inside = number_chances(mains, n, k)
        lo, hi = min(chances, key=chances.get), max(chances, key=chances.get)
        print(f"  each number under the hash: {100 * chances[lo]:.5f}% (no. {lo}, in {inside[lo]} past draws) to"
              f" {100 * chances[hi]:.5f}% (no. {hi}, in {inside[hi]}); fair {100 * k / n:.5f}%\n")
        report[code] = {"draws": len(rows), "versions": versions,
                        "number_chance_pct": {str(i): 100 * c for i, c in chances.items()}}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
