"""Run the numpy engine on each game: the three randomness diagnostics, the conditional-logit fit of the five ensemble
weights with error bars on the real history and on fake fair lotteries, the best-scored ticket and a 10-ticket greedy
portfolio compared with the covering design. Writes results/engine_study.json.

Usage: python3 lottery/engine_study.py [--fakes 2] [--seed 2026] [--no-update]
"""

import argparse
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_models as bm
import coverage
import prediction as app

try:
    import numpy as np
    from engine import Engine, candidates, diagnostics
except ImportError:
    np = None

OUT = ROOT / "results" / "engine_study.json"
POOL = 1_000_000
PORTFOLIO = 10


def fitted(draws, n, k, seed):
    eng = Engine(n, k, seed=seed)
    eng.extend(draws)
    eng.fit()
    return eng


def weights_line(fit):
    terms = fit.as_dict()
    parts = [f"{name} {v['w']:+.3f}±{v['se']:.3f} (z {v['z']:+.1f})" for name, v in terms.items()]
    flagged = [name for name, v in terms.items() if abs(v["z"]) > 3]
    verdict = ("all within error bars: nothing beyond a fair lottery" if not flagged
               else "beyond 3 errors: " + ", ".join(flagged))
    return " · ".join(parts), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()
    if np is None:
        raise SystemExit("engine_study needs numpy: pip3 install numpy")
    if not args.no_update:
        import update
        for code in bm.GAMES:
            update.refresh(code)
    report = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "games": {}}
    for code, game in bm.GAMES.items():
        n, k = game["balls"], game["k"]
        draws, _ = bm.load_game(code)
        chi = diagnostics.chi_square_uniformity_test(draws, n)
        runs = diagnostics.runs_test_autocorrelation(draws)
        ks = diagnostics.kolmogorov_smirnov_sum_test(draws, n, k)
        eng = fitted(draws, n, k, args.seed)
        if math.comb(n, k) <= candidates.MAX_ALL:
            pool, pool_note = candidates.all_tickets(n, k), f"all {math.comb(n, k):,} tickets"
        else:
            pool = candidates.sample(n, k, POOL, np.random.default_rng((args.seed, 1)))
            pool_note = f"{POOL:,} sampled tickets"
        best = eng.best(pool)
        port = eng.portfolio(PORTFOLIO, pool)
        design = coverage.design(PORTFOLIO, n, args.seed, size=k)
        p_port = app.simulated_any(port, n, args.seed, size=k)
        p_design = app.simulated_any(design, n, args.seed, size=k)
        fakes = []
        for f in range(args.fakes):
            r = random.Random(args.seed + 500 + f)
            fake = [sorted(r.sample(range(1, n + 1), k)) for _ in draws]
            fakes.append(fitted(fake, n, k, args.seed).fit_result.as_dict())
        line, verdict = weights_line(eng.fit_result)
        print(f"{game['name']}: {len(draws):,} draws")
        print(f"  chi-square uniformity {chi[0]:.1f} (p {chi[1]:.3f}) · runs test z {runs[0]:+.2f} (p {runs[1]:.3f})"
              f" · KS on sums D {ks[0]:.4f} (p {ks[1]:.3f})")
        print(f"  fitted weights on {eng.fit_result.draws_used:,} draws: {line}")
        print(f"  verdict: {verdict}")
        for i, fake in enumerate(fakes, 1):
            print(f"  fake lottery {i}: " + " · ".join(f"{name} z {v['z']:+.1f}" for name, v in fake.items()))
        print(f"  best ticket over {pool_note}: {' '.join(f'{x:02d}' for x in best)}")
        print(f"  {PORTFOLIO}-ticket portfolio (overlap ≤ {k - 2}): any prize {p_port * 100:.2f}% vs covering design"
              f" {p_design * 100:.2f}% ({app.SIMS:,} simulated draws)")
        for t in port:
            print("      " + " ".join(f"{x:02d}" for x in t))
        print()
        report["games"][code] = {
            "draws": len(draws),
            "diagnostics": {"chi_square": {"stat": chi[0], "p": chi[1]}, "runs": {"z": runs[0], "p": runs[1]},
                            "ks_sums": {"d": ks[0], "p": ks[1]}},
            "fit": eng.fit_result.as_dict(), "fit_draws": eng.fit_result.draws_used, "verdict": verdict,
            "fakes": fakes, "candidates": pool_note, "best_ticket": best, "portfolio": port,
            "p_any_portfolio": p_port, "p_any_design": p_design}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
