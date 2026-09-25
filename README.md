# Vietlott analysis: Mega 6/45 and Power 6/55

Collects every Vietlott Mega 6/45 and Power 6/55 draw from vietlott.vn, tests whether any prediction model beats a
random ticket, chooses tickets, and freezes each prediction before the draw so it can be reviewed honestly afterwards.

So far the draws behave exactly like a fair lottery: no model beats a random ticket. Two things do change the value of a
ticket: the size of the jackpot, and avoiding numbers other players like, so a jackpot is shared with fewer people.

Python 3.9 or newer, standard library only. Run every command from the repo root.

## Quick start

```bash
python3 prediction.py
```

A terminal session in the style of Claude Code: pick the game with the arrow keys and Enter, type how many tickets you
will buy, and it prints the tickets for the next draw with their chance of winning and their average value. Then pick
again, switch lottery or quit; Esc goes back. It checks vietlott.vn for new draws first. Skip the questions with
`python3 prediction.py --game 645 --tickets 10`; the last answer is saved to `results/last_prediction.txt`.

## Layout

| Folder | What is in it |
|---|---|
| `lottery/` | the scripts |
| `models/` | prediction models tested by `backtest_models.py`; one file per model |
| `data/` | every draw with prize and winner counts, collected from vietlott.vn |
| `predictions/` | frozen pre-draw predictions, their reviews, and the running scorecard |
| `results/` | generated output, not tracked except the log of model versions tried |

## Everyday commands

Update the data after a draw:

```bash
python3 lottery/collect_prizes.py
python3 lottery/collect_power655.py
```

Test every model in `models/` on every past draw, using only earlier draws each time. It writes `results/result.json`:

```bash
python3 lottery/backtest_models.py
python3 lottery/backtest_models.py --game 655
```

To add a model, copy `models/_template.py` to a new name and change `predict()`. A model only counts as real if it beats
a random ticket on all draws, on the hold-out of the last 300 draws, and then on new live draws.

Before a Mega 6/45 draw, sales close 17:45. Choose tickets, then freeze the prediction:

```bash
python3 lottery/optimise_tickets.py --tickets 10
python3 lottery/predict_draw.py
```

After the draw, compare the result with the frozen prediction:

```bash
python3 lottery/review_draw.py
```

For Power 6/55, the study freezes the next draw's prediction and the review compares it after the draw:

```bash
python3 lottery/power655_study.py
python3 lottery/review_power655.py
```

## Deeper studies

| Script | What it does |
|---|---|
| `power645_study.py` | 15 randomness tests, a Bayesian check, 9 models over 10 walk-forward runs, fake-lottery controls, sales and expected value |
| `power655_study.py` | the same for Power 6/55, plus Jackpot 2 and the bonus ball, a draw simulator and a long-run simulator |
| `backtest_checkpoints.py` | rebuilds the sales, jackpot and winner forecasts at past draws and scores them |
| `verify_tickets.py` | exact odds for a ticket set over all 8,145,060 possible 6/45 draws |

Playing the lottery loses money on average. These scripts measure the odds; they do not change them.
