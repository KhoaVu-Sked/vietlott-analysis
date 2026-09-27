# Vietlott analysis: Mega 6/45, Power 6/55 and Lotto 5/35

Collects every Vietlott Mega 6/45, Power 6/55 and Lotto 5/35 draw from vietlott.vn, tests whether any prediction model
beats a random ticket, chooses tickets, and freezes each prediction before the draw so it can be reviewed honestly
afterwards.

So far the draws behave exactly like a fair lottery: no model beats a random ticket. Two things do change the value of a
ticket: the size of the jackpot, and avoiding numbers other players like, so a jackpot is shared with fewer people.

Python 3.9 or newer, standard library only. Run every command from the repo root.

## The one real edge found: the Lotto 5/35 jackpot share-out

Lotto 5/35 has a rule the other games lack ("Chia Giải Độc Đắc"): when the jackpot passes 12 tỷ without a winner, the
last draw of the next day shares the whole jackpot out to the 3, 4 and 5-correct winners, unless someone wins it first.
It is the same kind of rule quirk the Selbees and the MIT group used in Michigan and Massachusetts. On the 21 share-out
draws so far a ticket returned 1.25× its price on average after tax, against about 0.35× on an ordinary draw.

The crowd has found it: share-out draws now sell about 18 times the usual tickets, and the return fell from 2.38× in
July 2025 to about 1.0× on the last two. `lotto535_study.py` forecasts the next share-out date and value; the app shows
both. Every draw is still random: nothing here predicts numbers.

## Quick start

```bash
python3 prediction.py
```

One command for everything, in a terminal session styled like Claude Code. It checks vietlott.vn for new draws on its
own, so there is nothing to fetch by hand. The main menu offers:

- **Get tickets**: pick the game with the arrow keys, choose what matters most, type how many tickets you will buy, and
  get the numbers for the next draw with their chance of winning and their average value.
  - *Most chance to win* (the default): tickets spread over every number with the least overlap, a covering design.
    Two tickets that share numbers tend to win on the same draws, so overlap wastes tickets. For 10 Mega 6/45 tickets this
    gives a 23.0% chance of a prize, against 22.6% for best value and 21.5% for random quick picks; 10 tickets can never
    pass 23.8%, ten times one ticket's 2.38%. The jackpot chance is the same for any different tickets.
  - *Best value*: tickets that avoid numbers other players like, so a jackpot would be shared with fewer people.
- **Test models**: run every model in `models/` on every past draw of all three games and show how each compares
  with luck.

Esc goes back a step. Without questions: `python3 prediction.py --game 645 --tickets 10` for tickets (`--goal value` for best value), and
`python3 lottery/backtest_models.py` for the model tests.

## Layout

| Folder | What is in it |
|---|---|
| `lottery/` | the scripts |
| `models/` | prediction models tested by `backtest_models.py`; one file per model |
| `data/` | every draw with prize and winner counts, collected from vietlott.vn |
| `predictions/` | frozen pre-draw predictions, their reviews, and the running scorecard |
| `results/` | generated output, not tracked except the log of model versions tried |

## Update everything at once

```bash
python3 update_all.py
```

Fetches new draws for all three games, then reruns the full studies, the model tests, the forecast backtests and the ticket
choice, and freezes the next Mega 6/45 and Power 6/55 predictions. It takes about 5 minutes; `--quick` skips the two slow
studies. Nothing is stored between runs: every model is fitted again from `data/` each time a script runs.

## Freeze, review and model testing

These cover what `prediction.py` does not: freezing a prediction before a draw and reviewing it afterwards. Every
command fetches new draws itself (`--no-update` skips it); the collectors only update the data without doing anything else.

```bash
python3 lottery/collect_prizes.py
python3 lottery/collect_power655.py
python3 lottery/collect_lotto535.py
```

Test every model in `models/` on every past draw of all three games, using only earlier draws each time. It writes
`results/result.json`; add `--game 645`, `--game 655` or `--game 535` for one game, or `--no-update` to skip fetching:

```bash
python3 lottery/backtest_models.py
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

## The models

Every model gets the draws before the one it predicts and returns a score per number; the 6 highest become its ticket.
`backtest_models.py` scores each on every past draw by hit rate and by log-score, the proper scoring rule from Bayes
Rules! ch. 10-11 (0 = as good as a random ticket, below 0 = worse).

| File in `models/` | Idea | Where it comes from |
|---|---|---|
| `fair.py` | every number has the same chance | complete pooling, Bayes Rules! ch. 15; the luck baseline |
| `frequency.py` | numbers drawn most often since the start | no pooling, ch. 15; law of large numbers |
| `hierarchical_bayes.py` | the same counts, pulled toward fair by as much as the data choose | partial pooling, ch. 15-16; Beta-Binomial, ch. 3-4 |
| `hot30.py` | numbers drawn most in the last 30 draws | |
| `recent_bayes.py` | the last 30 draws, pooled toward fair | Gamma-Poisson and Beta-Binomial conjugacy, ch. 3-5 |
| `ewma_hot.py` | every past appearance, fading 3% per draw | |
| `overdue.py` | numbers missing the longest | the gambler's fallacy |
| `gap_hazard.py` | learns the chance of a number returning after each gap length | Bernoulli trials: the wait is geometric and memoryless |
| `repeat_last.py` | the previous draw's numbers | |
| `markov_chain.py` | numbers that have followed the previous draw's numbers | Markov chains |
| `logistic_regression.py` | chance from recent, long-run and gap signals | ch. 13, Normal prior on the slopes from ch. 9 |
| `naive_bayes.py` | the same signals combined with Bayes' rule | ch. 14 |
| `bayes_average.py` | Bayes' rule over seven models, reweighted after every draw | ch. 2 and 10-11 |
| `bayes_average_forgetful.py` | the same, but old evidence fades 5% per draw so the weights can follow a changing best lens | dynamic model averaging |
| `frames_follow_last.py` | eight lenses on the same draws; use the one that caught the most numbers last draw | |
| `frames_follow_10.py` | use the lens with the most hits over the last 10 draws | |
| `frames_contrarian_10.py` | use the lens with the fewest hits over the last 10 draws | |
| `frames_vote.py` | the numbers most of the eight lenses agree on | |

Each model is a different lens on the same draws, and every draw some lens wins: the winning lens changes from draw
to draw exactly as luck predicts, so the lens-switching models can't pick the next winner in advance either.

Why none beats a random ticket: if each draw is a fresh random pick, any 6 numbers chosen from the past catch
6 × 6/45 = 0.8 numbers on average, however clever the method. The data leave room for only a tiny bias: at the largest
per-number bias still consistent with 1,567 draws, even a perfect model would catch about 0.83 numbers and win back about
1,500 VND of fixed prizes per 10,000 VND ticket instead of 1,370.

## Deeper studies

| Script | What it does |
|---|---|
| `power645_study.py` | 15 randomness tests, a Bayesian check, 9 models over 10 walk-forward runs, fake-lottery controls, sales and expected value |
| `power655_study.py` | the same for Power 6/55, plus Jackpot 2 and the bonus ball, a draw simulator and a long-run simulator |
| `backtest_checkpoints.py` | rebuilds the sales, jackpot and winner forecasts at past draws and scores them |
| `lotto535_study.py` | Lotto 5/35: odds with the special number, 16 randomness tests, sales, every jackpot share-out and what it paid, the next share-out forecast, tickets and a long-run replay |
| `verify_tickets.py` | exact odds for a ticket set over all 8,145,060 possible 6/45 draws |

Playing the lottery loses money on average. These scripts measure the odds; they do not change them.
