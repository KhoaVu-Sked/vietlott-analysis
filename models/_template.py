"""Copy this file to a new name in models/ (no leading underscore) and change predict().

predict() gets every draw BEFORE the one being predicted, oldest first, each a sorted list of 6 numbers, plus the
number of balls (45 or 55). It returns one score per number 1..n_balls: the 6 highest scores become the ticket.
Return non-negative scores that behave like chances if you also want a log-score; ties are broken at random.
"""


def predict(past, n_balls):
    return [1.0] * n_balls
