def predict(past, n_balls):
    scores = [1.0] * n_balls
    if past:
        for x in past[-1]:
            scores[x - 1] = 2.0
    return scores
