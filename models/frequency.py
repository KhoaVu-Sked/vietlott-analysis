def predict(past, n_balls):
    counts = [1.0] * n_balls
    for draw in past:
        for x in draw:
            counts[x - 1] += 1
    return counts
