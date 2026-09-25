def predict(past, n_balls):
    waited = [len(past) + 1.0] * n_balls
    for age, draw in enumerate(reversed(past), 1):
        for x in draw:
            if waited[x - 1] > age:
                waited[x - 1] = float(age)
    return waited
