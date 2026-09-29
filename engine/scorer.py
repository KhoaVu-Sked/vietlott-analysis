"""Joint ensemble scorer. A ticket's composite is $w \\cdot f(T)$ with $f = (\\text{bayes}, \\text{markov},
\\text{gap}, \\text{entropy}, \\text{overlap})$. The weights are fitted by conditional logit: each past draw $D_t$ is
the alternative chosen among itself and $S$ sampled tickets, $\\log P(D_t \\mid w) = w \\cdot f_{t0} - \\log \\sum_s
e^{w \\cdot f_{ts}}$, maximised with a $\\mathcal{N}(0, \\sigma^2)$ prior on each weight by Newton's method with step
halving. Standard errors are $\\sqrt{\\mathrm{diag}((-H)^{-1})}$ at the optimum."""

import numpy as np

TERMS = ("bayes", "markov", "gap", "entropy", "overlap")


class Fit:
    """Fitted weights $\\hat w$ with standard errors $\\mathrm{se}_j = \\sqrt{((-H)^{-1})_{jj}}$ and $z_j = \\hat
    w_j / \\mathrm{se}_j$, taken as 0 where $\\mathrm{se}_j$ is not finite."""

    def __init__(self, w, se, draws_used, iterations, converged):
        self.w, self.se = np.asarray(w, dtype=float), np.asarray(se, dtype=float)
        self.draws_used, self.iterations, self.converged = draws_used, iterations, converged

    @property
    def z(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(self.se) & (self.se > 0), self.w / self.se, 0.0)

    def as_dict(self):
        return {name: {"w": float(w), "se": float(se), "z": float(z)}
                for name, w, se, z in zip(TERMS, self.w, self.se, self.z)}


class JointEnsembleScorer:
    """Composite score $s(T) = w \\cdot f(T)$ over the features in TERMS order, with $w = 0$ until a fit sets it."""

    def __init__(self, weights=None):
        self.w = np.zeros(len(TERMS)) if weights is None else np.asarray(weights, dtype=float)

    def score(self, features):
        return features @ self.w


def fit_weights(chosen, sampled, prior_sd=1.0, max_iter=50, tol=1e-8):
    features = np.concatenate([np.asarray(chosen)[:, None, :], np.asarray(sampled)], axis=1)
    draws, dim = features.shape[0], features.shape[2]
    ridge = 1 / prior_sd ** 2
    w = np.zeros(dim)

    def objective(w):
        u = features @ w
        top = u.max(axis=1)
        return float((u[:, 0] - top - np.log(np.exp(u - top[:, None]).sum(axis=1))).sum() - 0.5 * ridge * (w @ w))

    def gradient_hessian(w):
        u = features @ w
        p = np.exp(u - u.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean = np.einsum("ts,tsf->tf", p, features)
        g = (features[:, 0, :] - mean).sum(axis=0) - ridge * w
        h = -(np.einsum("ts,tsf,tsg->fg", p, features, features) - mean.T @ mean) - ridge * np.eye(dim)
        return g, h

    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        g, h = gradient_hessian(w)
        try:
            step = np.linalg.solve(h, -g)
        except np.linalg.LinAlgError:
            return Fit(np.zeros(dim), np.full(dim, np.inf), draws, iterations, False)
        base, scale = objective(w), 1.0
        while scale > 1e-6 and objective(w + scale * step) < base:
            scale /= 2
        w = w + scale * step
        if float(np.abs(scale * step).max()) < tol:
            converged = True
            break
    _, h = gradient_hessian(w)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(-h)))
    except np.linalg.LinAlgError:
        se = np.full(dim, np.inf)
    return Fit(w, se, draws, iterations, converged)
