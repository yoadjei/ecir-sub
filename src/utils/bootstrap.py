"""resampling stats, every function takes a seed so results repeat."""
import numpy as np


def bootstrap_ci(values, n_resamples: int = 10000, seed: int = 13, alpha: float = 0.05):
    """percentile bootstrap ci for the mean, returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, n_resamples: int = 10000, seed: int = 13, alpha: float = 0.05):
    """ci for mean(a) - mean(b), each group resampled on its own (unpaired)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_resamples, len(a)))
    ib = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def bootstrap_group_means_diff(groups_a: list, groups_b: list, n_resamples: int = 10000, seed: int = 13, alpha: float = 0.05):
    """ci for mean of language means (a) minus mean of language means (b).

    groups are lists of per language arrays of query values. queries are resampled
    inside each language, then the language means are averaged with equal weight."""
    rng = np.random.default_rng(seed)
    ga = [np.asarray(g, dtype=float) for g in groups_a]
    gb = [np.asarray(g, dtype=float) for g in groups_b]

    def one(gs):
        return np.mean([g[rng.integers(0, len(g), size=len(g))].mean() for g in gs])

    obs = float(np.mean([g.mean() for g in ga]) - np.mean([g.mean() for g in gb]))
    diffs = np.array([one(ga) - one(gb) for _ in range(n_resamples)])
    return obs, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def paired_randomisation_p(x, y, n_perm: int = 10000, seed: int = 13) -> float:
    """two sided sign flip test on the paired differences x - y."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))


def holm(pvals) -> list[float]:
    """holm step down adjusted p values, in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()
