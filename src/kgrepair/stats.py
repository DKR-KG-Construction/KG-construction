import math
import random


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion. Returns (p, lo, hi); (nan,nan,nan) if n == 0."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def bootstrap_ci(items, stat_fn, b=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over `items` (resampled with replacement). stat_fn(list)->float|None."""
    rng = random.Random(seed)
    vals = []
    n = len(items)
    for _ in range(b):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        v = stat_fn(sample)
        if v is not None and not math.isnan(v):
            vals.append(v)
    vals.sort()
    return percentile(vals, alpha / 2), percentile(vals, 1 - alpha / 2)


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists."""
    assert len(a) == len(b) and a
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd
