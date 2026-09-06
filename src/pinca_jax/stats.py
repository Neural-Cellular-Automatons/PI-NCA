"""Statistics for headline claims: intervals, paired tests, multiple-comparison control.

A benchmark table of point estimates cannot support a sentence like "architecture A
beats architecture B on this PDE". Two things are needed and both are provided here:

1. **Uncertainty on each number.** `summarize` reports mean, std, median, IQR and a
   percentile bootstrap CI. With n seeds in the single digits the bootstrap percentile
   interval is the honest default; a normal-theory interval on 5 points is not.
2. **Paired comparisons.** Two architectures are evaluated on the *same* initial
   conditions with the *same* seeds, so the samples are paired and the correct test is
   on the differences, not on two independent means. `paired_test` reports the mean
   paired difference, its bootstrap CI, the Wilcoxon signed-rank p-value (no normality
   assumption) and the win rate.

Every comparison in the paper routes through `compare_archs`, which also applies
Holm-Bonferroni across the family of comparisons made per PDE, so "A beats B" is never
claimed from an uncorrected p-value in a table of a dozen models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


# ---------------------------------------------------------------- intervals ---
def bootstrap_ci(values, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0,
                 stat=np.mean) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over `values`.

    Deterministic: the resampling RNG is seeded, so a reported interval is
    reproducible from the same raw numbers.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float          # sample std (ddof=1); 0.0 when n == 1
    sem: float          # standard error of the mean
    median: float
    iqr: float
    ci_lo: float
    ci_hi: float
    vmin: float
    vmax: float

    def __str__(self):
        return (f"{self.mean:.4e} +/- {self.std:.2e} "
                f"[{self.ci_lo:.3e}, {self.ci_hi:.3e}] (n={self.n})")

    def as_dict(self):
        return asdict(self)


def summarize(values, alpha: float = 0.05, seed: int = 0) -> Summary:
    """Point estimate + spread + bootstrap CI for one metric across seeds/ICs."""
    v = np.asarray([float(x) for x in values], dtype=float)
    finite = v[np.isfinite(v)]
    n = int(finite.size)
    if n == 0:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    sd = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(finite, [25, 75])
    lo, hi = bootstrap_ci(finite, alpha=alpha, seed=seed)
    return Summary(n=n, mean=float(np.mean(finite)), std=sd,
                   sem=sd / math.sqrt(n) if n > 1 else 0.0,
                   median=float(np.median(finite)), iqr=float(q3 - q1),
                   ci_lo=lo, ci_hi=hi,
                   vmin=float(np.min(finite)), vmax=float(np.max(finite)))


# ------------------------------------------------------------ paired tests ---
def paired_test(a, b, alpha: float = 0.05, seed: int = 0, lower_is_better: bool = True):
    """Is `a` better than `b` on paired samples? (same ICs, same seeds, same order)

    Returns the mean paired difference d = a - b with a bootstrap CI on the *paired*
    differences, the Wilcoxon signed-rank p-value, and the fraction of pairs on which
    `a` wins. `significant` requires BOTH that the CI excludes zero and that Wilcoxon
    rejects at `alpha` -- an agreement rule, not a p-value hunt.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired_test needs equal-length samples, got {x.shape} vs {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "median_diff": float("nan"),
                "p_value": 1.0, "significant": False, "win_rate": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "better": "undetermined"}
    d = x - y
    lo, hi = bootstrap_ci(d, alpha=alpha, seed=seed)
    p = _wilcoxon_p(d)
    wins = float(np.mean(d < 0) if lower_is_better else np.mean(d > 0))
    ci_excludes_zero = (lo > 0) or (hi < 0)
    sig = bool(ci_excludes_zero and p < alpha)
    if not sig:
        better = "tie"
    elif (float(np.mean(d)) < 0) == lower_is_better:
        better = "a"
    else:
        better = "b"
    return {"n": n, "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
            "ci_lo": lo, "ci_hi": hi, "p_value": float(p), "significant": sig,
            "win_rate": wins, "better": better}


def _wilcoxon_p(d) -> float:
    """Wilcoxon signed-rank p (two-sided). 1.0 when the test is undefined."""
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    if nz.size < 6:          # too few nonzero pairs for any power; do not pretend
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(nz, alternative="two-sided").pvalue)
    except Exception:        # scipy absent -> sign test, strictly more conservative
        from math import comb
        k = int(np.sum(nz > 0))
        n = int(nz.size)
        tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
        return float(min(1.0, 2 * tail))


def holm_bonferroni(pvals, alpha: float = 0.05):
    """Holm step-down: returns a boolean list, True = reject at family-wise `alpha`.

    A per-PDE table compares ~12 architectures; testing each pair at 0.05 would produce
    a false "win" by construction. Holm controls the family-wise error rate without the
    power loss of plain Bonferroni.
    """
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break                       # step-down stops at the first failure
    return [bool(r) for r in reject]


def compare_archs(per_arch_samples: dict, reference: str | None = None,
                  alpha: float = 0.05, lower_is_better: bool = True) -> dict:
    """Paired comparison of every architecture against a reference, Holm-corrected.

    `per_arch_samples`: {arch: [per-IC (or per-seed) metric values]}, all in the SAME
    order so entries at index i share an initial condition. `reference` defaults to the
    architecture with the best mean.
    """
    names = [k for k, v in per_arch_samples.items() if len(v) > 0]
    if not names:
        return {"reference": None, "comparisons": {}}
    if reference is None:
        means = {k: float(np.nanmean(per_arch_samples[k])) for k in names}
        reference = min(means, key=means.get) if lower_is_better else max(means, key=means.get)
    others = [k for k in names if k != reference]
    tests = {k: paired_test(per_arch_samples[k], per_arch_samples[reference],
                            alpha=alpha, lower_is_better=lower_is_better)
             for k in others}
    rejects = holm_bonferroni([tests[k]["p_value"] for k in others], alpha=alpha)
    for k, r in zip(others, rejects):
        tests[k]["holm_significant"] = bool(r and tests[k]["significant"])
    return {"reference": reference, "alpha": alpha, "n_comparisons": len(others),
            "comparisons": tests}


def fmt_ci(s: Summary, sig: int = 3) -> str:
    """Compact 'mean+/-std [ci_lo, ci_hi]' for Markdown tables."""
    return f"{s.mean:.{sig}g}+/-{s.std:.2g} [{s.ci_lo:.{sig}g},{s.ci_hi:.{sig}g}]"


def demo():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 40)
    b = a + rng.normal(0.3, 0.1, 40)          # b is worse (higher) by ~0.3
    s = summarize(a)
    assert s.n == 40 and s.ci_lo < s.mean < s.ci_hi
    t = paired_test(a, b, lower_is_better=True)
    assert t["better"] == "a" and t["significant"] and t["win_rate"] > 0.9
    same = paired_test(a, a + rng.normal(0, 1e-12, 40))
    assert same["better"] == "tie", same
    assert holm_bonferroni([0.001, 0.04, 0.9]) == [True, False, False]
    assert holm_bonferroni([]) == []
    c = compare_archs({"good": list(a), "bad": list(b)})
    assert c["reference"] == "good" and c["comparisons"]["bad"]["holm_significant"]
    print("stats.demo OK")


if __name__ == "__main__":
    demo()
