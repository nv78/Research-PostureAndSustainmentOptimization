"""Statistical rigor utilities for postureopt experiments.

Provides confidence intervals, significance testing, Bonferroni correction,
two-level variance decomposition, Cohen's d effect sizes, percentile bootstrap
CIs, Shapiro-Wilk normality checks, and post-hoc power analysis.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def confidence_interval(
    samples: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute a confidence interval using the t-distribution.

    Uses the t-distribution rather than the normal approximation, which is
    appropriate when sample sizes are small (n < 30).

    Returns a dict with keys:
        mean, lower, upper, margin, n, confidence
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError("Need at least 2 samples to compute a confidence interval.")
    mean = float(arr.mean())
    se = float(_scipy_stats.sem(arr))
    t_crit = float(_scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * se
    return {
        "mean": mean,
        "lower": mean - margin,
        "upper": mean + margin,
        "margin": margin,
        "n": n,
        "confidence": confidence,
    }


def ci_str(samples: Sequence[float], confidence: float = 0.95, decimals: int = 3) -> str:
    """Return a compact CI string: 'mean (lower, upper)'."""
    ci = confidence_interval(samples, confidence)
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(ci['mean'])} ({fmt.format(ci['lower'])}, {fmt.format(ci['upper'])})"


def ci_latex(samples: Sequence[float], confidence: float = 0.95, decimals: int = 3) -> str:
    """Return a LaTeX-formatted CI string: 'mean~(lower, upper)'."""
    ci = confidence_interval(samples, confidence)
    fmt = f"{{:.{decimals}f}}"
    return (
        f"{fmt.format(ci['mean'])}~"
        f"({fmt.format(ci['lower'])}, {fmt.format(ci['upper'])})"
    )


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------


def paired_ttest(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired two-tailed t-test comparing samples_a vs. samples_b.

    Appropriate when each pair (a_i, b_i) comes from the same random seed,
    as is the case when comparing greedy vs. CEV under matched conditions.

    Returns a dict with keys:
        t_stat, p_value, mean_diff, reject_null, alpha
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("samples_a and samples_b must have the same length.")
    t_stat, p_value = _scipy_stats.ttest_rel(a, b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float((a - b).mean()),
        "reject_null": bool(p_value < alpha),
        "alpha": alpha,
    }


def bonferroni_correct(alpha: float, n_tests: int) -> float:
    """Return the Bonferroni-corrected significance threshold.

    With n_tests simultaneous comparisons, the per-test alpha is divided
    by n_tests to control the family-wise error rate at the nominal level.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1.")
    return alpha / n_tests


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


def variance_decomposition(data: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Two-level variance decomposition for nested Monte Carlo designs.

    Decomposes total variance into:
      - Outer variance: variance of group means across scenario seeds
        (captures sensitivity to the choice of scenario set)
      - Inner variance: mean within-group variance across simulation seeds
        (captures sensitivity to initial-state randomness)

    Parameters
    ----------
    data : array-like of shape (n_outer, n_inner)
        E.g., rows = different scenario seeds, columns = simulation seeds.

    Returns a dict with keys:
        outer_var, inner_var, total_var, icc, n_outer, n_inner
    where icc (intraclass correlation) = outer_var / (outer_var + inner_var).
    An ICC near 1 means scenario-set choice dominates; near 0 means
    initial-state randomness dominates.
    """
    arr = np.array(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be 2-dimensional: (n_outer, n_inner).")
    n_outer, n_inner = arr.shape

    group_means = arr.mean(axis=1)
    outer_var = float(np.var(group_means, ddof=1)) if n_outer > 1 else 0.0

    within_vars = np.var(arr, axis=1, ddof=1) if n_inner > 1 else np.zeros(n_outer)
    inner_var = float(within_vars.mean())

    total_var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0

    denom = outer_var + inner_var
    icc = outer_var / denom if denom > 0 else 0.0

    return {
        "outer_var": outer_var,
        "inner_var": inner_var,
        "total_var": total_var,
        "icc": icc,
        "n_outer": n_outer,
        "n_inner": n_inner,
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d_paired(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
) -> float:
    """Cohen's d for paired samples: mean(a - b) / SD(a - b).

    Returns +inf / -inf when SD of differences is zero (deterministic gap).
    Convention: |d| >= 0.8 is a large effect (Cohen 1988).
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        mean_d = float(diff.mean())
        return math.copysign(math.inf, mean_d) if mean_d != 0.0 else 0.0
    return float(diff.mean() / sd)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci_paired(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean paired difference (a - b).

    Uses resampling with replacement on the n paired differences.
    Seed-fixed for reproducibility; set seed=None for a fresh draw.

    Returns a dict with keys:
        mean, lower, upper, n_bootstrap, confidence
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(diffs.mean()),
        "lower": lo,
        "upper": hi,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Normality check
# ---------------------------------------------------------------------------


def shapiro_wilk_paired(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
) -> Dict[str, float]:
    """Shapiro-Wilk normality test on the paired differences (a - b).

    Returns a dict with keys: W, p_value, n, normal (True if p > 0.05).
    Not meaningful when differences have zero variance.
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    diffs = a - b
    if float(np.std(diffs)) == 0.0:
        return {"W": float("nan"), "p_value": float("nan"), "n": len(diffs), "normal": True}
    W, p = _scipy_stats.shapiro(diffs)
    return {"W": float(W), "p_value": float(p), "n": len(diffs), "normal": bool(p > 0.05)}


# ---------------------------------------------------------------------------
# Statistical power (post-hoc, paired two-tailed t-test)
# ---------------------------------------------------------------------------


def statistical_power_paired(
    cohens_d: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Post-hoc power for a paired two-tailed t-test via the noncentral t-distribution.

    noncentrality parameter λ = |d| * sqrt(n).
    Returns 1.0 for infinite d (deterministic comparisons).
    """
    if math.isinf(cohens_d) or math.isnan(cohens_d):
        return 1.0
    ncp = abs(cohens_d) * math.sqrt(n)
    df = n - 1
    t_crit = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df=df))
    power = (
        1.0
        - float(_scipy_stats.t.cdf(t_crit, df=df, loc=ncp))
        + float(_scipy_stats.t.cdf(-t_crit, df=df, loc=ncp))
    )
    return float(np.clip(power, 0.0, 1.0))
