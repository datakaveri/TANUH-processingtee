"""Shared helpers used by both the binary and multiclass metric implementations."""

from __future__ import annotations

from typing import Iterable


def safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising/NaN when denominator is 0.

    Every ratio metric here (precision, recall, specificity, ...) is undefined
    when its denominator is 0 (e.g. no negative samples at all, so specificity
    has no meaning). Scikit-learn would warn and emit NaN/0 inconsistently
    depending on the metric; we standardize on 0.0 so results.json always
    serializes to valid JSON and downstream consumers never see NaN.
    """
    return float(numerator) / float(denominator) if denominator else 0.0


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score from precision/recall directly, avoiding a second sklearn call."""
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    return safe_div((1 + beta_sq) * precision * recall, denom)


def unique_sorted_labels(*label_iterables: Iterable) -> list:
    """Union of all label values seen across the given iterables, sorted.

    Used to infer the multiclass label set (e.g. [1, 2, 3, 4]) when a dataset
    config doesn't pin one down explicitly.
    """
    seen = set()
    for values in label_iterables:
        seen.update(values)
    return sorted(seen)
