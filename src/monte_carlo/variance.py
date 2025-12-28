"""Variance Calculation

Calculate variance across Monte Carlo simulation outcomes.
"""

import statistics
from dataclasses import dataclass
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class VarianceResult:
    """Result of variance calculation."""
    variance: float
    std_dev: float
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    action_consistency: float  # Fraction with same action
    confidence_range: float
    n_samples: int


def calculate_variance(outcomes: List[dict]) -> VarianceResult:
    """Calculate variance across simulation outcomes.

    Args:
        outcomes: List of outcome dicts from simulation

    Returns:
        VarianceResult with variance metrics
    """
    if not outcomes:
        return VarianceResult(
            variance=0.0,
            std_dev=0.0,
            mean_confidence=0.5,
            min_confidence=0.5,
            max_confidence=0.5,
            action_consistency=1.0,
            confidence_range=0.0,
            n_samples=0
        )

    # Extract confidences
    confidences = [o.get("confidence", 0.5) for o in outcomes]

    # Calculate statistics
    mean_conf = statistics.mean(confidences)

    if len(confidences) > 1:
        variance = statistics.variance(confidences)
        std_dev = statistics.stdev(confidences)
    else:
        variance = 0.0
        std_dev = 0.0

    min_conf = min(confidences)
    max_conf = max(confidences)
    conf_range = max_conf - min_conf

    # Calculate action consistency
    actions = [o.get("action_type", "CONTINUE") for o in outcomes]
    if actions:
        most_common = max(set(actions), key=actions.count)
        action_consistency = actions.count(most_common) / len(actions)
    else:
        action_consistency = 1.0

    return VarianceResult(
        variance=variance,
        std_dev=std_dev,
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        max_confidence=max_conf,
        action_consistency=action_consistency,
        confidence_range=conf_range,
        n_samples=len(outcomes)
    )


def normalized_variance(outcomes: List[dict]) -> float:
    """Calculate normalized variance (0-1 scale).

    Args:
        outcomes: Simulation outcomes

    Returns:
        Normalized variance (0 = no variance, 1 = max variance)
    """
    result = calculate_variance(outcomes)

    # Normalize: variance of uniform [0,1] is 1/12 ≈ 0.083
    # So max practical variance is around 0.25
    normalized = min(1.0, result.variance / 0.25)

    return normalized


def variance_by_action(outcomes: List[dict]) -> dict:
    """Calculate variance grouped by action type.

    Args:
        outcomes: Simulation outcomes

    Returns:
        Dict of action_type -> VarianceResult
    """
    by_action = {}

    for outcome in outcomes:
        action = outcome.get("action_type", "UNKNOWN")
        if action not in by_action:
            by_action[action] = []
        by_action[action].append(outcome)

    results = {}
    for action, action_outcomes in by_action.items():
        results[action] = calculate_variance(action_outcomes)

    return results


def confidence_distribution(outcomes: List[dict], bins: int = 10) -> dict:
    """Get distribution of confidence values.

    Args:
        outcomes: Simulation outcomes
        bins: Number of bins

    Returns:
        Dict with distribution info
    """
    confidences = [o.get("confidence", 0.5) for o in outcomes]

    if not confidences:
        return {"bins": [], "counts": [], "histogram": {}}

    # Create histogram
    bin_width = 1.0 / bins
    histogram = {i: 0 for i in range(bins)}

    for conf in confidences:
        bin_idx = min(bins - 1, int(conf / bin_width))
        histogram[bin_idx] += 1

    bin_labels = [f"{i/bins:.1f}-{(i+1)/bins:.1f}" for i in range(bins)]

    return {
        "bins": bin_labels,
        "counts": list(histogram.values()),
        "histogram": histogram,
        "total": len(confidences)
    }


def compare_variances(
    outcomes_a: List[dict],
    outcomes_b: List[dict]
) -> dict:
    """Compare variance between two sets of outcomes.

    Args:
        outcomes_a: First set of outcomes
        outcomes_b: Second set of outcomes

    Returns:
        Comparison dict
    """
    var_a = calculate_variance(outcomes_a)
    var_b = calculate_variance(outcomes_b)

    return {
        "variance_a": var_a.variance,
        "variance_b": var_b.variance,
        "variance_diff": var_b.variance - var_a.variance,
        "more_stable": "a" if var_a.variance < var_b.variance else "b",
        "consistency_a": var_a.action_consistency,
        "consistency_b": var_b.action_consistency
    }
