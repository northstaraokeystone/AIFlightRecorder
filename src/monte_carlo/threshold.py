"""Stability Threshold Checking

Determine if variance is acceptable for decision stability.
"""

from dataclasses import dataclass
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import MONTE_CARLO_VARIANCE_THRESHOLD
from .variance import calculate_variance, VarianceResult


@dataclass
class StabilityResult:
    """Result of stability check."""
    is_stable: bool
    variance: float
    threshold: float
    variance_result: VarianceResult
    confidence_adjustment: float  # How much to adjust confidence
    recommendation: str


def check_stability(
    outcomes: List[dict],
    threshold: float = MONTE_CARLO_VARIANCE_THRESHOLD
) -> StabilityResult:
    """Determine if variance is acceptable for stability.

    Args:
        outcomes: Monte Carlo simulation outcomes
        threshold: Maximum acceptable variance (default 0.2)

    Returns:
        StabilityResult
    """
    variance_result = calculate_variance(outcomes)

    is_stable = variance_result.variance <= threshold
    confidence_adjustment = 0.0
    recommendation = "proceed"

    if not is_stable:
        # Calculate confidence penalty
        excess_variance = variance_result.variance - threshold
        confidence_adjustment = -min(0.3, excess_variance * 0.5)
        recommendation = "reduce_confidence"

        if variance_result.variance > threshold * 2:
            recommendation = "investigate"

        if variance_result.action_consistency < 0.7:
            recommendation = "high_uncertainty"

    return StabilityResult(
        is_stable=is_stable,
        variance=variance_result.variance,
        threshold=threshold,
        variance_result=variance_result,
        confidence_adjustment=confidence_adjustment,
        recommendation=recommendation
    )


def adaptive_threshold(
    base_threshold: float,
    context: dict
) -> float:
    """Calculate adaptive threshold based on context.

    More critical contexts get tighter thresholds.

    Args:
        base_threshold: Base variance threshold
        context: Decision context

    Returns:
        Adjusted threshold
    """
    threshold = base_threshold

    # Safety-critical contexts need tighter thresholds
    action_type = context.get("action_type", "CONTINUE")
    if action_type in ["ABORT", "RTB"]:
        threshold *= 0.5  # Half the threshold for critical actions

    if action_type == "ENGAGE":
        threshold *= 0.75  # Stricter for engagement

    # High-threat environments
    threats = context.get("threats", [])
    if threats:
        max_threat = max((t.get("severity", 0) for t in threats), default=0)
        if max_threat > 0.7:
            threshold *= 0.5

    # Low battery
    battery = context.get("battery_pct", 100)
    if battery < 20:
        threshold *= 0.75

    return threshold


def stability_score(outcomes: List[dict]) -> float:
    """Calculate a single stability score (0-1).

    Higher = more stable.

    Args:
        outcomes: Simulation outcomes

    Returns:
        Stability score
    """
    if not outcomes:
        return 1.0

    variance_result = calculate_variance(outcomes)

    # Combine variance and action consistency
    variance_factor = 1.0 - min(1.0, variance_result.variance / 0.25)
    consistency_factor = variance_result.action_consistency

    # Weighted combination
    score = 0.6 * variance_factor + 0.4 * consistency_factor

    return max(0.0, min(1.0, score))


def stability_grade(outcomes: List[dict]) -> str:
    """Get letter grade for stability.

    Args:
        outcomes: Simulation outcomes

    Returns:
        Grade A-F
    """
    score = stability_score(outcomes)

    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"


def should_retry_simulation(
    outcomes: List[dict],
    current_n: int,
    max_n: int = 200
) -> tuple[bool, int]:
    """Check if more simulations would improve stability estimate.

    Args:
        outcomes: Current outcomes
        current_n: Current number of sims
        max_n: Maximum sims to run

    Returns:
        Tuple of (should_retry, suggested_n)
    """
    if current_n >= max_n:
        return False, current_n

    variance_result = calculate_variance(outcomes)

    # High variance might benefit from more samples
    if variance_result.variance > 0.15 and current_n < 50:
        return True, min(max_n, current_n * 2)

    # Borderline stability might need confirmation
    if 0.18 <= variance_result.variance <= 0.22:
        return True, min(max_n, current_n + 50)

    return False, current_n


def combined_stability_assessment(
    variance_result: VarianceResult,
    threshold: float = MONTE_CARLO_VARIANCE_THRESHOLD
) -> dict:
    """Get comprehensive stability assessment.

    Args:
        variance_result: Variance calculation result
        threshold: Stability threshold

    Returns:
        Assessment dict
    """
    is_stable = variance_result.variance <= threshold

    # Determine confidence level
    if variance_result.variance <= threshold * 0.5:
        confidence_level = "high"
    elif variance_result.variance <= threshold:
        confidence_level = "moderate"
    elif variance_result.variance <= threshold * 1.5:
        confidence_level = "low"
    else:
        confidence_level = "very_low"

    # Risk assessment
    if variance_result.action_consistency < 0.5:
        risk = "high"
    elif variance_result.action_consistency < 0.75:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "is_stable": is_stable,
        "variance": variance_result.variance,
        "threshold": threshold,
        "confidence_level": confidence_level,
        "risk": risk,
        "action_consistency": variance_result.action_consistency,
        "recommendation": _get_recommendation(is_stable, confidence_level, risk)
    }


def _get_recommendation(is_stable: bool, confidence_level: str, risk: str) -> str:
    """Generate recommendation based on assessment."""
    if is_stable and confidence_level == "high" and risk == "low":
        return "Proceed with original confidence"
    elif is_stable and confidence_level == "moderate":
        return "Proceed with slight confidence reduction"
    elif not is_stable and risk == "low":
        return "Reduce confidence and monitor"
    elif risk == "high":
        return "Investigate before proceeding"
    else:
        return "Apply confidence penalty"
