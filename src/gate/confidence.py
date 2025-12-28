"""Confidence Score Calculation

Calculate decision stability score based on decision, context, and history.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    MONTE_CARLO_VARIANCE_THRESHOLD,
    WOUND_DROP_THRESHOLD
)


@dataclass
class ConfidenceScore:
    """Complete confidence assessment."""
    raw_confidence: float        # Original decision confidence
    variance_adjusted: float     # After Monte Carlo adjustment
    drift_adjusted: float        # After context drift adjustment
    final_score: float           # Final confidence score (0-1)
    components: dict             # Breakdown of score components
    is_stable: bool              # Whether decision is stable


def calculate_confidence(
    decision: dict,
    context: Optional[dict] = None,
    history: Optional[list[dict]] = None,
    monte_carlo_variance: Optional[float] = None,
    drift_score: Optional[float] = None
) -> ConfidenceScore:
    """Calculate decision stability score.

    Args:
        decision: The decision to evaluate
        context: Current context (environment, state)
        history: Recent decision history
        monte_carlo_variance: Variance from Monte Carlo simulation (0-1)
        drift_score: Context drift score (0-1)

    Returns:
        ConfidenceScore with breakdown
    """
    # Extract raw confidence from decision
    if "full_decision" in decision:
        raw_confidence = decision["full_decision"].get("confidence", 0.5)
    else:
        raw_confidence = decision.get("confidence", 0.5)

    # Component scores
    components = {
        "raw_confidence": raw_confidence,
        "monte_carlo_penalty": 0.0,
        "drift_penalty": 0.0,
        "history_stability": 1.0,
        "reasoning_entropy": 0.0
    }

    # 1. Apply Monte Carlo variance penalty
    variance_adjusted = raw_confidence
    if monte_carlo_variance is not None:
        # High variance reduces confidence
        if monte_carlo_variance > MONTE_CARLO_VARIANCE_THRESHOLD:
            penalty = (monte_carlo_variance - MONTE_CARLO_VARIANCE_THRESHOLD) * 0.5
            components["monte_carlo_penalty"] = min(penalty, 0.3)
            variance_adjusted = raw_confidence - components["monte_carlo_penalty"]

    # 2. Apply drift penalty
    drift_adjusted = variance_adjusted
    if drift_score is not None and drift_score > 0.3:
        # Significant drift reduces confidence
        components["drift_penalty"] = drift_score * 0.2
        drift_adjusted = variance_adjusted - components["drift_penalty"]

    # 3. Analyze history stability
    if history and len(history) >= 5:
        recent_confidences = []
        for h in history[-10:]:
            if "full_decision" in h:
                conf = h["full_decision"].get("confidence", 0.5)
            else:
                conf = h.get("confidence", 0.5)
            recent_confidences.append(conf)

        if len(recent_confidences) >= 2:
            # Check for wounds (sudden drops)
            wound_count = 0
            for i in range(1, len(recent_confidences)):
                drop = recent_confidences[i-1] - recent_confidences[i]
                if drop > WOUND_DROP_THRESHOLD:
                    wound_count += 1

            # Stability based on std dev and wound count
            std_dev = statistics.stdev(recent_confidences) if len(recent_confidences) > 1 else 0
            components["history_stability"] = max(0, 1.0 - std_dev - (wound_count * 0.1))

    # 4. Calculate reasoning entropy
    reasoning = ""
    if "full_decision" in decision:
        reasoning = decision["full_decision"].get("reasoning", "")
    else:
        reasoning = decision.get("reasoning", "")

    if reasoning:
        # Simple entropy estimate based on word diversity
        words = reasoning.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            # High diversity = more nuanced = slightly higher confidence
            # But too high = uncertain rambling
            components["reasoning_entropy"] = abs(0.6 - unique_ratio) * 0.1

    # Calculate final score
    final_score = drift_adjusted * components["history_stability"]
    final_score = final_score - components["reasoning_entropy"]
    final_score = max(0.0, min(1.0, final_score))

    # Determine stability
    is_stable = (
        monte_carlo_variance is None or monte_carlo_variance <= MONTE_CARLO_VARIANCE_THRESHOLD
    ) and (
        drift_score is None or drift_score <= 0.3
    ) and (
        components["history_stability"] >= 0.8
    )

    return ConfidenceScore(
        raw_confidence=raw_confidence,
        variance_adjusted=variance_adjusted,
        drift_adjusted=drift_adjusted,
        final_score=final_score,
        components=components,
        is_stable=is_stable
    )


def confidence_from_alternatives(decision: dict) -> float:
    """Estimate confidence based on alternatives considered.

    More alternatives with close scores = lower confidence.

    Args:
        decision: Decision with alternatives_considered

    Returns:
        Confidence factor (0.8-1.0)
    """
    if "full_decision" in decision:
        alts = decision["full_decision"].get("alternative_actions_considered", [])
    else:
        alts = decision.get("alternative_actions_considered", [])

    if not alts:
        return 1.0  # No alternatives = decisive

    # More alternatives = slightly less certain
    alt_penalty = min(len(alts) * 0.05, 0.15)

    return 1.0 - alt_penalty


def confidence_trend(history: list[dict], window: int = 10) -> str:
    """Analyze confidence trend over recent history.

    Args:
        history: Decision history
        window: Window size

    Returns:
        "improving", "stable", or "declining"
    """
    if len(history) < window:
        return "stable"

    recent = history[-window:]
    confidences = []

    for h in recent:
        if "full_decision" in h:
            conf = h["full_decision"].get("confidence", 0.5)
        else:
            conf = h.get("confidence", 0.5)
        confidences.append(conf)

    # Simple linear trend
    mid = window // 2
    first_half_avg = statistics.mean(confidences[:mid])
    second_half_avg = statistics.mean(confidences[mid:])

    diff = second_half_avg - first_half_avg

    if diff > 0.05:
        return "improving"
    elif diff < -0.05:
        return "declining"
    else:
        return "stable"
