"""Monte Carlo Simulation

Run N simulated decision variations with noise injection.
Must complete 100 simulations in <200ms.
"""

import random
import time
import copy
from dataclasses import dataclass
from typing import Optional, List, Callable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    MONTE_CARLO_SIMS,
    MONTE_CARLO_LATENCY_MS,
    MONTE_CARLO_NOISE
)
from config.features import is_feature_enabled
from src.core import emit_receipt


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""
    n_simulations: int
    outcomes: List[dict]
    latency_ms: float
    slo_met: bool
    receipt: Optional[dict] = None


def simulate(
    decision: dict,
    n_sims: int = MONTE_CARLO_SIMS,
    noise: float = MONTE_CARLO_NOISE,
    decision_fn: Optional[Callable] = None,
    emit: bool = True
) -> SimulationResult:
    """Run N simulated decision variations.

    Perturbs decision inputs with noise and re-runs decision logic.
    Fast implementation to meet <200ms for 100 sims SLO.

    Args:
        decision: Original decision to simulate variations of
        n_sims: Number of simulations (default 100)
        noise: Noise factor for perturbations (default 0.05)
        decision_fn: Optional custom decision function
        emit: Whether to emit receipt

    Returns:
        SimulationResult with all outcomes
    """
    if not is_feature_enabled("FEATURE_MONTE_CARLO_ENABLED"):
        # Shadow mode - return empty result
        return SimulationResult(
            n_simulations=0,
            outcomes=[],
            latency_ms=0.0,
            slo_met=True
        )

    start_time = time.perf_counter()
    outcomes = []

    # Extract decision components
    if "full_decision" in decision:
        base_decision = decision["full_decision"]
    else:
        base_decision = decision

    base_confidence = base_decision.get("confidence", 0.5)
    base_action = base_decision.get("action", {}).get("type", "CONTINUE")
    perception = base_decision.get("perception", {})
    telemetry = base_decision.get("telemetry_snapshot", {})

    for i in range(n_sims):
        # Create perturbed version
        perturbed = _perturb_decision(base_decision, noise, i)

        # Evaluate outcome
        if decision_fn:
            # Use custom decision function
            outcome = decision_fn(perturbed)
        else:
            # Fast approximation: perturb confidence directly
            outcome = _quick_simulate(perturbed, noise, i)

        outcomes.append(outcome)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    slo_met = elapsed_ms <= MONTE_CARLO_LATENCY_MS

    # Emit receipt
    receipt = None
    if emit:
        from .variance import calculate_variance

        variance_result = calculate_variance(outcomes)

        receipt = emit_receipt("monte_carlo", {
            "decision_id": base_decision.get("decision_id", "unknown"),
            "n_simulations": n_sims,
            "variance": variance_result.variance,
            "mean_outcome": variance_result.mean_confidence,
            "is_stable": variance_result.variance <= 0.2,
            "latency_ms": elapsed_ms,
            "slo_met": slo_met
        }, silent=True)

        # Emit SLO violation if latency exceeded
        if not slo_met:
            emit_receipt("anomaly", {
                "metric": "monte_carlo_latency",
                "baseline": MONTE_CARLO_LATENCY_MS,
                "actual": elapsed_ms,
                "delta": elapsed_ms - MONTE_CARLO_LATENCY_MS,
                "classification": "slo_violation",
                "action": "alert"
            }, silent=True)

    return SimulationResult(
        n_simulations=n_sims,
        outcomes=outcomes,
        latency_ms=elapsed_ms,
        slo_met=slo_met,
        receipt=receipt
    )


def _perturb_decision(decision: dict, noise: float, seed: int) -> dict:
    """Perturb a decision with noise.

    Args:
        decision: Original decision
        noise: Noise factor
        seed: Random seed for reproducibility

    Returns:
        Perturbed decision
    """
    random.seed(seed)

    # Deep copy to avoid modifying original
    perturbed = copy.deepcopy(decision)

    # Perturb numeric values in perception
    perception = perturbed.get("perception", {})

    # Perturb obstacle distances
    obstacles = perception.get("obstacles", [])
    for obs in obstacles:
        if "distance_m" in obs:
            obs["distance_m"] *= (1.0 + random.gauss(0, noise))
        if "confidence" in obs:
            obs["confidence"] = max(0, min(1, obs["confidence"] + random.gauss(0, noise)))

    # Perturb threat severity
    threats = perception.get("threats", [])
    for threat in threats:
        if "severity" in threat:
            threat["severity"] = max(0, min(1, threat["severity"] + random.gauss(0, noise)))

    # Perturb telemetry
    telemetry = perturbed.get("telemetry_snapshot", {})
    if "battery_pct" in telemetry:
        telemetry["battery_pct"] = max(0, min(100,
            telemetry["battery_pct"] + random.gauss(0, noise * 10)))

    gps = telemetry.get("gps", {})
    if gps:
        if "lat" in gps:
            gps["lat"] += random.gauss(0, noise * 0.0001)
        if "lon" in gps:
            gps["lon"] += random.gauss(0, noise * 0.0001)

    return perturbed


def _quick_simulate(decision: dict, noise: float, seed: int) -> dict:
    """Fast simulation that approximates decision outcome.

    Args:
        decision: Perturbed decision
        noise: Noise factor
        seed: Random seed

    Returns:
        Outcome dict
    """
    random.seed(seed)

    base_confidence = decision.get("confidence", 0.5)
    base_action = decision.get("action", {}).get("type", "CONTINUE")

    # Add noise to confidence
    perturbed_confidence = max(0, min(1,
        base_confidence + random.gauss(0, noise * 0.5)))

    # Determine if action would change
    perception = decision.get("perception", {})
    threats = perception.get("threats", [])
    obstacles = perception.get("obstacles", [])

    action = base_action

    # High threat might change action
    if threats:
        max_threat = max((t.get("severity", 0) for t in threats), default=0)
        if max_threat > 0.7 + random.gauss(0, noise):
            action = "ABORT"
            perturbed_confidence = 0.95
        elif max_threat > 0.4 + random.gauss(0, noise):
            action = "AVOID"
            perturbed_confidence = 0.85

    # Close obstacle might change action
    if obstacles and action not in ["ABORT"]:
        min_distance = min((o.get("distance_m", 999) for o in obstacles), default=999)
        if min_distance < 30 + random.gauss(0, noise * 10):
            action = "AVOID"
            perturbed_confidence = 0.8

    return {
        "confidence": perturbed_confidence,
        "action_type": action,
        "action_changed": action != base_action,
        "seed": seed
    }


def simulate_batch(
    decisions: List[dict],
    n_sims_per: int = 50
) -> List[SimulationResult]:
    """Simulate multiple decisions with reduced sims per decision.

    Args:
        decisions: List of decisions
        n_sims_per: Sims per decision (reduced for batch)

    Returns:
        List of SimulationResults
    """
    results = []

    for decision in decisions:
        result = simulate(decision, n_sims=n_sims_per, emit=False)
        results.append(result)

    return results


def estimate_stability_fast(decision: dict, n_quick: int = 20) -> float:
    """Fast stability estimate with fewer simulations.

    For quick screening before full Monte Carlo.

    Args:
        decision: Decision to evaluate
        n_quick: Number of quick sims (default 20)

    Returns:
        Stability score (0-1, higher = more stable)
    """
    result = simulate(decision, n_sims=n_quick, emit=False)

    if not result.outcomes:
        return 1.0  # No simulation = assume stable

    # Count action changes
    changes = sum(1 for o in result.outcomes if o.get("action_changed", False))
    change_rate = changes / len(result.outcomes)

    # Calculate confidence variance
    confidences = [o.get("confidence", 0.5) for o in result.outcomes]
    if len(confidences) > 1:
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
    else:
        variance = 0.0

    # Stability = low change rate + low variance
    stability = 1.0 - change_rate - (variance * 2)
    return max(0.0, min(1.0, stability))
