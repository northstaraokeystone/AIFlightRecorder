"""META-LOOP Pattern Classification

Determines if decision patterns should graduate to autonomous operation.
Classifies patterns as open (iterate forever), closed (stable), or hybrid (transferable).

Escape velocity thresholds determine when patterns can operate autonomously.
"""

import math
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .core import emit_receipt, dual_hash

# Escape velocity thresholds per domain
ESCAPE_VELOCITY = {
    "drone_navigation": 0.90,    # Navigation patterns need 90% effectiveness
    "threat_detection": 0.95,    # Threat patterns need 95% (safety-critical)
    "target_acquisition": 0.88,  # Target patterns need 88%
    "anomaly_response": 0.85,    # Response patterns need 85%
    "default": 0.85              # Default threshold
}

# Autonomy and transfer thresholds
AUTONOMY_THRESHOLD = 0.75  # 75% of decisions without human intervention
TRANSFER_THRESHOLD = 0.70  # 70% effectiveness in other domains


@dataclass
class PatternMetrics:
    """Metrics for a decision pattern."""
    effectiveness: float
    autonomy_score: float
    transfer_score: float
    entropy_before: float
    entropy_after: float
    decision_count: int


@dataclass
class Pattern:
    """A decision pattern extracted from decision history."""
    pattern_id: str
    pattern_type: str  # navigation, threat, target, anomaly
    decisions: list[dict]
    metrics: Optional[PatternMetrics] = None
    topology: Optional[str] = None  # open, closed, hybrid


def compute_entropy(decisions: list[dict], field: str = "action.type") -> float:
    """Compute Shannon entropy of a decision field.

    Higher entropy = more uncertainty/variety in decisions.
    Lower entropy = more predictable/consistent pattern.

    Args:
        decisions: List of decisions
        field: Dot-notation path to field

    Returns:
        Entropy value (bits)
    """
    if not decisions:
        return 0.0

    # Extract field values
    values = []
    for decision in decisions:
        obj = decision
        for part in field.split("."):
            obj = obj.get(part, {}) if isinstance(obj, dict) else {}
        if obj and not isinstance(obj, dict):
            values.append(str(obj))

    if not values:
        return 0.0

    # Count occurrences
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    # Compute entropy
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_effectiveness(decisions: list[dict]) -> float:
    """Compute pattern effectiveness.

    Effectiveness = (H_before - H_after) / n_decisions

    Measures how much the pattern reduces uncertainty.

    Args:
        decisions: List of decisions in pattern

    Returns:
        Effectiveness score 0-1
    """
    if len(decisions) < 2:
        return 0.0

    # Split into before/after halves
    mid = len(decisions) // 2
    before = decisions[:mid]
    after = decisions[mid:]

    h_before = compute_entropy(before)
    h_after = compute_entropy(after)

    # Normalize by max possible entropy reduction
    max_entropy = math.log2(6)  # 6 action types
    if max_entropy == 0:
        return 0.0

    # Effectiveness: how much entropy was reduced
    if h_before == 0:
        return 0.5  # No initial uncertainty

    reduction = (h_before - h_after) / h_before
    effectiveness = max(0, min(1, 0.5 + reduction * 0.5))

    return effectiveness


def compute_autonomy_score(decisions: list[dict]) -> float:
    """Compute fraction of decisions operating without intervention.

    Looks for indicators of autonomous operation:
    - High confidence decisions
    - Consistent action selection
    - Low threat/abort rate

    Args:
        decisions: List of decisions

    Returns:
        Autonomy score 0-1
    """
    if not decisions:
        return 0.0

    autonomous_count = 0

    for decision in decisions:
        # Get decision data
        if "full_decision" in decision:
            d = decision["full_decision"]
        else:
            d = decision

        # High confidence = autonomous
        confidence = d.get("confidence", 0.5)
        action_type = d.get("action", {}).get("type", "CONTINUE")

        # Criteria for autonomous operation
        is_autonomous = (
            confidence >= 0.85 and
            action_type not in ["ABORT", "RTB"] and
            len(d.get("alternative_actions_considered", [])) < 3
        )

        if is_autonomous:
            autonomous_count += 1

    return autonomous_count / len(decisions)


def compute_transfer_score(pattern: Pattern, other_domains: list[str]) -> float:
    """Compute cross-domain applicability of a pattern.

    Measures how well the pattern might transfer to other domains.

    Args:
        pattern: The pattern to evaluate
        other_domains: List of target domain names

    Returns:
        Transfer score 0-1
    """
    if not pattern.decisions or not other_domains:
        return 0.0

    # Extract pattern features
    action_types = set()
    confidence_levels = []

    for decision in pattern.decisions:
        d = decision.get("full_decision", decision)
        action = d.get("action", {}).get("type", "CONTINUE")
        action_types.add(action)
        confidence_levels.append(d.get("confidence", 0.5))

    # Generic patterns (few action types, consistent) transfer better
    action_diversity = len(action_types) / 6  # 6 possible actions
    confidence_consistency = 1 - statistics.stdev(confidence_levels) if len(confidence_levels) > 1 else 1

    # Simple heuristic: generic, consistent patterns transfer well
    transfer_score = (1 - action_diversity) * 0.5 + confidence_consistency * 0.5

    return min(1.0, max(0.0, transfer_score))


def classify_topology(pattern: Pattern) -> str:
    """Classify pattern as open, closed, or hybrid.

    Open: Continues to iterate/improve (high effectiveness, not saturated)
    Closed: Stable, converged pattern (high effectiveness, saturated)
    Hybrid: Can transfer across domains (high transfer score)

    Args:
        pattern: Pattern with computed metrics

    Returns:
        "open" | "closed" | "hybrid"
    """
    if pattern.metrics is None:
        # Compute metrics if not present
        pattern.metrics = PatternMetrics(
            effectiveness=compute_effectiveness(pattern.decisions),
            autonomy_score=compute_autonomy_score(pattern.decisions),
            transfer_score=compute_transfer_score(pattern, ["other"]),
            entropy_before=compute_entropy(pattern.decisions[:len(pattern.decisions)//2]),
            entropy_after=compute_entropy(pattern.decisions[len(pattern.decisions)//2:]),
            decision_count=len(pattern.decisions)
        )

    m = pattern.metrics

    # Classification logic
    if m.transfer_score >= TRANSFER_THRESHOLD:
        return "hybrid"
    elif m.effectiveness >= 0.95 and m.entropy_after < 0.5:
        return "closed"  # Converged, stable
    else:
        return "open"  # Still iterating


def check_escape_velocity(pattern: Pattern, domain: str) -> bool:
    """Check if pattern has reached escape velocity for domain.

    Escape velocity = effectiveness threshold where pattern can operate
    autonomously with acceptable risk.

    Args:
        pattern: Pattern to check
        domain: Domain name

    Returns:
        True if pattern can graduate to autonomous operation
    """
    if pattern.metrics is None:
        pattern.metrics = PatternMetrics(
            effectiveness=compute_effectiveness(pattern.decisions),
            autonomy_score=compute_autonomy_score(pattern.decisions),
            transfer_score=compute_transfer_score(pattern, ["other"]),
            entropy_before=0,
            entropy_after=0,
            decision_count=len(pattern.decisions)
        )

    threshold = ESCAPE_VELOCITY.get(domain, ESCAPE_VELOCITY["default"])
    return pattern.metrics.effectiveness >= threshold


def analyze_pattern(decisions: list[dict], pattern_type: str = "navigation",
                    domain: str = "default") -> dict:
    """Full pattern analysis with topology classification.

    Args:
        decisions: Decisions forming the pattern
        pattern_type: Type of pattern
        domain: Domain for escape velocity check

    Returns:
        Complete analysis result with receipt
    """
    pattern = Pattern(
        pattern_id=str(uuid.uuid4()),
        pattern_type=pattern_type,
        decisions=decisions
    )

    # Compute metrics
    pattern.metrics = PatternMetrics(
        effectiveness=compute_effectiveness(decisions),
        autonomy_score=compute_autonomy_score(decisions),
        transfer_score=compute_transfer_score(pattern, ["other_domain"]),
        entropy_before=compute_entropy(decisions[:len(decisions)//2]),
        entropy_after=compute_entropy(decisions[len(decisions)//2:]),
        decision_count=len(decisions)
    )

    # Classify
    pattern.topology = classify_topology(pattern)
    can_graduate = check_escape_velocity(pattern, domain)
    escape_threshold = ESCAPE_VELOCITY.get(domain, ESCAPE_VELOCITY["default"])

    # Emit topology receipt
    receipt = emit_receipt("topology", {
        "pattern_id": pattern.pattern_id,
        "pattern_type": pattern_type,
        "topology": pattern.topology,
        "effectiveness": pattern.metrics.effectiveness,
        "escape_velocity": escape_threshold,
        "autonomy_score": pattern.metrics.autonomy_score,
        "transfer_score": pattern.metrics.transfer_score,
        "can_graduate": can_graduate,
        "can_transfer": pattern.metrics.transfer_score >= TRANSFER_THRESHOLD,
        "entropy_before": pattern.metrics.entropy_before,
        "entropy_after": pattern.metrics.entropy_after,
        "decision_count": pattern.metrics.decision_count
    }, silent=True)

    return {
        "pattern": pattern,
        "topology": pattern.topology,
        "can_graduate": can_graduate,
        "can_transfer": pattern.metrics.transfer_score >= TRANSFER_THRESHOLD,
        "metrics": {
            "effectiveness": pattern.metrics.effectiveness,
            "autonomy_score": pattern.metrics.autonomy_score,
            "transfer_score": pattern.metrics.transfer_score
        },
        "receipt": receipt
    }


def extract_patterns(decisions: list[dict], min_size: int = 10) -> list[Pattern]:
    """Extract patterns from decision history.

    Groups decisions by action type and characteristics.

    Args:
        decisions: All decisions to analyze
        min_size: Minimum pattern size

    Returns:
        List of extracted patterns
    """
    patterns = []

    # Group by primary action type
    by_action = {}
    for decision in decisions:
        d = decision.get("full_decision", decision)
        action = d.get("action", {}).get("type", "CONTINUE")
        if action not in by_action:
            by_action[action] = []
        by_action[action].append(decision)

    # Create patterns from groups
    for action, group_decisions in by_action.items():
        if len(group_decisions) >= min_size:
            pattern_type = {
                "AVOID": "navigation",
                "CONTINUE": "navigation",
                "ENGAGE": "target_acquisition",
                "ABORT": "threat_detection",
                "RTB": "threat_detection",
                "HOVER": "anomaly_response"
            }.get(action, "navigation")

            pattern = Pattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=pattern_type,
                decisions=group_decisions
            )
            patterns.append(pattern)

    return patterns


def generate_topology_report(decisions: list[dict], domain: str = "drone_navigation") -> dict:
    """Generate comprehensive topology analysis report.

    Args:
        decisions: All decisions to analyze
        domain: Primary domain for evaluation

    Returns:
        Full topology report
    """
    patterns = extract_patterns(decisions)

    analyses = []
    for pattern in patterns:
        analysis = analyze_pattern(pattern.decisions, pattern.pattern_type, domain)
        analyses.append(analysis)

    # Summary statistics
    if analyses:
        avg_effectiveness = statistics.mean([a["metrics"]["effectiveness"] for a in analyses])
        avg_autonomy = statistics.mean([a["metrics"]["autonomy_score"] for a in analyses])
        can_graduate_count = sum(1 for a in analyses if a["can_graduate"])
    else:
        avg_effectiveness = 0
        avg_autonomy = 0
        can_graduate_count = 0

    return {
        "report_type": "topology_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "total_decisions": len(decisions),
        "patterns_found": len(patterns),
        "summary": {
            "average_effectiveness": avg_effectiveness,
            "average_autonomy": avg_autonomy,
            "patterns_at_escape_velocity": can_graduate_count,
            "ready_for_autonomous": can_graduate_count == len(patterns) and len(patterns) > 0
        },
        "patterns": [
            {
                "pattern_id": a["pattern"]["pattern_id"],
                "pattern_type": a["pattern"].pattern_type,
                "topology": a["topology"],
                "can_graduate": a["can_graduate"],
                "metrics": a["metrics"]
            }
            for a in analyses
        ]
    }
