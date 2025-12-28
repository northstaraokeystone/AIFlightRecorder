"""Gate Decision Module

Three-tier gate decision based on confidence score.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    GATE_GREEN_THRESHOLD,
    GATE_YELLOW_THRESHOLD,
    GATE_GREEN,
    GATE_YELLOW,
    GATE_RED,
    TRIGGER_GREEN_GATE,
    TRIGGER_YELLOW_GATE,
    TRIGGER_RED_GATE
)
from config.features import (
    FEATURE_GATE_ENABLED,
    FEATURE_GATE_YELLOW_ONLY,
    is_feature_enabled
)
from src.core import emit_receipt, dual_hash


@dataclass
class GateDecision:
    """Result of a gate decision."""
    decision_id: str
    confidence_score: float
    gate_tier: str              # GREEN, YELLOW, or RED
    context_drift: float
    reasoning_entropy: float
    should_spawn: bool
    spawn_trigger: Optional[str]
    agents_to_spawn: list[str]  # Types of agents to spawn
    receipt: Optional[dict] = None


def decide(
    confidence_score: float,
    context_drift: float = 0.0,
    reasoning_entropy: float = 0.0,
    decision_id: Optional[str] = None,
    wound_count: int = 0,
    emit: bool = True
) -> GateDecision:
    """Make a gate decision based on confidence score.

    Args:
        confidence_score: Final confidence score (0-1)
        context_drift: Current context drift (0-1)
        reasoning_entropy: Reasoning entropy measure
        decision_id: Optional decision ID (generated if not provided)
        wound_count: Number of wounds in recent history
        emit: Whether to emit receipt

    Returns:
        GateDecision with tier and spawn instructions
    """
    decision_id = decision_id or str(uuid.uuid4())

    # Determine gate tier
    if confidence_score >= GATE_GREEN_THRESHOLD:
        gate_tier = GATE_GREEN
    elif confidence_score >= GATE_YELLOW_THRESHOLD:
        gate_tier = GATE_YELLOW
    else:
        gate_tier = GATE_RED

    # Check feature flags
    gate_enabled = is_feature_enabled("FEATURE_GATE_ENABLED")
    yellow_only = is_feature_enabled("FEATURE_GATE_YELLOW_ONLY")

    # Determine spawn instructions
    should_spawn = False
    spawn_trigger = None
    agents_to_spawn = []

    if gate_enabled:
        if yellow_only:
            # Only spawn for YELLOW, treat RED as YELLOW
            if gate_tier in [GATE_YELLOW, GATE_RED]:
                should_spawn = True
                spawn_trigger = TRIGGER_YELLOW_GATE
                agents_to_spawn = ["drift_watcher", "wound_watcher", "success_watcher"]
        else:
            # Full gating
            if gate_tier == GATE_GREEN:
                should_spawn = True
                spawn_trigger = TRIGGER_GREEN_GATE
                agents_to_spawn = ["success_learner"]
            elif gate_tier == GATE_YELLOW:
                should_spawn = True
                spawn_trigger = TRIGGER_YELLOW_GATE
                agents_to_spawn = ["drift_watcher", "wound_watcher", "success_watcher"]
            elif gate_tier == GATE_RED:
                should_spawn = True
                spawn_trigger = TRIGGER_RED_GATE
                # Helper count based on wound count: (wound_count // 2) + 1, max 6
                helper_count = min(6, max(1, (wound_count // 2) + 1))
                agents_to_spawn = [f"helper_{i}" for i in range(helper_count)]

    # Build receipt
    receipt = None
    if emit:
        receipt_data = {
            "decision_id": decision_id,
            "confidence_score": confidence_score,
            "gate_tier": gate_tier,
            "context_drift": context_drift,
            "reasoning_entropy": reasoning_entropy,
            "agents_spawned": agents_to_spawn if should_spawn else [],
            "spawn_trigger": spawn_trigger,
            "gate_enabled": gate_enabled,
            "yellow_only_mode": yellow_only
        }
        receipt = emit_receipt("gate_decision", receipt_data, silent=True)

    return GateDecision(
        decision_id=decision_id,
        confidence_score=confidence_score,
        gate_tier=gate_tier,
        context_drift=context_drift,
        reasoning_entropy=reasoning_entropy,
        should_spawn=should_spawn,
        spawn_trigger=spawn_trigger,
        agents_to_spawn=agents_to_spawn,
        receipt=receipt
    )


def decide_from_raw(raw_confidence: float, **kwargs) -> GateDecision:
    """Make gate decision directly from raw confidence.

    Convenience function when Monte Carlo/drift not available.

    Args:
        raw_confidence: Raw confidence value from decision
        **kwargs: Additional args passed to decide()

    Returns:
        GateDecision
    """
    return decide(raw_confidence, **kwargs)


def gate_color_symbol(tier: str) -> str:
    """Get visual symbol for gate tier.

    Args:
        tier: Gate tier string

    Returns:
        Color symbol
    """
    return {
        GATE_GREEN: "🟢",
        GATE_YELLOW: "🟡",
        GATE_RED: "🔴"
    }.get(tier, "⚫")


def should_block_decision(gate_decision: GateDecision) -> bool:
    """Check if a decision should be blocked based on gate.

    RED gate blocks execution pending investigation.

    Args:
        gate_decision: The gate decision

    Returns:
        True if decision should be blocked
    """
    if not is_feature_enabled("FEATURE_GATE_ENABLED"):
        return False  # Shadow mode, don't block

    return gate_decision.gate_tier == GATE_RED


def get_gate_action(tier: str) -> str:
    """Get recommended action for gate tier.

    Args:
        tier: Gate tier

    Returns:
        Action description
    """
    actions = {
        GATE_GREEN: "Execute + spawn success_learner",
        GATE_YELLOW: "Execute + spawn watchers (drift, wound, success)",
        GATE_RED: "Block + spawn helpers to investigate"
    }
    return actions.get(tier, "Unknown tier")
