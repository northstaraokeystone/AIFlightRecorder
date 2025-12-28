"""Agent Graduation Module

Promote effective agents to permanent patterns.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    EFFECTIVENESS_OPEN_THRESHOLD,
    AUTONOMY_THRESHOLD,
    AGENT_STATE_GRADUATED,
    TOPOLOGY_OPEN
)
from config.features import is_feature_enabled
from src.core import emit_receipt
from .registry import get_registry
from .patterns import get_pattern_store


@dataclass
class GraduationResult:
    """Result of graduation evaluation."""
    agent_id: str
    graduated: bool
    pattern_id: Optional[str]
    effectiveness: float
    autonomy_score: float
    rejection_reason: Optional[str] = None
    receipt: Optional[dict] = None


def evaluate_graduation(
    agent_id: str,
    effectiveness: float,
    autonomy_score: float = 0.0,
    solution_pattern: Optional[dict] = None,
    emit: bool = True
) -> GraduationResult:
    """Evaluate if an agent should graduate to permanent pattern.

    Args:
        agent_id: Agent ID
        effectiveness: Agent effectiveness score (0-1)
        autonomy_score: Autonomy score (0-1)
        solution_pattern: The pattern/solution the agent found
        emit: Whether to emit graduation receipt

    Returns:
        GraduationResult
    """
    registry = get_registry()
    agent = registry.get(agent_id)

    if not agent:
        return GraduationResult(
            agent_id=agent_id,
            graduated=False,
            pattern_id=None,
            effectiveness=effectiveness,
            autonomy_score=autonomy_score,
            rejection_reason="Agent not found"
        )

    # Check feature flag
    if not is_feature_enabled("FEATURE_PATTERN_GRADUATION_ENABLED"):
        return GraduationResult(
            agent_id=agent_id,
            graduated=False,
            pattern_id=None,
            effectiveness=effectiveness,
            autonomy_score=autonomy_score,
            rejection_reason="Graduation disabled (shadow mode)"
        )

    # Check thresholds
    if effectiveness < EFFECTIVENESS_OPEN_THRESHOLD:
        return GraduationResult(
            agent_id=agent_id,
            graduated=False,
            pattern_id=None,
            effectiveness=effectiveness,
            autonomy_score=autonomy_score,
            rejection_reason=f"Effectiveness {effectiveness:.2f} below threshold {EFFECTIVENESS_OPEN_THRESHOLD}"
        )

    if autonomy_score < AUTONOMY_THRESHOLD:
        return GraduationResult(
            agent_id=agent_id,
            graduated=False,
            pattern_id=None,
            effectiveness=effectiveness,
            autonomy_score=autonomy_score,
            rejection_reason=f"Autonomy {autonomy_score:.2f} below threshold {AUTONOMY_THRESHOLD}"
        )

    # Graduate the agent
    registry.update_state(agent_id, AGENT_STATE_GRADUATED)
    registry.update_effectiveness(agent_id, effectiveness)

    # Store pattern
    pattern_id = None
    if solution_pattern:
        pattern_store = get_pattern_store()
        pattern_id = pattern_store.store(
            pattern=solution_pattern,
            effectiveness=effectiveness,
            source_agent_id=agent_id
        )

    # Emit graduation receipt
    receipt = None
    if emit:
        receipt = emit_receipt("graduation", {
            "agent_id": agent_id,
            "solution_pattern_id": pattern_id,
            "effectiveness": effectiveness,
            "autonomy_score": autonomy_score,
            "promoted_to": "permanent_helper"
        }, silent=True)

    return GraduationResult(
        agent_id=agent_id,
        graduated=True,
        pattern_id=pattern_id,
        effectiveness=effectiveness,
        autonomy_score=autonomy_score,
        receipt=receipt
    )


def check_graduation_eligibility(agent_id: str) -> tuple[bool, str]:
    """Check if agent is eligible for graduation.

    Args:
        agent_id: Agent ID

    Returns:
        Tuple of (eligible, reason)
    """
    registry = get_registry()
    agent = registry.get(agent_id)

    if not agent:
        return False, "Agent not found"

    if not is_feature_enabled("FEATURE_PATTERN_GRADUATION_ENABLED"):
        return False, "Graduation disabled"

    if agent.effectiveness < EFFECTIVENESS_OPEN_THRESHOLD:
        return False, f"Effectiveness {agent.effectiveness:.2f} below threshold"

    # Would check autonomy score here if tracked

    return True, "Eligible for graduation"


def get_graduated_agents() -> list:
    """Get all graduated agents.

    Returns:
        List of graduated agent infos
    """
    registry = get_registry()
    all_agents = list(registry._agents.values())

    return [a for a in all_agents if a.state == AGENT_STATE_GRADUATED]


def graduation_stats() -> dict:
    """Get graduation statistics.

    Returns:
        Stats dict
    """
    graduated = get_graduated_agents()
    pattern_store = get_pattern_store()

    avg_effectiveness = 0.0
    if graduated:
        avg_effectiveness = sum(a.effectiveness for a in graduated) / len(graduated)

    return {
        "total_graduated": len(graduated),
        "average_effectiveness": avg_effectiveness,
        "patterns_stored": pattern_store.count(),
        "pattern_usage_count": pattern_store.total_usage()
    }
