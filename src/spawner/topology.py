"""Agent Topology Classification

Classify agent topology as OPEN, CLOSED, or HYBRID based on META-LOOP patterns.
"""

from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    EFFECTIVENESS_OPEN_THRESHOLD,
    AUTONOMY_THRESHOLD,
    TRANSFER_THRESHOLD,
    TOPOLOGY_OPEN,
    TOPOLOGY_CLOSED,
    TOPOLOGY_HYBRID,
    ESCAPE_VELOCITY
)
from config.features import is_feature_enabled
from src.core import emit_receipt
from .registry import get_registry


@dataclass
class TopologyResult:
    """Result of topology classification."""
    agent_id: str
    topology: str              # open, closed, or hybrid
    effectiveness: float
    autonomy_score: float
    transfer_score: float
    can_graduate: bool
    can_transfer: bool
    domain: str
    escape_velocity_threshold: float
    receipt: Optional[dict] = None


def classify_agent_topology(
    agent_id: str,
    effectiveness: float,
    autonomy_score: float,
    transfer_score: float,
    domain: str = "default",
    emit: bool = True
) -> TopologyResult:
    """Classify agent topology based on META-LOOP patterns.

    Classification rules:
    - OPEN: effectiveness >= 0.85, autonomy > 0.75 → Graduate to permanent
    - CLOSED: effectiveness < 0.85 → Prune, extract learnings
    - HYBRID: transfer_score > 0.70 → Transfer to other context

    Args:
        agent_id: Agent ID
        effectiveness: Effectiveness score (0-1)
        autonomy_score: Autonomy score (0-1)
        transfer_score: Transfer score (0-1)
        domain: Domain for escape velocity lookup
        emit: Whether to emit topology receipt

    Returns:
        TopologyResult
    """
    # Determine topology
    if transfer_score >= TRANSFER_THRESHOLD:
        topology = TOPOLOGY_HYBRID
    elif effectiveness >= EFFECTIVENESS_OPEN_THRESHOLD and autonomy_score >= AUTONOMY_THRESHOLD:
        topology = TOPOLOGY_OPEN
    else:
        topology = TOPOLOGY_CLOSED

    # Check escape velocity
    escape_threshold = ESCAPE_VELOCITY.get(domain, ESCAPE_VELOCITY["default"])
    can_graduate = effectiveness >= escape_threshold and autonomy_score >= AUTONOMY_THRESHOLD
    can_transfer = transfer_score >= TRANSFER_THRESHOLD

    # Emit topology receipt
    receipt = None
    if emit and is_feature_enabled("FEATURE_TOPOLOGY_CLASSIFICATION_ENABLED"):
        receipt = emit_receipt("topology", {
            "agent_id": agent_id,
            "topology": topology,
            "effectiveness": effectiveness,
            "autonomy_score": autonomy_score,
            "transfer_score": transfer_score,
            "domain": domain,
            "escape_velocity": escape_threshold,
            "can_graduate": can_graduate,
            "can_transfer": can_transfer
        }, silent=True)

    return TopologyResult(
        agent_id=agent_id,
        topology=topology,
        effectiveness=effectiveness,
        autonomy_score=autonomy_score,
        transfer_score=transfer_score,
        can_graduate=can_graduate,
        can_transfer=can_transfer,
        domain=domain,
        escape_velocity_threshold=escape_threshold,
        receipt=receipt
    )


def classify_pattern(pattern: dict) -> TopologyResult:
    """Classify a pattern's topology.

    Convenience function for patterns that aren't tied to agents.

    Args:
        pattern: Pattern dict with effectiveness, autonomy_score, transfer_score

    Returns:
        TopologyResult
    """
    return classify_agent_topology(
        agent_id=pattern.get("pattern_id", "unknown"),
        effectiveness=pattern.get("effectiveness", 0.0),
        autonomy_score=pattern.get("autonomy_score", 0.0),
        transfer_score=pattern.get("transfer_score", 0.0),
        domain=pattern.get("domain", "default"),
        emit=False
    )


def topology_action(topology: str) -> str:
    """Get recommended action for topology.

    Args:
        topology: Topology classification

    Returns:
        Action description
    """
    actions = {
        TOPOLOGY_OPEN: "Graduate to permanent pattern",
        TOPOLOGY_CLOSED: "Prune and extract learnings",
        TOPOLOGY_HYBRID: "Transfer to compatible context"
    }
    return actions.get(topology, "Unknown topology")


def check_escape_velocity(
    effectiveness: float,
    autonomy_score: float,
    domain: str = "default"
) -> tuple[bool, float]:
    """Check if pattern has reached escape velocity.

    Args:
        effectiveness: Effectiveness score
        autonomy_score: Autonomy score
        domain: Domain name

    Returns:
        Tuple of (reached_velocity, threshold)
    """
    threshold = ESCAPE_VELOCITY.get(domain, ESCAPE_VELOCITY["default"])
    reached = effectiveness >= threshold and autonomy_score >= AUTONOMY_THRESHOLD

    return reached, threshold


def topology_stats() -> dict:
    """Get topology classification statistics.

    Returns:
        Stats dict
    """
    registry = get_registry()
    active = registry.get_active()

    by_topology = {
        TOPOLOGY_OPEN: 0,
        TOPOLOGY_CLOSED: 0,
        TOPOLOGY_HYBRID: 0
    }

    graduation_candidates = 0
    transfer_candidates = 0

    for agent in active:
        # Classify each agent
        result = classify_agent_topology(
            agent.agent_id,
            agent.effectiveness,
            0.5,  # Default autonomy
            0.5,  # Default transfer
            emit=False
        )

        by_topology[result.topology] = by_topology.get(result.topology, 0) + 1

        if result.can_graduate:
            graduation_candidates += 1
        if result.can_transfer:
            transfer_candidates += 1

    return {
        "by_topology": by_topology,
        "graduation_candidates": graduation_candidates,
        "transfer_candidates": transfer_candidates
    }
