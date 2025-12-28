"""Agent Birth Module

Spawn agents based on gate color and wound count.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    GATE_GREEN,
    GATE_YELLOW,
    GATE_RED,
    GREEN_LEARNER_TTL,
    YELLOW_WATCHER_TTL_EXTRA,
    RED_HELPER_TTL,
    MIN_HELPER_SPAWN,
    MAX_HELPER_SPAWN,
    TRIGGER_GREEN_GATE,
    TRIGGER_YELLOW_GATE,
    TRIGGER_RED_GATE,
    TRIGGER_WOUND_THRESHOLD,
    AGENT_STATE_SPAWNED
)
from config.features import (
    is_feature_enabled
)
from src.core import emit_receipt, dual_hash
from .registry import get_registry


@dataclass
class SpawnResult:
    """Result of agent spawning."""
    success: bool
    agent_ids: List[str]
    trigger: str
    confidence_at_spawn: float
    depth_level: int
    rejection_reason: Optional[str] = None
    receipt: Optional[dict] = None


def spawn(
    gate_tier: str,
    confidence: float,
    wound_count: int = 0,
    parent_id: Optional[str] = None,
    action_duration_seconds: int = 10,
    emit: bool = True
) -> SpawnResult:
    """Spawn agents based on gate color.

    Spawning rules by gate:
    - GREEN: 1 success_learner (TTL=60s)
    - YELLOW: 3 watchers (drift, wound, success) (TTL=action_duration+30s)
    - RED: (wound_count // 2) + 1 helpers (min 1, max 6) (TTL=300s)

    Args:
        gate_tier: GREEN, YELLOW, or RED
        confidence: Confidence score at spawn time
        wound_count: Number of wounds in recent history
        parent_id: Parent agent ID if spawned by another agent
        action_duration_seconds: Expected action duration
        emit: Whether to emit spawn receipt

    Returns:
        SpawnResult with spawned agent IDs
    """
    registry = get_registry()
    agent_ids = []
    rejection_reason = None

    # Determine trigger
    trigger = {
        GATE_GREEN: TRIGGER_GREEN_GATE,
        GATE_YELLOW: TRIGGER_YELLOW_GATE,
        GATE_RED: TRIGGER_RED_GATE
    }.get(gate_tier, TRIGGER_WOUND_THRESHOLD)

    # Check feature flags
    spawning_enabled = is_feature_enabled("FEATURE_AGENT_SPAWNING_ENABLED")

    if gate_tier == GATE_GREEN:
        if not is_feature_enabled("FEATURE_GREEN_LEARNERS_ENABLED"):
            spawning_enabled = False
    elif gate_tier == GATE_YELLOW:
        if not is_feature_enabled("FEATURE_YELLOW_WATCHERS_ENABLED"):
            spawning_enabled = False
    elif gate_tier == GATE_RED:
        if not is_feature_enabled("FEATURE_RED_HELPERS_ENABLED"):
            spawning_enabled = False

    if not spawning_enabled:
        # Shadow mode - log what would happen but don't spawn
        return SpawnResult(
            success=False,
            agent_ids=[],
            trigger=trigger,
            confidence_at_spawn=confidence,
            depth_level=0,
            rejection_reason="Spawning disabled (shadow mode)"
        )

    # Spawn based on gate tier
    if gate_tier == GATE_GREEN:
        # Spawn 1 success_learner
        agent_id, reason = registry.register(
            agent_type="success_learner",
            parent_id=parent_id,
            ttl_seconds=GREEN_LEARNER_TTL,
            trigger=trigger,
            confidence=confidence
        )
        if agent_id:
            agent_ids.append(agent_id)
        else:
            rejection_reason = reason

    elif gate_tier == GATE_YELLOW:
        # Spawn 3 watchers
        ttl = action_duration_seconds + YELLOW_WATCHER_TTL_EXTRA
        watcher_types = ["drift_watcher", "wound_watcher", "success_watcher"]

        for wtype in watcher_types:
            agent_id, reason = registry.register(
                agent_type=wtype,
                parent_id=parent_id,
                ttl_seconds=ttl,
                trigger=trigger,
                confidence=confidence
            )
            if agent_id:
                agent_ids.append(agent_id)
            elif not rejection_reason:
                rejection_reason = reason

    elif gate_tier == GATE_RED:
        # Spawn (wound_count // 2) + 1 helpers, min 1, max 6
        helper_count = max(MIN_HELPER_SPAWN, min(MAX_HELPER_SPAWN, (wound_count // 2) + 1))

        for i in range(helper_count):
            agent_id, reason = registry.register(
                agent_type="helper",
                parent_id=parent_id,
                ttl_seconds=RED_HELPER_TTL,
                trigger=trigger,
                confidence=confidence,
                metadata={"helper_index": i}
            )
            if agent_id:
                agent_ids.append(agent_id)
            elif not rejection_reason:
                rejection_reason = reason

    # Get depth level
    depth_level = 0
    if agent_ids:
        agent = registry.get(agent_ids[0])
        if agent:
            depth_level = agent.depth

    # Emit spawn receipt
    receipt = None
    if emit and agent_ids:
        receipt_data = {
            "parent_agent_id": parent_id,
            "child_agents": agent_ids,
            "trigger": trigger,
            "confidence_at_spawn": confidence,
            "depth_level": depth_level,
            "max_ttl_seconds": _get_max_ttl(gate_tier, action_duration_seconds),
            "wound_count": wound_count,
            "gate_tier": gate_tier
        }
        receipt = emit_receipt("spawn", receipt_data, silent=True)

    return SpawnResult(
        success=len(agent_ids) > 0,
        agent_ids=agent_ids,
        trigger=trigger,
        confidence_at_spawn=confidence,
        depth_level=depth_level,
        rejection_reason=rejection_reason,
        receipt=receipt
    )


def _get_max_ttl(gate_tier: str, action_duration: int) -> int:
    """Get max TTL for gate tier.

    Args:
        gate_tier: Gate tier
        action_duration: Action duration in seconds

    Returns:
        TTL in seconds
    """
    if gate_tier == GATE_GREEN:
        return GREEN_LEARNER_TTL
    elif gate_tier == GATE_YELLOW:
        return action_duration + YELLOW_WATCHER_TTL_EXTRA
    elif gate_tier == GATE_RED:
        return RED_HELPER_TTL
    return RED_HELPER_TTL


def spawn_from_wound_threshold(
    wound_count: int,
    confidence: float,
    parent_id: Optional[str] = None,
    emit: bool = True
) -> SpawnResult:
    """Spawn helpers when wound threshold is exceeded.

    Args:
        wound_count: Current wound count
        confidence: Current confidence
        parent_id: Parent agent ID
        emit: Whether to emit receipt

    Returns:
        SpawnResult
    """
    registry = get_registry()
    agent_ids = []

    if not is_feature_enabled("FEATURE_RED_HELPERS_ENABLED"):
        return SpawnResult(
            success=False,
            agent_ids=[],
            trigger=TRIGGER_WOUND_THRESHOLD,
            confidence_at_spawn=confidence,
            depth_level=0,
            rejection_reason="Helpers disabled"
        )

    # Spawn helpers based on wound count
    helper_count = max(MIN_HELPER_SPAWN, min(MAX_HELPER_SPAWN, (wound_count // 2) + 1))

    for i in range(helper_count):
        agent_id, reason = registry.register(
            agent_type="helper",
            parent_id=parent_id,
            ttl_seconds=RED_HELPER_TTL,
            trigger=TRIGGER_WOUND_THRESHOLD,
            confidence=confidence,
            metadata={"helper_index": i, "wound_count": wound_count}
        )
        if agent_id:
            agent_ids.append(agent_id)

    depth_level = 0
    if agent_ids:
        agent = registry.get(agent_ids[0])
        if agent:
            depth_level = agent.depth

    receipt = None
    if emit and agent_ids:
        receipt_data = {
            "parent_agent_id": parent_id,
            "child_agents": agent_ids,
            "trigger": TRIGGER_WOUND_THRESHOLD,
            "confidence_at_spawn": confidence,
            "depth_level": depth_level,
            "max_ttl_seconds": RED_HELPER_TTL,
            "wound_count": wound_count
        }
        receipt = emit_receipt("spawn", receipt_data, silent=True)

    return SpawnResult(
        success=len(agent_ids) > 0,
        agent_ids=agent_ids,
        trigger=TRIGGER_WOUND_THRESHOLD,
        confidence_at_spawn=confidence,
        depth_level=depth_level,
        receipt=receipt
    )


def can_spawn() -> tuple[bool, str]:
    """Check if spawning is currently allowed.

    Returns:
        Tuple of (can_spawn, reason)
    """
    if not is_feature_enabled("FEATURE_AGENT_SPAWNING_ENABLED"):
        return False, "Agent spawning disabled"

    registry = get_registry()
    stats = registry.get_stats()

    if stats["population_headroom"] <= 0:
        return False, "Population limit reached"

    return True, ""
