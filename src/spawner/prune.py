"""Agent Pruning Module

Terminate agents based on:
- TTL expiration
- Sibling solved the problem
- Depth limit exceeded
- Resource cap reached
- Low effectiveness
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    PRUNE_TTL_EXPIRED,
    PRUNE_SIBLING_SOLVED,
    PRUNE_DEPTH_LIMIT,
    PRUNE_RESOURCE_CAP,
    PRUNE_LOW_EFFECTIVENESS,
    AGENT_STATE_PRUNED,
    AGENT_STATE_SUPERPOSITION,
    EFFECTIVENESS_OPEN_THRESHOLD
)
from src.core import emit_receipt
from .registry import get_registry


@dataclass
class PruneResult:
    """Result of agent pruning."""
    agents_terminated: List[str]
    reason: str
    resources_freed: dict
    receipt: Optional[dict] = None


def prune_agents(
    agent_ids: List[str],
    reason: str,
    emit: bool = True
) -> PruneResult:
    """Prune specified agents.

    Args:
        agent_ids: List of agent IDs to prune
        reason: Reason for pruning
        emit: Whether to emit pruning receipt

    Returns:
        PruneResult
    """
    registry = get_registry()
    terminated = []
    resources_freed = {
        "memory_mb": 0,
        "cpu_slots": 0
    }

    for agent_id in agent_ids:
        agent = registry.get(agent_id)
        if agent:
            # Track resources (estimated)
            resources_freed["memory_mb"] += 1  # Approximate
            resources_freed["cpu_slots"] += 1

            # Update state and remove
            registry.update_state(agent_id, AGENT_STATE_PRUNED)
            registry.remove(agent_id)
            terminated.append(agent_id)

    # Emit pruning receipt
    receipt = None
    if emit and terminated:
        receipt = emit_receipt("pruning", {
            "agents_terminated": terminated,
            "reason": reason,
            "resources_freed": resources_freed
        }, silent=True)

    return PruneResult(
        agents_terminated=terminated,
        reason=reason,
        resources_freed=resources_freed,
        receipt=receipt
    )


def prune_expired() -> PruneResult:
    """Prune all agents past their TTL.

    Returns:
        PruneResult
    """
    registry = get_registry()
    expired = registry.get_expired()
    agent_ids = [a.agent_id for a in expired]

    return prune_agents(agent_ids, PRUNE_TTL_EXPIRED)


def prune_siblings(solved_agent_id: str) -> PruneResult:
    """Prune siblings when one agent solves the problem.

    Args:
        solved_agent_id: ID of agent that found solution

    Returns:
        PruneResult
    """
    registry = get_registry()
    siblings = registry.get_siblings(solved_agent_id)
    agent_ids = [s.agent_id for s in siblings]

    return prune_agents(agent_ids, PRUNE_SIBLING_SOLVED)


def prune_low_effectiveness(threshold: float = EFFECTIVENESS_OPEN_THRESHOLD) -> PruneResult:
    """Prune agents with low effectiveness.

    Args:
        threshold: Effectiveness threshold

    Returns:
        PruneResult
    """
    registry = get_registry()
    active = registry.get_active()

    low_effectiveness = [
        a.agent_id for a in active
        if a.effectiveness < threshold and a.decisions_processed >= 5
    ]

    return prune_agents(low_effectiveness, PRUNE_LOW_EFFECTIVENESS)


def move_to_superposition(agent_ids: List[str]) -> int:
    """Move agents to superposition state instead of destroying.

    Superposition agents can resurface if conditions change.

    Args:
        agent_ids: Agents to move to superposition

    Returns:
        Number of agents moved
    """
    registry = get_registry()
    count = 0

    for agent_id in agent_ids:
        if registry.update_state(agent_id, AGENT_STATE_SUPERPOSITION):
            count += 1

    return count


def auto_prune_cycle() -> dict:
    """Run automatic pruning cycle.

    Handles:
    1. TTL expired agents
    2. Low effectiveness agents

    Returns:
        Summary of pruning actions
    """
    results = {
        "expired_pruned": 0,
        "low_effectiveness_pruned": 0,
        "total_pruned": 0,
        "resources_freed": {"memory_mb": 0, "cpu_slots": 0}
    }

    # Prune expired
    expired_result = prune_expired()
    results["expired_pruned"] = len(expired_result.agents_terminated)
    results["resources_freed"]["memory_mb"] += expired_result.resources_freed["memory_mb"]
    results["resources_freed"]["cpu_slots"] += expired_result.resources_freed["cpu_slots"]

    # Prune low effectiveness
    low_eff_result = prune_low_effectiveness()
    results["low_effectiveness_pruned"] = len(low_eff_result.agents_terminated)
    results["resources_freed"]["memory_mb"] += low_eff_result.resources_freed["memory_mb"]
    results["resources_freed"]["cpu_slots"] += low_eff_result.resources_freed["cpu_slots"]

    results["total_pruned"] = results["expired_pruned"] + results["low_effectiveness_pruned"]

    return results


def prune_stats() -> dict:
    """Get pruning statistics.

    Returns:
        Stats dict
    """
    registry = get_registry()
    expired = registry.get_expired()
    active = registry.get_active()

    low_eff_count = sum(
        1 for a in active
        if a.effectiveness < EFFECTIVENESS_OPEN_THRESHOLD and a.decisions_processed >= 5
    )

    return {
        "pending_expiration": len(expired),
        "low_effectiveness_candidates": low_eff_count,
        "total_prune_candidates": len(expired) + low_eff_count
    }
