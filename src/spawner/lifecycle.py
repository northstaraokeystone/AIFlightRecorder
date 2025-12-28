"""Agent Lifecycle Management

Track agent state transitions: SPAWNED → ACTIVE → (GRADUATED | PRUNED)
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    AGENT_STATE_SPAWNED,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_GRADUATED,
    AGENT_STATE_PRUNED,
    AGENT_STATE_SUPERPOSITION
)
from src.core import emit_receipt
from .registry import get_registry, AgentInfo


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    state: str
    time_in_state_seconds: float
    ttl_remaining_seconds: float
    effectiveness: float
    decisions_processed: int
    is_expired: bool


def get_agent_state(agent_id: str) -> Optional[AgentState]:
    """Get current state of an agent.

    Args:
        agent_id: Agent ID

    Returns:
        AgentState or None if not found
    """
    registry = get_registry()
    agent = registry.get(agent_id)

    if not agent:
        return None

    now = time.time()

    return AgentState(
        agent_id=agent_id,
        state=agent.state,
        time_in_state_seconds=now - agent.spawned_at,
        ttl_remaining_seconds=max(0, agent.expires_at - now),
        effectiveness=agent.effectiveness,
        decisions_processed=agent.decisions_processed,
        is_expired=now > agent.expires_at
    )


def update_lifecycle(
    agent_id: str,
    events: list[dict],
    emit: bool = True
) -> Optional[dict]:
    """Update agent lifecycle based on events.

    Args:
        agent_id: Agent ID
        events: List of events (e.g., decision_processed, solution_found)
        emit: Whether to emit lifecycle receipt

    Returns:
        Lifecycle receipt or None
    """
    registry = get_registry()
    agent = registry.get(agent_id)

    if not agent:
        return None

    state_changed = False
    old_state = agent.state
    new_state = agent.state

    for event in events:
        event_type = event.get("type", "")

        if event_type == "activated" and agent.state == AGENT_STATE_SPAWNED:
            new_state = AGENT_STATE_ACTIVE
            state_changed = True

        elif event_type == "decision_processed":
            registry.increment_decisions(agent_id)

        elif event_type == "effectiveness_update":
            effectiveness = event.get("effectiveness", 0.0)
            registry.update_effectiveness(agent_id, effectiveness)

        elif event_type == "solution_found":
            # Agent found a solution, may be eligible for graduation
            effectiveness = event.get("effectiveness", 0.0)
            registry.update_effectiveness(agent_id, effectiveness)

        elif event_type == "graduated":
            new_state = AGENT_STATE_GRADUATED
            state_changed = True

        elif event_type == "pruned":
            new_state = AGENT_STATE_PRUNED
            state_changed = True

        elif event_type == "superposition":
            new_state = AGENT_STATE_SUPERPOSITION
            state_changed = True

    if state_changed:
        registry.update_state(agent_id, new_state)

    # Emit lifecycle receipt if state changed
    receipt = None
    if emit and state_changed:
        receipt = emit_receipt("lifecycle", {
            "agent_id": agent_id,
            "old_state": old_state,
            "new_state": new_state,
            "events_processed": len(events),
            "effectiveness": agent.effectiveness,
            "decisions_processed": agent.decisions_processed
        }, silent=True)

    return receipt


def activate_agent(agent_id: str) -> bool:
    """Transition agent from SPAWNED to ACTIVE.

    Args:
        agent_id: Agent ID

    Returns:
        True if activated, False otherwise
    """
    registry = get_registry()
    agent = registry.get(agent_id)

    if not agent or agent.state != AGENT_STATE_SPAWNED:
        return False

    registry.update_state(agent_id, AGENT_STATE_ACTIVE)
    return True


def get_all_states() -> List[AgentState]:
    """Get states of all active agents.

    Returns:
        List of AgentState
    """
    registry = get_registry()
    active = registry.get_active()

    states = []
    for agent in active:
        state = get_agent_state(agent.agent_id)
        if state:
            states.append(state)

    return states


def get_expired_agents() -> List[AgentState]:
    """Get all expired agents.

    Returns:
        List of expired AgentState
    """
    registry = get_registry()
    expired = registry.get_expired()

    states = []
    for agent in expired:
        state = get_agent_state(agent.agent_id)
        if state:
            states.append(state)

    return states


def lifecycle_summary() -> dict:
    """Get lifecycle summary for all agents.

    Returns:
        Summary dict
    """
    registry = get_registry()
    stats = registry.get_stats()

    all_agents = registry.get_active()
    expired = registry.get_expired()

    by_state = {}
    for agent in all_agents:
        by_state[agent.state] = by_state.get(agent.state, 0) + 1

    avg_effectiveness = 0.0
    if all_agents:
        avg_effectiveness = sum(a.effectiveness for a in all_agents) / len(all_agents)

    return {
        "total_active": len(all_agents),
        "by_state": by_state,
        "expired_pending_prune": len(expired),
        "average_effectiveness": avg_effectiveness,
        "by_type": stats["by_type"],
        "by_depth": stats["by_depth"]
    }
