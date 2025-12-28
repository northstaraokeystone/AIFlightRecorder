"""Agent Registry

Track active agents, enforce population limits.
Single source of truth for agent state.
"""

import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List
from threading import Lock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    MAX_AGENT_DEPTH,
    MAX_AGENT_POPULATION,
    DEFAULT_TTL_SECONDS,
    AGENT_STATE_SPAWNED,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_GRADUATED,
    AGENT_STATE_PRUNED,
    AGENT_STATE_SUPERPOSITION
)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    agent_type: str
    parent_id: Optional[str]
    depth: int
    state: str
    spawned_at: float          # Unix timestamp
    ttl_seconds: int
    expires_at: float          # Unix timestamp
    trigger: str               # What triggered spawn
    confidence_at_spawn: float
    effectiveness: float = 0.0
    decisions_processed: int = 0
    metadata: dict = field(default_factory=dict)


class AgentRegistry:
    """Thread-safe registry for tracking active agents."""

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._by_parent: Dict[str, List[str]] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._lock = Lock()

    def register(
        self,
        agent_type: str,
        parent_id: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        trigger: str = "manual",
        confidence: float = 0.5,
        metadata: Optional[dict] = None
    ) -> tuple[Optional[str], str]:
        """Register a new agent.

        Args:
            agent_type: Type of agent (success_learner, helper, etc.)
            parent_id: Parent agent ID if spawned by another agent
            ttl_seconds: Time-to-live in seconds
            trigger: What triggered the spawn
            confidence: Confidence at spawn time
            metadata: Additional metadata

        Returns:
            Tuple of (agent_id or None, rejection_reason or "")
        """
        with self._lock:
            # Check population limit
            active_count = self._count_active()
            if active_count >= MAX_AGENT_POPULATION:
                return None, f"Population limit ({MAX_AGENT_POPULATION}) reached"

            # Calculate depth
            depth = 0
            if parent_id and parent_id in self._agents:
                parent = self._agents[parent_id]
                depth = parent.depth + 1

            # Check depth limit
            if depth > MAX_AGENT_DEPTH:
                return None, f"Depth limit ({MAX_AGENT_DEPTH}) exceeded"

            # Generate agent ID
            agent_id = str(uuid.uuid4())
            now = time.time()

            # Create agent info
            agent = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                parent_id=parent_id,
                depth=depth,
                state=AGENT_STATE_SPAWNED,
                spawned_at=now,
                ttl_seconds=ttl_seconds,
                expires_at=now + ttl_seconds,
                trigger=trigger,
                confidence_at_spawn=confidence,
                metadata=metadata or {}
            )

            # Register
            self._agents[agent_id] = agent

            # Track by parent
            if parent_id:
                if parent_id not in self._by_parent:
                    self._by_parent[parent_id] = []
                self._by_parent[parent_id].append(agent_id)

            # Track by type
            if agent_type not in self._by_type:
                self._by_type[agent_type] = []
            self._by_type[agent_type].append(agent_id)

            return agent_id, ""

    def update_state(self, agent_id: str, new_state: str) -> bool:
        """Update agent state.

        Args:
            agent_id: Agent ID
            new_state: New state

        Returns:
            True if updated, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            self._agents[agent_id].state = new_state
            return True

    def update_effectiveness(self, agent_id: str, effectiveness: float) -> bool:
        """Update agent effectiveness score.

        Args:
            agent_id: Agent ID
            effectiveness: New effectiveness score

        Returns:
            True if updated, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            self._agents[agent_id].effectiveness = effectiveness
            return True

    def increment_decisions(self, agent_id: str) -> bool:
        """Increment decisions processed count.

        Args:
            agent_id: Agent ID

        Returns:
            True if updated, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            self._agents[agent_id].decisions_processed += 1
            return True

    def get(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info.

        Args:
            agent_id: Agent ID

        Returns:
            AgentInfo or None
        """
        with self._lock:
            return self._agents.get(agent_id)

    def get_active(self) -> List[AgentInfo]:
        """Get all active agents.

        Returns:
            List of active agents
        """
        with self._lock:
            return [
                a for a in self._agents.values()
                if a.state in [AGENT_STATE_SPAWNED, AGENT_STATE_ACTIVE]
            ]

    def get_expired(self) -> List[AgentInfo]:
        """Get all agents past their TTL.

        Returns:
            List of expired agents
        """
        now = time.time()
        with self._lock:
            return [
                a for a in self._agents.values()
                if a.state in [AGENT_STATE_SPAWNED, AGENT_STATE_ACTIVE]
                and a.expires_at < now
            ]

    def get_children(self, parent_id: str) -> List[AgentInfo]:
        """Get all children of an agent.

        Args:
            parent_id: Parent agent ID

        Returns:
            List of child agents
        """
        with self._lock:
            child_ids = self._by_parent.get(parent_id, [])
            return [self._agents[cid] for cid in child_ids if cid in self._agents]

    def get_siblings(self, agent_id: str) -> List[AgentInfo]:
        """Get all siblings of an agent (same parent).

        Args:
            agent_id: Agent ID

        Returns:
            List of sibling agents
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent or not agent.parent_id:
                return []

            child_ids = self._by_parent.get(agent.parent_id, [])
            return [
                self._agents[cid]
                for cid in child_ids
                if cid in self._agents and cid != agent_id
            ]

    def get_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Get all agents of a specific type.

        Args:
            agent_type: Agent type

        Returns:
            List of agents
        """
        with self._lock:
            agent_ids = self._by_type.get(agent_type, [])
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def remove(self, agent_id: str) -> bool:
        """Remove agent from registry.

        Args:
            agent_id: Agent ID

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]

            # Remove from parent tracking
            if agent.parent_id and agent.parent_id in self._by_parent:
                self._by_parent[agent.parent_id] = [
                    aid for aid in self._by_parent[agent.parent_id]
                    if aid != agent_id
                ]

            # Remove from type tracking
            if agent.agent_type in self._by_type:
                self._by_type[agent.agent_type] = [
                    aid for aid in self._by_type[agent.agent_type]
                    if aid != agent_id
                ]

            del self._agents[agent_id]
            return True

    def _count_active(self) -> int:
        """Count active agents (internal, assumes lock held)."""
        return sum(
            1 for a in self._agents.values()
            if a.state in [AGENT_STATE_SPAWNED, AGENT_STATE_ACTIVE]
        )

    def get_stats(self) -> dict:
        """Get registry statistics.

        Returns:
            Stats dict
        """
        with self._lock:
            active = [a for a in self._agents.values()
                     if a.state in [AGENT_STATE_SPAWNED, AGENT_STATE_ACTIVE]]

            by_type = {}
            for a in active:
                by_type[a.agent_type] = by_type.get(a.agent_type, 0) + 1

            by_depth = {}
            for a in active:
                by_depth[a.depth] = by_depth.get(a.depth, 0) + 1

            return {
                "total_registered": len(self._agents),
                "active_count": len(active),
                "by_type": by_type,
                "by_depth": by_depth,
                "max_depth_used": max((a.depth for a in active), default=0),
                "population_headroom": MAX_AGENT_POPULATION - len(active)
            }


# Global registry instance
_global_registry = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry.

    Returns:
        Global AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def reset_registry():
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = AgentRegistry()
