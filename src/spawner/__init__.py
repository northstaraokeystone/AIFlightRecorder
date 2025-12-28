"""Agent Spawner Module - v2.0

META-LOOP agent birthing architecture:
- Birth agents based on gate color
- Track agent lifecycle (spawned → active → graduated/pruned)
- Graduate effective agents to permanent patterns
- Prune agents on TTL, sibling success, or resource limits
"""

from .birth import spawn, SpawnResult
from .lifecycle import update_lifecycle, AgentState, get_agent_state
from .graduate import evaluate_graduation, GraduationResult
from .prune import prune_agents, PruneResult
from .registry import AgentRegistry, get_registry
from .topology import classify_agent_topology, TopologyResult
from .patterns import PatternStore

__all__ = [
    "spawn",
    "SpawnResult",
    "update_lifecycle",
    "AgentState",
    "get_agent_state",
    "evaluate_graduation",
    "GraduationResult",
    "prune_agents",
    "PruneResult",
    "AgentRegistry",
    "get_registry",
    "classify_agent_topology",
    "TopologyResult",
    "PatternStore"
]
