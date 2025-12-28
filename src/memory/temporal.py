"""Temporal Knowledge Graph - Decision Memory (v2.2)

Implements Graphiti-inspired temporal graph for decision history.
Episodes = Decision events with temporal context
Edges = Causal relationships between decisions
Decay = Older decisions less weighted (configurable tau)

Key insight: Memory as temporal knowledge graph, not vector store.
"""

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import dual_hash, emit_receipt


@dataclass
class Episode:
    """A decision event with temporal context."""
    episode_id: str
    decision_id: str
    decision: dict
    context: dict
    timestamp: str
    weight: float = 1.0
    edges_in: List[str] = field(default_factory=list)  # Episodes that led to this
    edges_out: List[str] = field(default_factory=list)  # Episodes caused by this


@dataclass
class Edge:
    """Causal relationship between episodes."""
    edge_id: str
    source_id: str  # Episode ID
    target_id: str  # Episode ID
    weight: float
    relationship: str  # "caused_by", "related_to", "followed_by"
    created_at: str


class TemporalGraph:
    """Temporal knowledge graph for decision history.

    Implements:
    - Episode storage with temporal decay
    - Bidirectional causality tracing
    - Relevance-based querying
    """

    def __init__(self, decay_tau: float = 0.1):
        """Initialize temporal graph.

        Args:
            decay_tau: Time decay constant (higher = faster decay)
        """
        self._episodes: Dict[str, Episode] = {}
        self._edges: Dict[str, Edge] = {}
        self._decision_to_episode: Dict[str, str] = {}  # decision_id -> episode_id
        self._decay_tau = decay_tau
        self._creation_time = datetime.now(timezone.utc)

    def add_episode(self, decision: dict, context: Optional[dict] = None,
                    caused_by: Optional[List[str]] = None) -> str:
        """Add decision to temporal graph.

        Args:
            decision: Decision dict to add
            context: Context around the decision
            caused_by: List of decision_ids that caused this decision

        Returns:
            Episode ID
        """
        episode_id = str(uuid.uuid4())
        decision_id = decision.get("decision_id", str(uuid.uuid4()))
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create episode
        episode = Episode(
            episode_id=episode_id,
            decision_id=decision_id,
            decision=decision,
            context=context or {},
            timestamp=timestamp,
            weight=1.0
        )

        # Add causal edges
        if caused_by:
            for cause_decision_id in caused_by:
                if cause_decision_id in self._decision_to_episode:
                    cause_episode_id = self._decision_to_episode[cause_decision_id]
                    edge = self._create_edge(
                        cause_episode_id, episode_id, "caused_by"
                    )
                    episode.edges_in.append(edge.edge_id)
                    if cause_episode_id in self._episodes:
                        self._episodes[cause_episode_id].edges_out.append(edge.edge_id)

        # Store
        self._episodes[episode_id] = episode
        self._decision_to_episode[decision_id] = episode_id

        return episode_id

    def _create_edge(self, source_id: str, target_id: str,
                     relationship: str) -> Edge:
        """Create an edge between episodes.

        Args:
            source_id: Source episode ID
            target_id: Target episode ID
            relationship: Type of relationship

        Returns:
            Created Edge
        """
        edge_id = str(uuid.uuid4())
        edge = Edge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            weight=1.0,
            relationship=relationship,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self._edges[edge_id] = edge
        return edge

    def query_relevant(self, query: dict,
                       time_window: Optional[Tuple[str, str]] = None,
                       limit: int = 10) -> List[Episode]:
        """Retrieve relevant past decisions.

        Args:
            query: Query criteria (action_type, confidence range, etc.)
            time_window: Optional (start, end) ISO timestamps
            limit: Maximum results

        Returns:
            List of relevant episodes
        """
        candidates = []

        for episode in self._episodes.values():
            # Apply time window filter
            if time_window:
                start, end = time_window
                if start and episode.timestamp < start:
                    continue
                if end and episode.timestamp > end:
                    continue

            # Calculate relevance score
            relevance = self._calculate_relevance(episode, query)

            # Apply temporal decay
            decayed_weight = self._apply_decay(episode)
            final_score = relevance * decayed_weight

            if final_score > 0.1:  # Threshold for relevance
                candidates.append((episode, final_score))

        # Sort by score and return top results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in candidates[:limit]]

    def _calculate_relevance(self, episode: Episode, query: dict) -> float:
        """Calculate relevance of episode to query.

        Args:
            episode: Episode to score
            query: Query criteria

        Returns:
            Relevance score 0-1
        """
        score = 0.0
        matches = 0

        decision = episode.decision

        # Match action type
        if "action_type" in query:
            decision_action = decision.get("action", {}).get("type", "")
            if decision_action == query["action_type"]:
                score += 0.3
                matches += 1

        # Match confidence range
        if "confidence_min" in query or "confidence_max" in query:
            confidence = decision.get("confidence", 0.5)
            min_conf = query.get("confidence_min", 0)
            max_conf = query.get("confidence_max", 1)
            if min_conf <= confidence <= max_conf:
                score += 0.2
                matches += 1

        # Match context keys
        if "context_keys" in query:
            for key in query["context_keys"]:
                if key in episode.context:
                    score += 0.1
                    matches += 1

        # Match anomaly status
        if "is_anomaly" in query:
            is_anomaly = decision.get("is_anomaly", False)
            if is_anomaly == query["is_anomaly"]:
                score += 0.2
                matches += 1

        # Match decision_id pattern
        if "decision_id_pattern" in query:
            pattern = query["decision_id_pattern"]
            if pattern in episode.decision_id:
                score += 0.4
                matches += 1

        # Normalize score
        if matches > 0:
            score = min(1.0, score)
        else:
            # Default relevance for no specific matches
            score = 0.1

        return score

    def _apply_decay(self, episode: Episode) -> float:
        """Apply temporal decay to episode weight.

        Args:
            episode: Episode to decay

        Returns:
            Decayed weight
        """
        # Parse timestamp
        try:
            ep_time = datetime.fromisoformat(episode.timestamp.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return episode.weight

        now = datetime.now(timezone.utc)
        delta_hours = (now - ep_time).total_seconds() / 3600

        # Exponential decay: w * e^(-tau * t)
        decay_factor = math.exp(-self._decay_tau * delta_hours)

        return episode.weight * decay_factor

    def decay_edges(self, tau: Optional[float] = None) -> int:
        """Apply temporal decay to all edge weights.

        Args:
            tau: Optional decay constant override

        Returns:
            Number of edges modified
        """
        decay_tau = tau if tau is not None else self._decay_tau
        modified = 0
        now = datetime.now(timezone.utc)

        for edge in self._edges.values():
            try:
                edge_time = datetime.fromisoformat(edge.created_at.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                continue

            delta_hours = (now - edge_time).total_seconds() / 3600
            decay_factor = math.exp(-decay_tau * delta_hours)

            old_weight = edge.weight
            edge.weight *= decay_factor

            if abs(old_weight - edge.weight) > 0.001:
                modified += 1

        return modified

    def get_decision_lineage(self, decision_id: str,
                             direction: str = "both",
                             max_depth: int = 5) -> Dict[str, Any]:
        """Trace decision causality.

        Args:
            decision_id: Decision to trace
            direction: "backward" (root cause), "forward" (blast radius), or "both"
            max_depth: Maximum traversal depth

        Returns:
            Graph structure with lineage
        """
        if decision_id not in self._decision_to_episode:
            return {"decision_id": decision_id, "found": False, "lineage": []}

        episode_id = self._decision_to_episode[decision_id]
        episode = self._episodes.get(episode_id)

        if not episode:
            return {"decision_id": decision_id, "found": False, "lineage": []}

        result = {
            "decision_id": decision_id,
            "found": True,
            "episode_id": episode_id,
            "timestamp": episode.timestamp,
            "ancestors": [],
            "descendants": []
        }

        # Trace backward (root cause)
        if direction in ("backward", "both"):
            visited = set()
            result["ancestors"] = self._trace_edges(
                episode, "backward", visited, max_depth
            )

        # Trace forward (blast radius)
        if direction in ("forward", "both"):
            visited = set()
            result["descendants"] = self._trace_edges(
                episode, "forward", visited, max_depth
            )

        return result

    def _trace_edges(self, episode: Episode, direction: str,
                     visited: Set[str], max_depth: int,
                     current_depth: int = 0) -> List[Dict]:
        """Recursively trace edges.

        Args:
            episode: Starting episode
            direction: "backward" or "forward"
            visited: Set of visited episode IDs
            max_depth: Maximum depth
            current_depth: Current depth

        Returns:
            List of lineage entries
        """
        if current_depth >= max_depth:
            return []
        if episode.episode_id in visited:
            return []

        visited.add(episode.episode_id)
        lineage = []

        # Get relevant edges
        if direction == "backward":
            edge_ids = episode.edges_in
        else:
            edge_ids = episode.edges_out

        for edge_id in edge_ids:
            edge = self._edges.get(edge_id)
            if not edge:
                continue

            # Get connected episode
            if direction == "backward":
                connected_id = edge.source_id
            else:
                connected_id = edge.target_id

            connected_episode = self._episodes.get(connected_id)
            if not connected_episode:
                continue

            entry = {
                "episode_id": connected_id,
                "decision_id": connected_episode.decision_id,
                "timestamp": connected_episode.timestamp,
                "relationship": edge.relationship,
                "weight": edge.weight,
                "depth": current_depth + 1
            }

            # Recurse
            children = self._trace_edges(
                connected_episode, direction, visited,
                max_depth, current_depth + 1
            )
            if children:
                entry["children"] = children

            lineage.append(entry)

        return lineage

    def get_stats(self) -> dict:
        """Get graph statistics.

        Returns:
            Stats dict
        """
        oldest_ts = None
        for ep in self._episodes.values():
            if oldest_ts is None or ep.timestamp < oldest_ts:
                oldest_ts = ep.timestamp

        return {
            "nodes": len(self._episodes),
            "edges": len(self._edges),
            "oldest_node": oldest_ts,
            "decay_tau": self._decay_tau
        }

    def prune_old(self, older_than_hours: float = 24.0) -> int:
        """Remove episodes older than threshold.

        Args:
            older_than_hours: Age threshold in hours

        Returns:
            Number of episodes removed
        """
        now = datetime.now(timezone.utc)
        to_remove = []

        for ep_id, episode in self._episodes.items():
            try:
                ep_time = datetime.fromisoformat(episode.timestamp.replace('Z', '+00:00'))
                delta_hours = (now - ep_time).total_seconds() / 3600
                if delta_hours > older_than_hours:
                    to_remove.append(ep_id)
            except (ValueError, TypeError):
                continue

        for ep_id in to_remove:
            episode = self._episodes.pop(ep_id, None)
            if episode:
                # Remove from decision mapping
                if episode.decision_id in self._decision_to_episode:
                    del self._decision_to_episode[episode.decision_id]
                # Remove associated edges
                for edge_id in episode.edges_in + episode.edges_out:
                    self._edges.pop(edge_id, None)

        return len(to_remove)


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_temporal_graph: Optional[TemporalGraph] = None


def get_temporal_graph() -> TemporalGraph:
    """Get the global temporal graph instance."""
    global _temporal_graph
    if _temporal_graph is None:
        _temporal_graph = TemporalGraph()
    return _temporal_graph


def reset_temporal_graph():
    """Reset the global temporal graph (for testing)."""
    global _temporal_graph
    _temporal_graph = TemporalGraph()


def add_episode(decision: dict, context: Optional[dict] = None,
                caused_by: Optional[List[str]] = None) -> str:
    """Add decision to temporal graph.

    Args:
        decision: Decision dict
        context: Decision context
        caused_by: Decision IDs that caused this

    Returns:
        Episode ID
    """
    graph = get_temporal_graph()
    episode_id = graph.add_episode(decision, context, caused_by)

    # Emit receipt
    emit_memory_receipt(episode_id, "ADD", [decision.get("decision_id", "")])

    return episode_id


def query_relevant(query: dict,
                   time_window: Optional[Tuple[str, str]] = None,
                   limit: int = 10) -> List[dict]:
    """Query for relevant past decisions.

    Args:
        query: Query criteria
        time_window: Optional time range
        limit: Max results

    Returns:
        List of relevant decision dicts
    """
    graph = get_temporal_graph()
    episodes = graph.query_relevant(query, time_window, limit)

    # Extract decisions
    results = []
    for ep in episodes:
        results.append({
            "decision_id": ep.decision_id,
            "episode_id": ep.episode_id,
            "decision": ep.decision,
            "context": ep.context,
            "timestamp": ep.timestamp
        })

    # Emit receipt
    emit_memory_receipt(str(uuid.uuid4()), "QUERY",
                        [r["decision_id"] for r in results])

    return results


def decay_edges(tau: Optional[float] = None) -> int:
    """Apply temporal decay to graph edges.

    Args:
        tau: Decay constant

    Returns:
        Number of edges modified
    """
    graph = get_temporal_graph()
    modified = graph.decay_edges(tau)

    emit_memory_receipt(str(uuid.uuid4()), "DECAY", [], modified)

    return modified


def get_decision_lineage(decision_id: str,
                         direction: str = "both",
                         max_depth: int = 5) -> dict:
    """Get causality lineage for a decision.

    Args:
        decision_id: Decision to trace
        direction: "backward", "forward", or "both"
        max_depth: Max traversal depth

    Returns:
        Lineage graph
    """
    graph = get_temporal_graph()
    return graph.get_decision_lineage(decision_id, direction, max_depth)


def emit_memory_receipt(episode_id: str, operation: str,
                        decision_ids: List[str],
                        edges_modified: int = 0) -> dict:
    """Emit memory operation receipt.

    Args:
        episode_id: Episode involved
        operation: ADD, QUERY, DECAY, PRUNE
        decision_ids: Decisions affected
        edges_modified: Number of edges modified

    Returns:
        Receipt dict
    """
    graph = get_temporal_graph()
    stats = graph.get_stats()

    return emit_receipt("memory", {
        "episode_id": episode_id,
        "operation": operation,
        "decision_ids_affected": decision_ids,
        "edges_modified": edges_modified,
        "graph_stats": stats
    }, silent=True)
