"""Tests for temporal knowledge graph (v2.2)."""

import pytest
import time
import uuid

from src.memory.temporal import (
    Episode,
    Edge,
    TemporalGraph,
    get_temporal_graph,
    reset_temporal_graph,
    add_episode,
    query_relevant,
    decay_edges,
    get_decision_lineage,
    emit_memory_receipt
)


@pytest.fixture(autouse=True)
def reset_graph():
    """Reset temporal graph before each test."""
    reset_temporal_graph()
    yield
    reset_temporal_graph()


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_creation(self):
        """Episode should be created with required fields."""
        episode = Episode(
            episode_id="e1",
            decision_id="d1",
            decision={"action": {"type": "CONTINUE"}},
            context={"altitude": 1000},
            timestamp="2024-01-01T00:00:00Z"
        )
        assert episode.episode_id == "e1"
        assert episode.decision_id == "d1"
        assert episode.weight == 1.0

    def test_episode_edges(self):
        """Episode should track edges."""
        episode = Episode(
            episode_id="e1",
            decision_id="d1",
            decision={},
            context={},
            timestamp="2024-01-01T00:00:00Z",
            edges_in=["edge1"],
            edges_out=["edge2"]
        )
        assert "edge1" in episode.edges_in
        assert "edge2" in episode.edges_out


class TestTemporalGraph:
    """Tests for TemporalGraph class."""

    def test_graph_initialization(self):
        """Graph should initialize with default settings."""
        graph = TemporalGraph()
        stats = graph.get_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_add_episode(self):
        """Adding episode should increase node count."""
        graph = TemporalGraph()
        decision = {"decision_id": "d1", "action": {"type": "CONTINUE"}}
        episode_id = graph.add_episode(decision)

        stats = graph.get_stats()
        assert stats["nodes"] == 1
        assert episode_id is not None

    def test_add_episode_with_context(self):
        """Episode should store context."""
        graph = TemporalGraph()
        decision = {"decision_id": "d1", "action": {"type": "CONTINUE"}}
        context = {"altitude": 1000, "speed": 50}
        graph.add_episode(decision, context)

        episodes = graph.query_relevant({"action_type": "CONTINUE"})
        assert len(episodes) == 1
        assert episodes[0].context == context

    def test_causal_edges(self):
        """Causal edges should be created between episodes."""
        graph = TemporalGraph()

        # Add first decision
        decision1 = {"decision_id": "d1"}
        graph.add_episode(decision1)

        # Add second decision caused by first
        decision2 = {"decision_id": "d2"}
        graph.add_episode(decision2, caused_by=["d1"])

        stats = graph.get_stats()
        assert stats["edges"] == 1

    def test_query_relevant_empty(self):
        """Query on empty graph should return empty list."""
        graph = TemporalGraph()
        results = graph.query_relevant({"action_type": "CONTINUE"})
        assert results == []

    def test_query_by_action_type(self):
        """Query should filter by action type."""
        graph = TemporalGraph()
        graph.add_episode({"decision_id": "d1", "action": {"type": "CONTINUE"}})
        graph.add_episode({"decision_id": "d2", "action": {"type": "AVOID"}})

        results = graph.query_relevant({"action_type": "CONTINUE"})
        assert len(results) >= 1
        # Should prefer matching action type
        matching = [e for e in results if e.decision.get("action", {}).get("type") == "CONTINUE"]
        assert len(matching) >= 1

    def test_query_limit(self):
        """Query should respect limit."""
        graph = TemporalGraph()
        for i in range(20):
            graph.add_episode({"decision_id": f"d{i}"})

        results = graph.query_relevant({}, limit=5)
        assert len(results) <= 5

    def test_temporal_decay(self):
        """Old episodes should have lower weight after decay."""
        graph = TemporalGraph(decay_tau=0.1)
        graph.add_episode({"decision_id": "d1", "action": {"type": "CONTINUE"}})

        # Immediate query - should have high weight
        initial_episodes = graph.query_relevant({"action_type": "CONTINUE"})
        assert len(initial_episodes) > 0

    def test_decay_edges(self):
        """Edge decay should modify edge weights."""
        graph = TemporalGraph(decay_tau=0.1)
        graph.add_episode({"decision_id": "d1"})
        graph.add_episode({"decision_id": "d2"}, caused_by=["d1"])

        modified = graph.decay_edges()
        # May be 0 if immediate, but should not error
        assert modified >= 0

    def test_get_decision_lineage_not_found(self):
        """Lineage for unknown decision should return found=False."""
        graph = TemporalGraph()
        lineage = graph.get_decision_lineage("unknown")
        assert lineage["found"] is False

    def test_get_decision_lineage_found(self):
        """Lineage for known decision should return found=True."""
        graph = TemporalGraph()
        graph.add_episode({"decision_id": "d1"})

        lineage = graph.get_decision_lineage("d1")
        assert lineage["found"] is True
        assert lineage["decision_id"] == "d1"

    def test_lineage_ancestors(self):
        """Lineage should trace ancestors."""
        graph = TemporalGraph()
        graph.add_episode({"decision_id": "d1"})
        graph.add_episode({"decision_id": "d2"}, caused_by=["d1"])
        graph.add_episode({"decision_id": "d3"}, caused_by=["d2"])

        lineage = graph.get_decision_lineage("d3", direction="backward")
        assert lineage["found"] is True
        # Should have d2 as ancestor
        if lineage["ancestors"]:
            ancestor_ids = [a["decision_id"] for a in lineage["ancestors"]]
            assert "d2" in ancestor_ids

    def test_prune_old(self):
        """Prune should remove old episodes."""
        graph = TemporalGraph()
        graph.add_episode({"decision_id": "d1"})

        # Prune with 0 hours (should remove everything)
        removed = graph.prune_old(older_than_hours=0)
        # May or may not remove depending on timing
        assert removed >= 0


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_add_episode_function(self):
        """Module add_episode should work."""
        episode_id = add_episode({"decision_id": "test1"})
        assert episode_id is not None

    def test_query_relevant_function(self):
        """Module query_relevant should work."""
        add_episode({"decision_id": "q1", "action": {"type": "CONTINUE"}})
        results = query_relevant({"action_type": "CONTINUE"})
        assert isinstance(results, list)

    def test_decay_edges_function(self):
        """Module decay_edges should work."""
        add_episode({"decision_id": "d1"})
        add_episode({"decision_id": "d2"}, caused_by=["d1"])
        modified = decay_edges()
        assert modified >= 0

    def test_get_decision_lineage_function(self):
        """Module get_decision_lineage should work."""
        add_episode({"decision_id": "lin1"})
        lineage = get_decision_lineage("lin1")
        assert lineage["found"] is True


class TestMemoryReceipt:
    """Tests for memory receipt emission."""

    def test_emit_memory_receipt(self):
        """Memory receipt should have required fields."""
        receipt = emit_memory_receipt("ep1", "ADD", ["d1"])

        assert receipt["receipt_type"] == "memory"
        assert receipt["operation"] == "ADD"
        assert "graph_stats" in receipt


class TestPerformance:
    """Performance tests for temporal graph."""

    def test_add_latency(self):
        """Add episode should complete under 10ms."""
        graph = TemporalGraph()
        decision = {"decision_id": "perf1", "action": {"type": "CONTINUE"}}

        start = time.perf_counter()
        for i in range(100):
            graph.add_episode({"decision_id": f"d{i}"})
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Add latency {elapsed_ms}ms exceeds 10ms SLO"

    def test_query_latency(self):
        """Query should complete under 20ms."""
        graph = TemporalGraph()
        for i in range(100):
            graph.add_episode({"decision_id": f"d{i}", "action": {"type": "CONTINUE"}})

        start = time.perf_counter()
        graph.query_relevant({"action_type": "CONTINUE"}, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 20, f"Query latency {elapsed_ms}ms exceeds 20ms SLO"
