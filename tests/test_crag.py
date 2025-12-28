"""Tests for CRAG fallback pattern (v2.2)."""

import pytest
import time

from src.knowledge.crag import (
    ExternalResult,
    FusedResult,
    CRAGResult,
    assess_knowledge_sufficiency,
    fallback_to_external,
    fuse_internal_external,
    perform_crag,
    emit_crag_receipt,
    register_external_source,
    _external_sources
)


@pytest.fixture(autouse=True)
def clear_sources():
    """Clear external sources before each test."""
    _external_sources.clear()
    yield
    _external_sources.clear()


class TestKnowledgeSufficiency:
    """Tests for knowledge sufficiency assessment."""

    def test_empty_results_zero_sufficiency(self):
        """Empty results should have zero sufficiency."""
        sufficiency = assess_knowledge_sufficiency({}, [])
        assert sufficiency == 0.0

    def test_single_result_partial_sufficiency(self):
        """Single result should have partial sufficiency."""
        results = [{"confidence": 0.8}]
        sufficiency = assess_knowledge_sufficiency({}, results)
        assert 0 < sufficiency < 1.0

    def test_multiple_results_higher_sufficiency(self):
        """More results should increase sufficiency."""
        single_result = [{"confidence": 0.8}]
        multiple_results = [{"confidence": 0.8} for _ in range(5)]

        single_suff = assess_knowledge_sufficiency({}, single_result)
        multi_suff = assess_knowledge_sufficiency({}, multiple_results)

        assert multi_suff > single_suff

    def test_high_confidence_increases_sufficiency(self):
        """Higher confidence should increase sufficiency."""
        low_conf = [{"confidence": 0.3}]
        high_conf = [{"confidence": 0.95}]

        low_suff = assess_knowledge_sufficiency({}, low_conf)
        high_suff = assess_knowledge_sufficiency({}, high_conf)

        assert high_suff > low_suff

    def test_recency_affects_sufficiency(self):
        """Recent results should have higher sufficiency."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        old = "2020-01-01T00:00:00Z"

        recent_result = [{"confidence": 0.8, "timestamp": now}]
        old_result = [{"confidence": 0.8, "timestamp": old}]

        recent_suff = assess_knowledge_sufficiency({}, recent_result)
        old_suff = assess_knowledge_sufficiency({}, old_result)

        assert recent_suff >= old_suff


class TestExternalFallback:
    """Tests for external knowledge fallback."""

    def test_fallback_returns_result(self):
        """Fallback should always return a result."""
        result = fallback_to_external({"action_type": "CONTINUE"})
        assert isinstance(result, ExternalResult)
        assert result.success is True

    def test_fallback_includes_latency(self):
        """Fallback should measure latency."""
        result = fallback_to_external({})
        assert result.latency_ms > 0

    def test_registered_source_used(self):
        """Registered source should be used."""
        def custom_source(query):
            return {"custom": True, "confidence": 0.99}

        register_external_source("custom", custom_source)
        result = fallback_to_external({}, sources=["custom"])

        assert result.source == "custom"
        assert result.data.get("custom") is True

    def test_avoid_action_recommendations(self):
        """AVOID action should get specific recommendations."""
        result = fallback_to_external({"action_type": "AVOID"})
        recommendations = result.data.get("recommendations", [])
        assert len(recommendations) > 0

    def test_abort_action_recommendations(self):
        """ABORT action should get critical recommendations."""
        result = fallback_to_external({"action_type": "ABORT"})
        recommendations = result.data.get("recommendations", [])
        # Should have critical priority
        priorities = [r.get("priority") for r in recommendations]
        assert "critical" in priorities or "high" in priorities


class TestKnowledgeFusion:
    """Tests for internal/external knowledge fusion."""

    def test_internal_only_fusion(self):
        """Fusion without external should use internal."""
        internal = [{"confidence": 0.8}]
        fused = fuse_internal_external(internal, None)

        assert fused.resolution_strategy == "internal_only"
        assert fused.external_result is None

    def test_external_preferred_fusion(self):
        """Higher external confidence should prefer external."""
        internal = [{"confidence": 0.5}]
        external = ExternalResult(
            source="ground_control",
            data={"recommendations": []},
            confidence=0.9,
            latency_ms=10,
            success=True
        )

        fused = fuse_internal_external(internal, external)
        assert fused.resolution_strategy == "external_preferred"

    def test_balanced_fusion(self):
        """Similar confidence should use balanced fusion."""
        internal = [{"confidence": 0.7}]
        external = ExternalResult(
            source="ground_control",
            data={"recommendations": []},
            confidence=0.75,
            latency_ms=10,
            success=True
        )

        fused = fuse_internal_external(internal, external)
        assert fused.resolution_strategy == "balanced_fusion"

    def test_fusion_confidence_boost(self):
        """Multiple sources should boost confidence."""
        internal = [{"confidence": 0.7}]
        external = ExternalResult(
            source="ground_control",
            data={"recommendations": []},
            confidence=0.7,
            latency_ms=10,
            success=True
        )

        fused = fuse_internal_external(internal, external)
        # Balanced fusion should boost
        assert fused.fused_confidence >= 0.7

    def test_requires_helpers_low_confidence(self):
        """Low fusion confidence should require helpers."""
        internal = [{"confidence": 0.3}]
        fused = fuse_internal_external(internal, None)

        assert fused.requires_helpers is True

    def test_no_helpers_high_confidence(self):
        """High fusion confidence should not require helpers."""
        internal = [{"confidence": 0.9}]
        external = ExternalResult(
            source="ground_control",
            data={"recommendations": []},
            confidence=0.9,
            latency_ms=10,
            success=True
        )

        fused = fuse_internal_external(internal, external)
        assert fused.requires_helpers is False


class TestPerformCRAG:
    """Tests for complete CRAG workflow."""

    def test_crag_with_sufficient_internal(self):
        """Sufficient internal knowledge should not query external."""
        internal = [{"confidence": 0.9} for _ in range(5)]
        result = perform_crag(
            query={},
            decision_id="d1",
            internal_results=internal,
            sufficiency_threshold=0.7
        )

        assert result.external_queried is False
        assert result.resolved is True

    def test_crag_with_insufficient_internal(self):
        """Insufficient internal should query external."""
        internal = [{"confidence": 0.3}]
        result = perform_crag(
            query={},
            decision_id="d1",
            internal_results=internal,
            sufficiency_threshold=0.7
        )

        assert result.external_queried is True
        assert len(result.sources_used) > 1

    def test_crag_latency_tracked(self):
        """CRAG should track latency."""
        result = perform_crag(
            query={},
            decision_id="d1",
            internal_results=[],
            sufficiency_threshold=0.7
        )

        assert result.latency_ms > 0

    def test_crag_empty_internal(self):
        """Empty internal should trigger external."""
        result = perform_crag(
            query={"action_type": "AVOID"},
            decision_id="d1",
            internal_results=[],
            sufficiency_threshold=0.7
        )

        assert result.internal_sufficiency == 0.0
        assert result.external_queried is True


class TestCRAGReceipt:
    """Tests for CRAG receipt emission."""

    def test_emit_crag_receipt(self):
        """CRAG receipt should have required fields."""
        result = CRAGResult(
            query_id="q1",
            decision_id="d1",
            internal_sufficiency=0.5,
            external_queried=True,
            sources_used=["internal", "ground_control"],
            fusion_confidence=0.8,
            latency_ms=15.5,
            resolved=True
        )
        receipt = emit_crag_receipt(result)

        assert receipt["receipt_type"] == "crag"
        assert receipt["query_id"] == "q1"
        assert receipt["decision_id"] == "d1"
        assert receipt["external_queried"] is True


class TestPerformance:
    """Performance tests for CRAG."""

    def test_sufficiency_assessment_latency(self):
        """Sufficiency assessment should be fast."""
        results = [{"confidence": 0.8, "timestamp": "2024-01-01T00:00:00Z"} for _ in range(100)]

        start = time.perf_counter()
        assess_knowledge_sufficiency({"action_type": "CONTINUE"}, results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Assessment latency {elapsed_ms}ms exceeds 10ms"

    def test_crag_total_latency(self):
        """Complete CRAG should complete under SLO."""
        internal = [{"confidence": 0.5}]  # Will trigger external

        start = time.perf_counter()
        perform_crag(
            query={},
            decision_id="perf1",
            internal_results=internal,
            sufficiency_threshold=0.7
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # External can take time, so allow up to 200ms
        assert elapsed_ms < 200, f"CRAG latency {elapsed_ms}ms exceeds 200ms SLO"
