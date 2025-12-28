"""Corrective RAG (CRAG) Fallback Pattern (v2.2)

Implements knowledge sufficiency assessment and external fallback.
When internal knowledge is insufficient, query external sources
(ground control, reference databases) before spawning helpers.

Key insight: CRAG happens BEFORE spawning helpers.
If external knowledge resolves uncertainty, no helpers needed.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core import emit_receipt


@dataclass
class ExternalResult:
    """Result from external knowledge source."""
    source: str
    data: dict
    confidence: float
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class FusedResult:
    """Combined internal and external knowledge."""
    internal_results: List[dict]
    external_result: Optional[ExternalResult]
    fused_confidence: float
    resolution_strategy: str
    requires_helpers: bool


@dataclass
class CRAGResult:
    """Complete CRAG operation result."""
    query_id: str
    decision_id: str
    internal_sufficiency: float
    external_queried: bool
    sources_used: List[str]
    fusion_confidence: float
    latency_ms: float
    resolved: bool


# =============================================================================
# KNOWLEDGE SUFFICIENCY ASSESSMENT
# =============================================================================

def assess_knowledge_sufficiency(query: dict,
                                 internal_results: List[dict]) -> float:
    """Score 0-1 if internal knowledge is sufficient.

    Higher score = more sufficient, less need for external.

    Args:
        query: Query criteria
        internal_results: Results from internal knowledge (temporal graph, etc.)

    Returns:
        Sufficiency score 0-1
    """
    if not internal_results:
        return 0.0

    # Base score from result count
    count_score = min(1.0, len(internal_results) / 5.0)  # 5+ results = full score

    # Score from confidence in results
    confidence_sum = 0.0
    for result in internal_results:
        # Check various confidence fields
        if "confidence" in result:
            confidence_sum += result["confidence"]
        elif "decision" in result and "confidence" in result["decision"]:
            confidence_sum += result["decision"]["confidence"]
        else:
            confidence_sum += 0.5  # Default confidence

    avg_confidence = confidence_sum / len(internal_results) if internal_results else 0.0

    # Score from relevance (if timestamp is recent)
    recency_score = 0.0
    for result in internal_results:
        ts = result.get("timestamp")
        if ts:
            try:
                result_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                hours_old = (now - result_time).total_seconds() / 3600
                if hours_old < 1:
                    recency_score += 1.0
                elif hours_old < 24:
                    recency_score += 0.5
                else:
                    recency_score += 0.2
            except (ValueError, TypeError):
                recency_score += 0.3

    recency_score = recency_score / len(internal_results) if internal_results else 0.0

    # Score from query coverage
    coverage_score = _calculate_coverage(query, internal_results)

    # Weighted combination
    sufficiency = (
        0.3 * count_score +
        0.3 * avg_confidence +
        0.2 * recency_score +
        0.2 * coverage_score
    )

    return min(1.0, max(0.0, sufficiency))


def _calculate_coverage(query: dict, results: List[dict]) -> float:
    """Calculate how well results cover query requirements.

    Args:
        query: Query criteria
        results: Internal results

    Returns:
        Coverage score 0-1
    """
    if not query:
        return 1.0  # No specific requirements = full coverage

    required_keys = set(query.keys())
    covered_keys = set()

    for result in results:
        decision = result.get("decision", result)
        context = result.get("context", {})

        for key in required_keys:
            if key in decision or key in context:
                covered_keys.add(key)

    if not required_keys:
        return 1.0

    return len(covered_keys) / len(required_keys)


# =============================================================================
# EXTERNAL FALLBACK
# =============================================================================

# External source registry (simulated)
_external_sources: Dict[str, callable] = {}


def register_external_source(name: str, handler: callable):
    """Register an external knowledge source.

    Args:
        name: Source name
        handler: Callable that takes query dict and returns dict
    """
    _external_sources[name] = handler


def fallback_to_external(query: dict,
                         sources: Optional[List[str]] = None) -> ExternalResult:
    """Query ground control or reference databases.

    Args:
        query: Query to send to external source
        sources: Specific sources to query, or None for all

    Returns:
        ExternalResult with response
    """
    start_time = time.perf_counter()

    # Default sources
    if sources is None:
        sources = ["ground_control", "reference_db"]

    # Try each source
    for source_name in sources:
        if source_name in _external_sources:
            try:
                handler = _external_sources[source_name]
                result_data = handler(query)
                latency_ms = (time.perf_counter() - start_time) * 1000

                return ExternalResult(
                    source=source_name,
                    data=result_data,
                    confidence=result_data.get("confidence", 0.8),
                    latency_ms=latency_ms,
                    success=True
                )
            except Exception as e:
                continue  # Try next source

    # Simulated ground control response (default behavior)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Simulate ground control knowledge
    simulated_response = _simulate_ground_control(query)

    return ExternalResult(
        source="ground_control",
        data=simulated_response,
        confidence=simulated_response.get("confidence", 0.7),
        latency_ms=latency_ms + 50,  # Add simulated network latency
        success=True
    )


def _simulate_ground_control(query: dict) -> dict:
    """Simulate ground control response.

    Args:
        query: Query to process

    Returns:
        Simulated response
    """
    response = {
        "query_received": True,
        "recommendations": [],
        "confidence": 0.75
    }

    # Generate recommendations based on query
    action_type = query.get("action_type")
    if action_type == "AVOID":
        response["recommendations"] = [
            {"action": "increase_altitude", "priority": "high"},
            {"action": "check_sensors", "priority": "medium"}
        ]
        response["confidence"] = 0.85

    elif action_type == "RTB":
        response["recommendations"] = [
            {"action": "confirm_base_location", "priority": "high"},
            {"action": "check_fuel", "priority": "high"}
        ]
        response["confidence"] = 0.90

    elif action_type == "ABORT":
        response["recommendations"] = [
            {"action": "safe_landing", "priority": "critical"},
            {"action": "notify_operators", "priority": "high"}
        ]
        response["confidence"] = 0.95

    else:
        response["recommendations"] = [
            {"action": "continue_monitoring", "priority": "low"}
        ]
        response["confidence"] = 0.70

    return response


# =============================================================================
# KNOWLEDGE FUSION
# =============================================================================

def fuse_internal_external(internal: List[dict],
                           external: Optional[ExternalResult]) -> FusedResult:
    """Combine internal and external knowledge sources.

    Args:
        internal: Internal knowledge results
        external: External query result (may be None)

    Returns:
        FusedResult with combined knowledge
    """
    # If no external, use internal only
    if external is None or not external.success:
        internal_confidence = _average_confidence(internal)
        return FusedResult(
            internal_results=internal,
            external_result=external,
            fused_confidence=internal_confidence,
            resolution_strategy="internal_only",
            requires_helpers=internal_confidence < 0.7
        )

    # Calculate confidences
    internal_confidence = _average_confidence(internal)
    external_confidence = external.confidence

    # Fusion strategy based on confidence levels
    if external_confidence > internal_confidence + 0.2:
        # External significantly more confident - use external primarily
        fused_confidence = external_confidence * 0.7 + internal_confidence * 0.3
        resolution_strategy = "external_preferred"

    elif internal_confidence > external_confidence + 0.2:
        # Internal significantly more confident - use internal primarily
        fused_confidence = internal_confidence * 0.7 + external_confidence * 0.3
        resolution_strategy = "internal_preferred"

    else:
        # Similar confidence - weighted average with boost
        fused_confidence = (internal_confidence + external_confidence) / 2
        # Boost for having multiple sources
        fused_confidence = min(1.0, fused_confidence * 1.1)
        resolution_strategy = "balanced_fusion"

    # Handle conflicts
    if _has_conflict(internal, external):
        # Conflict detected - lower confidence, may need helpers
        fused_confidence *= 0.8
        resolution_strategy = "conflict_resolution"

    return FusedResult(
        internal_results=internal,
        external_result=external,
        fused_confidence=fused_confidence,
        resolution_strategy=resolution_strategy,
        requires_helpers=fused_confidence < 0.7
    )


def _average_confidence(results: List[dict]) -> float:
    """Calculate average confidence from results.

    Args:
        results: List of result dicts

    Returns:
        Average confidence
    """
    if not results:
        return 0.0

    total = 0.0
    for r in results:
        if "confidence" in r:
            total += r["confidence"]
        elif "decision" in r and "confidence" in r["decision"]:
            total += r["decision"]["confidence"]
        else:
            total += 0.5

    return total / len(results)


def _has_conflict(internal: List[dict], external: ExternalResult) -> bool:
    """Detect if internal and external knowledge conflict.

    Args:
        internal: Internal results
        external: External result

    Returns:
        True if conflict detected
    """
    if not internal or not external or not external.success:
        return False

    external_recommendations = external.data.get("recommendations", [])

    for result in internal:
        decision = result.get("decision", result)
        action = decision.get("action", {})
        action_type = action.get("type", "")

        # Check if external contradicts internal
        for rec in external_recommendations:
            rec_action = rec.get("action", "")

            # Conflict patterns
            if action_type == "CONTINUE" and rec_action in ["abort", "rtb", "avoid"]:
                return True
            if action_type == "AVOID" and rec_action == "continue":
                return True

    return False


# =============================================================================
# UNIFIED CRAG INTERFACE
# =============================================================================

def perform_crag(query: dict, decision_id: str,
                 internal_results: List[dict],
                 external_sources: Optional[List[str]] = None,
                 sufficiency_threshold: float = 0.7) -> CRAGResult:
    """Complete CRAG workflow.

    Args:
        query: Query criteria
        decision_id: Associated decision ID
        internal_results: Results from internal knowledge
        external_sources: Sources to query if needed
        sufficiency_threshold: Threshold for external fallback

    Returns:
        CRAGResult with complete outcome
    """
    start_time = time.perf_counter()
    query_id = str(uuid.uuid4())

    # Assess internal sufficiency
    sufficiency = assess_knowledge_sufficiency(query, internal_results)

    # Decide on external fallback
    external_queried = False
    sources_used = ["internal"]
    external_result = None
    fused = None

    if sufficiency < sufficiency_threshold:
        # Need external knowledge
        external_result = fallback_to_external(query, external_sources)
        external_queried = True
        if external_result.success:
            sources_used.append(external_result.source)

        # Fuse knowledge
        fused = fuse_internal_external(internal_results, external_result)
        fusion_confidence = fused.fused_confidence
    else:
        fusion_confidence = sufficiency

    latency_ms = (time.perf_counter() - start_time) * 1000

    result = CRAGResult(
        query_id=query_id,
        decision_id=decision_id,
        internal_sufficiency=sufficiency,
        external_queried=external_queried,
        sources_used=sources_used,
        fusion_confidence=fusion_confidence,
        latency_ms=latency_ms,
        resolved=fusion_confidence >= 0.7
    )

    # Emit receipt
    emit_crag_receipt(result)

    return result


def emit_crag_receipt(result: CRAGResult) -> dict:
    """Emit CRAG operation receipt.

    Args:
        result: CRAGResult to record

    Returns:
        Receipt dict
    """
    return emit_receipt("crag", {
        "query_id": result.query_id,
        "decision_id": result.decision_id,
        "internal_sufficiency": result.internal_sufficiency,
        "external_queried": result.external_queried,
        "sources_used": result.sources_used,
        "fusion_confidence": result.fusion_confidence,
        "latency_ms": result.latency_ms,
        "resolved": result.resolved
    }, silent=True)
