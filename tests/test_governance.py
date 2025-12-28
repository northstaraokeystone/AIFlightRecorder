"""Tests for governance modules (v2.1)."""

import pytest
import time

from src.governance import (
    assign_accountability,
    get_raci_for_decision,
    validate_raci_coverage,
    capture_provenance,
    validate_reason_code,
    get_reason_code_info,
    route_escalation,
    get_escalation_path
)
from src.governance.provenance import compare_provenance
from src.governance.raci import RACIAssignment, RACIMatrix, emit_raci_receipt
from src.governance.provenance import ProvenanceRecord, ProvenanceTracker, emit_provenance_receipt
from src.governance.reason_codes import emit_intervention_receipt, InterventionTracker
from src.governance.escalation import EscalationLevel, EscalationPath, EscalationRouter, emit_escalation_receipt


class TestRACIAssignment:
    """Tests for RACI accountability assignment."""

    def test_assign_accountability(self):
        """Should assign RACI roles."""
        decision = {"decision_id": "d1", "action": {"type": "CONTINUE"}}
        assignment = assign_accountability(decision, "continue")

        assert assignment is not None
        assert assignment.responsible is not None
        assert assignment.accountable is not None

    def test_default_assignment(self):
        """Unknown decision type should use defaults."""
        decision = {"decision_id": "d1"}
        assignment = assign_accountability(decision, "unknown_type")

        assert assignment.responsible == "ai_system"
        assert assignment.accountable == "operator"

    def test_avoidance_assignment(self):
        """Avoidance decisions should have safety officer."""
        decision = {"decision_id": "d1", "action": {"type": "AVOID"}}
        assignment = assign_accountability(decision, "avoid")

        assert assignment.accountable == "safety_officer"

    def test_emergency_assignment(self):
        """Emergency decisions should have full escalation."""
        decision = {"decision_id": "d1", "action": {"type": "ABORT"}}
        assignment = assign_accountability(decision, "emergency")

        assert "ground_control" in assignment.consulted or "safety_systems" in assignment.consulted
        assert len(assignment.informed) > 2

    def test_get_raci_for_decision(self):
        """get_raci_for_decision should work."""
        decision = {"decision_id": "d1", "action": {"type": "CONTINUE"}}
        raci = get_raci_for_decision(decision)

        assert "responsible" in raci
        assert "accountable" in raci

    def test_validate_raci_coverage(self):
        """validate_raci_coverage should check assignments."""
        # Should return coverage percentage
        coverage = validate_raci_coverage([], [])
        assert 0.0 <= coverage <= 1.0


class TestRACIReceipt:
    """Tests for RACI receipt emission."""

    def test_emit_raci_receipt(self):
        """RACI receipt should have required fields."""
        assignment = RACIAssignment(
            decision_id="d1",
            decision_type="continue",
            responsible="ai_system",
            accountable="operator",
            consulted=[],
            informed=["audit_log"]
        )
        receipt = emit_raci_receipt(assignment)

        assert receipt["receipt_type"] == "raci"
        assert receipt["decision_id"] == "d1"
        assert receipt["responsible"] == "ai_system"


class TestProvenance:
    """Tests for provenance tracking."""

    def test_capture_provenance(self):
        """Should capture model/policy provenance."""
        record = capture_provenance("d1")

        assert record is not None
        assert record.decision_id == "d1"
        assert record.model_version is not None

    def test_provenance_has_hashes(self):
        """Provenance should include hashes."""
        record = capture_provenance("d1")

        assert record.model_hash is not None
        assert ":" in record.model_hash  # Dual hash format

    def test_compare_provenance_same(self):
        """Same provenance should match."""
        record1 = capture_provenance("d1")
        record2 = capture_provenance("d2")

        # Same model should match
        comparison = compare_provenance(record1, record2)
        assert comparison["model_match"] is True

    def test_provenance_tracker(self):
        """ProvenanceTracker should track changes."""
        tracker = ProvenanceTracker()
        record = tracker.capture("d1")

        assert record is not None
        changes = tracker.get_changes_since(record.captured_at)
        assert isinstance(changes, list)


class TestProvenanceReceipt:
    """Tests for provenance receipt emission."""

    def test_emit_provenance_receipt(self):
        """Provenance receipt should have required fields."""
        record = capture_provenance("d1")
        receipt = emit_provenance_receipt(record)

        assert receipt["receipt_type"] == "provenance"
        assert receipt["decision_id"] == "d1"
        assert "model_version" in receipt


class TestReasonCodes:
    """Tests for reason code validation."""

    def test_validate_known_code(self):
        """Known codes should validate."""
        is_valid = validate_reason_code("SAFETY_CRITICAL")
        assert is_valid is True

    def test_validate_unknown_code(self):
        """Unknown codes should not validate."""
        is_valid = validate_reason_code("UNKNOWN_CODE")
        assert is_valid is False

    def test_get_reason_code_info(self):
        """Should get code information."""
        info = get_reason_code_info("SAFETY_CRITICAL")

        assert info is not None
        assert info["category"] == "safety"
        assert info["requires_report"] is True

    def test_model_error_code(self):
        """MODEL_ERROR code should require training."""
        info = get_reason_code_info("MODEL_ERROR")

        assert info["category"] == "model"
        assert info["auto_training"] is True

    def test_testing_code(self):
        """TESTING code should not require report."""
        info = get_reason_code_info("TESTING")

        assert info["category"] == "testing"
        assert info["requires_report"] is False


class TestInterventionReceipt:
    """Tests for intervention receipt emission."""

    def test_emit_intervention_receipt(self):
        """Intervention receipt should have required fields."""
        receipt = emit_intervention_receipt(
            decision_id="d1",
            reason_code="MODEL_ERROR",
            correction={"action": {"type": "AVOID"}},
            operator_id="op1"
        )

        assert receipt["receipt_type"] == "intervention"
        assert receipt["decision_id"] == "d1"
        assert receipt["reason_code"] == "MODEL_ERROR"


class TestEscalation:
    """Tests for escalation routing."""

    def test_route_escalation_low_severity(self):
        """Low severity should go to operator."""
        path = route_escalation("d1", severity=0.3, confidence=0.8)

        assert path.current_level == EscalationLevel.OPERATOR

    def test_route_escalation_high_severity(self):
        """High severity should escalate higher."""
        path = route_escalation("d1", severity=0.9, confidence=0.5)

        assert path.current_level.value >= EscalationLevel.SUPERVISOR.value

    def test_route_escalation_low_confidence(self):
        """Low confidence should trigger escalation."""
        path = route_escalation("d1", severity=0.5, confidence=0.3)

        # Should escalate due to low confidence
        assert path.current_level.value >= EscalationLevel.OPERATOR.value

    def test_get_escalation_path(self):
        """get_escalation_path should return path details."""
        path_dict = get_escalation_path("d1", severity=0.9, confidence=0.4)

        assert "decision_id" in path_dict
        assert "current_level" in path_dict
        assert "escalation_reasons" in path_dict

    def test_escalation_reasons(self):
        """Escalation reasons should be captured."""
        path = route_escalation("d1", severity=0.95, confidence=0.3)

        assert len(path.escalation_reasons) > 0


class TestEscalationReceipt:
    """Tests for escalation receipt emission."""

    def test_emit_escalation_receipt(self):
        """Escalation receipt should have required fields."""
        path = EscalationPath(
            decision_id="d1",
            original_level=EscalationLevel.OPERATOR,
            current_level=EscalationLevel.SAFETY_OFFICER,
            escalation_reasons=["high_severity"],
            escalated_at="2024-01-01T00:00:00Z"
        )
        receipt = emit_escalation_receipt(path)

        assert receipt["receipt_type"] == "escalation"
        assert receipt["decision_id"] == "d1"
        assert "current_level" in receipt


class TestInterventionTracker:
    """Tests for intervention tracking."""

    def test_tracker_record(self):
        """Tracker should record interventions."""
        tracker = InterventionTracker()
        intervention_id = tracker.record(
            decision_id="d1",
            reason_code="TESTING",
            correction={},
            operator_id="op1"
        )

        assert intervention_id is not None

    def test_tracker_get_by_decision(self):
        """Tracker should retrieve by decision."""
        tracker = InterventionTracker()
        tracker.record(
            decision_id="d1",
            reason_code="TESTING",
            correction={},
            operator_id="op1"
        )

        interventions = tracker.get_by_decision("d1")
        assert len(interventions) == 1


class TestPerformance:
    """Performance tests for governance modules."""

    def test_raci_assignment_latency(self):
        """RACI assignment should be fast."""
        decision = {"decision_id": "perf1", "action": {"type": "CONTINUE"}}

        start = time.perf_counter()
        for _ in range(100):
            assign_accountability(decision, "continue")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 5, f"RACI latency {elapsed_ms}ms exceeds 5ms"

    def test_provenance_capture_latency(self):
        """Provenance capture should be fast."""
        start = time.perf_counter()
        for i in range(100):
            capture_provenance(f"d{i}")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Provenance latency {elapsed_ms}ms exceeds 10ms"

    def test_escalation_routing_latency(self):
        """Escalation routing should be fast."""
        start = time.perf_counter()
        for i in range(100):
            route_escalation(f"d{i}", severity=0.5, confidence=0.7)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 5, f"Escalation latency {elapsed_ms}ms exceeds 5ms"
