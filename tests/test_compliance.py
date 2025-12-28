"""Tests for compliance modules (v2.1)."""

import pytest
import time

from src.compliance import (
    AuditTrail,
    generate_audit_trail,
    emit_audit_trail_receipt,
    get_audit_summary,
    ProvenanceReport,
    generate_provenance_report,
    detect_provenance_drift
)
from src.compliance.audit_trail import _generate_findings, _determine_compliance
from src.compliance.provenance_report import get_current_provenance


class TestAuditTrailGeneration:
    """Tests for audit trail generation."""

    def test_generate_empty_trail(self):
        """Should generate trail with no receipts."""
        trail = generate_audit_trail(include_details=False)

        assert trail is not None
        assert trail.trail_id is not None
        assert trail.decision_count >= 0

    def test_generate_trail_with_time_range(self):
        """Should filter by time range."""
        trail = generate_audit_trail(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-12-31T23:59:59Z",
            include_details=False
        )

        assert trail.start_time == "2024-01-01T00:00:00Z"
        assert trail.end_time == "2024-12-31T23:59:59Z"

    def test_trail_has_coverage_metrics(self):
        """Trail should include coverage metrics."""
        trail = generate_audit_trail(include_details=False)

        assert hasattr(trail, "raci_coverage")
        assert hasattr(trail, "provenance_coverage")
        assert 0.0 <= trail.raci_coverage <= 1.0
        assert 0.0 <= trail.provenance_coverage <= 1.0

    def test_trail_has_evidence(self):
        """Trail should include evidence summary."""
        trail = generate_audit_trail(include_details=False)

        assert trail.evidence_summary is not None
        assert hasattr(trail.evidence_summary, "confidence")

    def test_trail_compliance_status(self):
        """Trail should have compliance status."""
        trail = generate_audit_trail(include_details=False)

        assert trail.compliance_status in ["compliant", "non_compliant", "review_required"]


class TestAuditFindings:
    """Tests for audit finding generation."""

    def test_findings_for_unresolved_anomalies(self):
        """Should detect unresolved anomalies."""
        receipts = [
            {"receipt_type": "anomaly", "anomaly_id": "a1"}
        ]
        anomalies = [{"anomaly_id": "a1"}]

        findings = _generate_findings(receipts, [], [], anomalies)

        unresolved = [f for f in findings if f["type"] == "unresolved_anomalies"]
        assert len(unresolved) > 0

    def test_findings_for_missing_raci(self):
        """Should detect missing RACI assignments."""
        receipts = []
        decisions = [{"decision_id": "d1"}, {"decision_id": "d2"}]

        findings = _generate_findings(receipts, decisions, [], [])

        missing_raci = [f for f in findings if f["type"] == "missing_raci"]
        assert len(missing_raci) > 0
        assert missing_raci[0]["count"] == 2

    def test_findings_severity_levels(self):
        """Findings should have severity levels."""
        receipts = [{"receipt_type": "anomaly", "anomaly_id": "a1", "severity": 0.9}]
        anomalies = [{"anomaly_id": "a1", "severity": 0.9}]

        findings = _generate_findings(receipts, [], [], anomalies)

        for finding in findings:
            assert "severity" in finding
            assert finding["severity"] in ["info", "low", "medium", "high", "critical"]


class TestComplianceDetermination:
    """Tests for compliance status determination."""

    def test_compliant_status(self):
        """High coverage, no findings = compliant."""
        status = _determine_compliance(
            raci_coverage=0.98,
            provenance_coverage=0.99,
            findings=[]
        )

        assert status == "compliant"

    def test_non_compliant_high_severity(self):
        """High severity findings = non_compliant."""
        findings = [{"severity": "high", "type": "missing_raci"}]

        status = _determine_compliance(
            raci_coverage=0.95,
            provenance_coverage=0.95,
            findings=findings
        )

        assert status == "non_compliant"

    def test_review_required_low_coverage(self):
        """Low coverage = review_required."""
        status = _determine_compliance(
            raci_coverage=0.80,
            provenance_coverage=0.95,
            findings=[]
        )

        assert status == "review_required"


class TestAuditSummary:
    """Tests for quick audit summary."""

    def test_get_audit_summary(self):
        """Should return summary without full trail."""
        summary = get_audit_summary()

        assert "total_receipts" in summary
        assert "by_type" in summary
        assert "time_range" in summary

    def test_summary_time_filtered(self):
        """Summary should respect time filter."""
        summary = get_audit_summary(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-06-01T00:00:00Z"
        )

        assert summary["time_range"]["start"] == "2024-01-01T00:00:00Z"


class TestAuditTrailReceipt:
    """Tests for audit trail receipt emission."""

    def test_emit_audit_trail_receipt(self):
        """Audit trail receipt should have required fields."""
        from src.proof import Evidence

        trail = AuditTrail(
            trail_id="t1",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-12-31T23:59:59Z",
            generated_at="2024-01-15T00:00:00Z",
            decision_count=100,
            intervention_count=5,
            anomaly_count=2,
            evidence_summary=Evidence(
                evidence_id="e1",
                summary="Test summary",
                source_receipts=[],
                confidence=0.85,
                dialectical_record={"pro": [], "con": [], "gaps": []}
            ),
            raci_coverage=0.95,
            provenance_coverage=0.98,
            findings=[],
            compliance_status="compliant"
        )

        receipt = emit_audit_trail_receipt(trail)

        assert receipt["receipt_type"] == "audit_trail"
        assert receipt["trail_id"] == "t1"
        assert receipt["compliance_status"] == "compliant"


class TestProvenanceReport:
    """Tests for provenance reporting."""

    def test_generate_provenance_report(self):
        """Should generate provenance report."""
        report = generate_provenance_report()

        assert report is not None
        assert report.report_id is not None

    def test_report_has_model_info(self):
        """Report should include model information."""
        report = generate_provenance_report()

        assert report.current_model_version is not None
        assert report.current_model_hash is not None

    def test_report_drift_detection(self):
        """Report should include drift detection."""
        report = generate_provenance_report()

        assert hasattr(report, "drift_detected")
        assert isinstance(report.drift_detected, bool)


class TestProvenanceDrift:
    """Tests for provenance drift detection."""

    def test_detect_drift_no_baseline(self):
        """Should handle no baseline case."""
        drift = detect_provenance_drift()

        assert "drift_detected" in drift
        # Without baseline, should not detect drift

    def test_detect_drift_same_version(self):
        """Same version should not drift."""
        current = get_current_provenance()

        drift = detect_provenance_drift(baseline=current)

        assert drift["drift_detected"] is False

    def test_drift_types(self):
        """Should identify drift types."""
        drift = detect_provenance_drift()

        assert "drift_types" in drift
        assert isinstance(drift["drift_types"], list)


class TestCurrentProvenance:
    """Tests for current provenance capture."""

    def test_get_current_provenance(self):
        """Should get current system provenance."""
        prov = get_current_provenance()

        assert "model_version" in prov
        assert "model_hash" in prov
        assert "captured_at" in prov


class TestPerformance:
    """Performance tests for compliance modules."""

    def test_audit_summary_latency(self):
        """Audit summary should be fast."""
        start = time.perf_counter()
        get_audit_summary()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Summary should be quick even with many receipts
        assert elapsed_ms < 100, f"Summary latency {elapsed_ms}ms exceeds 100ms"

    def test_compliance_determination_latency(self):
        """Compliance determination should be fast."""
        findings = [{"severity": "medium", "type": "test"} for _ in range(100)]

        start = time.perf_counter()
        _determine_compliance(0.95, 0.95, findings)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Determination latency {elapsed_ms}ms exceeds 10ms"

    def test_provenance_report_latency(self):
        """Provenance report should complete reasonably fast."""
        start = time.perf_counter()
        generate_provenance_report()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Allow more time as this involves disk I/O
        assert elapsed_ms < 200, f"Report latency {elapsed_ms}ms exceeds 200ms"
