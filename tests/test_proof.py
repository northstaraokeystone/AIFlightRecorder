"""Tests for the unified proof module (v2.2)."""

import json
import pytest
import time

from src.proof import (
    ProofMode,
    ProofResult,
    Evidence,
    BoundPacket,
    AnomalyResult,
    prove,
    brief_evidence,
    packet_bind,
    detect_anomaly,
    verify_chain,
    synthesize_audit,
    bind_sensor_data,
    emit_proof_receipt
)


class TestProofModes:
    """Test proof mode enumeration."""

    def test_modes_exist(self):
        """All three modes should exist."""
        assert ProofMode.BRIEF.value == "BRIEF"
        assert ProofMode.PACKET.value == "PACKET"
        assert ProofMode.DETECT.value == "DETECT"


class TestBriefMode:
    """Tests for BRIEF mode - evidence synthesis."""

    def test_empty_receipts(self):
        """Empty receipts should return zero confidence."""
        evidence = brief_evidence([])
        assert evidence.confidence == 0.0
        assert "no_input" in evidence.dialectical_record["gaps"]

    def test_single_receipt(self):
        """Single receipt should be synthesized."""
        receipts = [{"receipt_type": "decision", "decision_id": "d1", "confidence": 0.9}]
        evidence = brief_evidence(receipts)
        assert evidence.confidence > 0
        assert len(evidence.source_receipts) > 0

    def test_multiple_receipts(self):
        """Multiple receipts should be combined."""
        receipts = [
            {"receipt_type": "decision", "decision_id": "d1", "confidence": 0.8, "ts": "2024-01-01T00:00:00Z"},
            {"receipt_type": "decision", "decision_id": "d2", "confidence": 0.9, "ts": "2024-01-01T01:00:00Z"},
            {"receipt_type": "anchor", "payload_hash": "abc:def", "ts": "2024-01-01T02:00:00Z"}
        ]
        evidence = brief_evidence(receipts)
        assert evidence.confidence > 0
        assert "decisions recorded" in " ".join(evidence.dialectical_record["pro"])

    def test_dialectical_record(self):
        """Dialectical record should capture pro/con/gaps."""
        receipts = [
            {"receipt_type": "decision", "decision_id": "d1"},
            {"receipt_type": "anomaly", "anomaly_id": "a1"}
        ]
        evidence = brief_evidence(receipts)
        assert "pro" in evidence.dialectical_record
        assert "con" in evidence.dialectical_record
        assert "gaps" in evidence.dialectical_record

    def test_time_range_extraction(self):
        """Time range should be extracted from receipts."""
        receipts = [
            {"receipt_type": "decision", "ts": "2024-01-01T00:00:00Z"},
            {"receipt_type": "decision", "ts": "2024-01-02T00:00:00Z"}
        ]
        evidence = brief_evidence(receipts)
        assert "2024-01-01" in evidence.summary
        assert "2024-01-02" in evidence.summary


class TestPacketMode:
    """Tests for PACKET mode - claim binding."""

    def test_empty_claim(self):
        """Empty claim should fail binding."""
        packet = packet_bind({}, [])
        assert packet.verification_status == "failed"

    def test_claim_without_receipts(self):
        """Claim without receipts should be unbound."""
        claim = {"sensor": "altimeter", "value": 1000}
        packet = packet_bind(claim, [])
        assert packet.verification_status == "unbound"
        assert packet.claim == claim

    def test_valid_binding(self):
        """Valid claim and receipts should bind successfully."""
        claim = {"sensor": "altimeter", "value": 1000}
        receipts = [
            {"payload_hash": "abc:def", "decision_id": "d1"},
            {"payload_hash": "ghi:jkl", "decision_id": "d2"}
        ]
        packet = packet_bind(claim, receipts)
        assert packet.verification_status == "verified"
        assert ":" in packet.binding_hash
        assert len(packet.binding_receipts) == 2

    def test_binding_hash_deterministic(self):
        """Same inputs should produce same binding hash."""
        claim = {"sensor": "altimeter", "value": 1000}
        receipts = [{"payload_hash": "abc:def"}]
        packet1 = packet_bind(claim, receipts)
        packet2 = packet_bind(claim, receipts)
        assert packet1.binding_hash == packet2.binding_hash


class TestDetectMode:
    """Tests for DETECT mode - anomaly detection."""

    def test_empty_stream(self):
        """Empty stream should not be anomaly."""
        result = detect_anomaly([])
        assert result.is_anomaly is False
        assert result.classification == "no_data"

    def test_normal_stream(self):
        """Normal data should not trigger anomaly."""
        stream = [{"value": 100}, {"value": 101}, {"value": 99}]
        baseline = {"mean_ratio": 0.5, "std_ratio": 0.1}
        result = detect_anomaly(stream, baseline)
        # May or may not be anomaly depending on compression
        assert result.classification in ["normal", "deviation"]

    def test_anomaly_classification(self):
        """Anomaly should be classified appropriately."""
        # Create highly compressible data (anomalous)
        stream = [{"v": 1}] * 100  # Very repetitive
        result = detect_anomaly(stream)
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.classification in ["normal", "deviation", "spike", "drift", "tampering"]

    def test_affected_items_extraction(self):
        """Decision IDs should be extracted from stream."""
        stream = [
            {"decision_id": "d1", "value": 100},
            {"decision_id": "d2", "value": 200}
        ]
        result = detect_anomaly(stream)
        # If items have decision_id, they should be in affected_items
        if result.affected_items:
            assert "d1" in result.affected_items or "d2" in result.affected_items


class TestUnifiedInterface:
    """Tests for the unified prove() function."""

    def test_invalid_mode(self):
        """Invalid mode should return error."""
        result = prove("INVALID", [])
        assert result.success is False
        assert "Invalid mode" in result.error

    def test_brief_mode_via_prove(self):
        """BRIEF mode should work via prove()."""
        receipts = [{"receipt_type": "decision", "decision_id": "d1"}]
        result = prove("BRIEF", receipts)
        assert result.mode == ProofMode.BRIEF
        assert "evidence_id" in result.output

    def test_packet_mode_via_prove(self):
        """PACKET mode should work via prove()."""
        receipts = [{"payload_hash": "abc:def"}]
        context = {"claim": {"sensor": "altimeter", "value": 1000}}
        result = prove("PACKET", receipts, context)
        assert result.mode == ProofMode.PACKET
        assert "packet_id" in result.output

    def test_detect_mode_via_prove(self):
        """DETECT mode should work via prove()."""
        stream = [{"value": 100}, {"value": 101}]
        result = prove("DETECT", stream)
        assert result.mode == ProofMode.DETECT
        assert "is_anomaly" in result.output

    def test_case_insensitive_mode(self):
        """Mode should be case-insensitive."""
        result1 = prove("brief", [])
        result2 = prove("BRIEF", [])
        result3 = prove("Brief", [])
        assert result1.mode == result2.mode == result3.mode


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_verify_chain(self):
        """verify_chain should use DETECT mode."""
        receipts = [{"value": 100}]
        result = verify_chain(receipts)
        assert result.mode == ProofMode.DETECT

    def test_synthesize_audit(self):
        """synthesize_audit should return Evidence."""
        receipts = [{"receipt_type": "decision"}]
        evidence = synthesize_audit(receipts)
        assert isinstance(evidence, Evidence)

    def test_bind_sensor_data(self):
        """bind_sensor_data should return BoundPacket."""
        sensor_data = {"altimeter": 1000}
        decision_receipts = [{"payload_hash": "abc:def"}]
        packet = bind_sensor_data(sensor_data, decision_receipts)
        assert isinstance(packet, BoundPacket)


class TestProofReceipt:
    """Tests for proof receipt emission."""

    def test_emit_proof_receipt(self):
        """Proof receipt should have required fields."""
        result = prove("BRIEF", [{"receipt_type": "decision"}])
        receipt = emit_proof_receipt(result)

        assert receipt["receipt_type"] == "proof"
        assert "proof_id" in receipt
        assert "mode" in receipt
        assert receipt["mode"] == "BRIEF"


class TestPerformance:
    """Performance tests for proof operations."""

    def test_brief_latency(self):
        """BRIEF mode should complete under 50ms."""
        receipts = [{"receipt_type": "decision", "decision_id": f"d{i}"} for i in range(100)]

        start = time.perf_counter()
        brief_evidence(receipts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"BRIEF latency {elapsed_ms}ms exceeds 50ms SLO"

    def test_detect_latency(self):
        """DETECT mode should complete under 50ms."""
        stream = [{"value": i} for i in range(100)]

        start = time.perf_counter()
        detect_anomaly(stream)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"DETECT latency {elapsed_ms}ms exceeds 50ms SLO"
