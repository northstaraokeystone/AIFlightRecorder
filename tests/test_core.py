"""Tests for core foundation functions."""

import json
import pytest
import time

from src.core import (
    dual_hash,
    emit_receipt,
    merkle_root,
    merkle_proof,
    verify_proof,
    StopRule,
    GENESIS_HASH
)


class TestDualHash:
    """Tests for dual_hash function."""

    def test_returns_dual_format(self):
        """Hash must be in SHA256:BLAKE3 format."""
        result = dual_hash(b"test")
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 64  # SHA256 hex length
        assert len(parts[1]) == 64  # BLAKE3 hex length

    def test_handles_bytes(self):
        """Should handle bytes input."""
        result = dual_hash(b"hello world")
        assert isinstance(result, str)
        assert ":" in result

    def test_handles_string(self):
        """Should handle string input by encoding to UTF-8."""
        result = dual_hash("hello world")
        assert isinstance(result, str)
        assert ":" in result

    def test_deterministic(self):
        """Same input should always produce same hash."""
        input_data = b"deterministic test"
        hash1 = dual_hash(input_data)
        hash2 = dual_hash(input_data)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        hash1 = dual_hash(b"input one")
        hash2 = dual_hash(b"input two")
        assert hash1 != hash2

    def test_empty_input(self):
        """Should handle empty input."""
        result = dual_hash(b"")
        assert ":" in result

    def test_latency_slo(self):
        """Hash computation should be under 10ms."""
        start = time.perf_counter()
        for _ in range(100):
            dual_hash(b"performance test data")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Hash latency {elapsed_ms}ms exceeds 10ms SLO"


class TestEmitReceipt:
    """Tests for emit_receipt function."""

    def test_contains_required_fields(self):
        """Receipt must have required CLAUDEME fields."""
        receipt = emit_receipt("test", {"data": "value"}, silent=True, to_file=False)

        assert "receipt_type" in receipt
        assert "ts" in receipt
        assert "tenant_id" in receipt
        assert "payload_hash" in receipt

    def test_receipt_type_set(self):
        """Receipt type should match input."""
        receipt = emit_receipt("custom_type", {}, silent=True, to_file=False)
        assert receipt["receipt_type"] == "custom_type"

    def test_payload_hash_is_dual(self):
        """Payload hash should be in dual format."""
        receipt = emit_receipt("test", {"data": "value"}, silent=True, to_file=False)
        assert ":" in receipt["payload_hash"]

    def test_timestamp_format(self):
        """Timestamp should be ISO8601 with Z suffix."""
        receipt = emit_receipt("test", {}, silent=True, to_file=False)
        assert receipt["ts"].endswith("Z")

    def test_data_included(self):
        """Input data should be included in receipt."""
        data = {"key1": "value1", "key2": 42}
        receipt = emit_receipt("test", data, silent=True, to_file=False)

        assert receipt["key1"] == "value1"
        assert receipt["key2"] == 42


class TestMerkleRoot:
    """Tests for merkle_root function."""

    def test_empty_list(self):
        """Empty list should return hash of 'empty_tree'."""
        result = merkle_root([])
        assert ":" in result  # Still dual format

    def test_single_item(self):
        """Single item should return hash of that item."""
        items = [{"data": "single"}]
        result = merkle_root(items)
        assert ":" in result

    def test_two_items(self):
        """Two items should combine correctly."""
        items = [{"a": 1}, {"b": 2}]
        result = merkle_root(items)
        assert ":" in result

    def test_deterministic(self):
        """Same items should always produce same root."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        root1 = merkle_root(items)
        root2 = merkle_root(items)
        assert root1 == root2

    def test_order_matters(self):
        """Different order should produce different root."""
        items1 = [{"a": 1}, {"b": 2}]
        items2 = [{"b": 2}, {"a": 1}]
        root1 = merkle_root(items1)
        root2 = merkle_root(items2)
        assert root1 != root2


class TestMerkleProof:
    """Tests for merkle_proof and verify_proof functions."""

    def test_proof_for_first_item(self):
        """Should generate valid proof for first item."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
        proof = merkle_proof(items, 0)
        root = merkle_root(items)

        assert verify_proof(items[0], proof, root)

    def test_proof_for_last_item(self):
        """Should generate valid proof for last item."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
        proof = merkle_proof(items, 3)
        root = merkle_root(items)

        assert verify_proof(items[3], proof, root)

    def test_proof_for_middle_item(self):
        """Should generate valid proof for middle item."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
        proof = merkle_proof(items, 1)
        root = merkle_root(items)

        assert verify_proof(items[1], proof, root)

    def test_invalid_index_raises(self):
        """Out of range index should raise IndexError."""
        items = [{"a": 1}, {"b": 2}]

        with pytest.raises(IndexError):
            merkle_proof(items, 5)

    def test_wrong_item_fails_verification(self):
        """Proof for wrong item should fail verification."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        proof = merkle_proof(items, 0)
        root = merkle_root(items)

        # Try to verify different item with proof for item 0
        assert not verify_proof(items[1], proof, root)

    def test_proof_size_logarithmic(self):
        """Proof size should be O(log N)."""
        items = [{"i": i} for i in range(1000)]
        proof = merkle_proof(items, 500)

        # log2(1000) ≈ 10
        assert len(proof) <= 12


class TestStopRule:
    """Tests for StopRule exception."""

    def test_stoprule_is_exception(self):
        """StopRule should be an Exception."""
        assert issubclass(StopRule, Exception)

    def test_stoprule_message(self):
        """StopRule should preserve message."""
        try:
            raise StopRule("test message")
        except StopRule as e:
            assert "test message" in str(e)

    def test_stoprule_attributes(self):
        """StopRule should have metric and action attributes."""
        sr = StopRule("error", metric="test_metric", action="halt")
        assert sr.metric == "test_metric"
        assert sr.action == "halt"
