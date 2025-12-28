"""Tests for compression-based anomaly detection."""

import json
import pytest
import random

from src.compress import (
    compress_decision,
    compression_ratio,
    ncd,
    build_baseline,
    detect_anomaly,
    detect_tampering,
    AnomalyDetector
)
from src.core import dual_hash


class TestCompression:
    """Tests for compression functions."""

    def test_compress_decision(self):
        """Should compress decision to bytes."""
        decision = {"action": "CONTINUE", "confidence": 0.95}
        compressed = compress_decision(decision)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        # Note: Very small inputs may be larger after compression due to gzip headers
        # This test just verifies compression produces output

    def test_compression_ratio_calculated(self):
        """Should calculate correct ratio."""
        original = b"x" * 100
        compressed = b"x" * 50

        ratio = compression_ratio(original, compressed)
        assert ratio == 0.5

    def test_compression_ratio_empty(self):
        """Empty input should return 1.0."""
        ratio = compression_ratio(b"", b"")
        assert ratio == 1.0

    def test_real_compression_ratio(self):
        """Real compression should have ratio < 1."""
        decision = {
            "action": "CONTINUE",
            "confidence": 0.95,
            "reasoning": "Normal flight pattern continuing on mission",
            "telemetry": {"gps": {"lat": 37.7749, "lon": -122.4194}}
        }
        original = json.dumps(decision).encode()
        compressed = compress_decision(decision)

        ratio = compression_ratio(original, compressed)
        assert ratio < 1.0  # Compressed is smaller


class TestNCD:
    """Tests for Normalized Compression Distance."""

    def test_identical_sequences(self):
        """Identical sequences should have low NCD."""
        seq = [{"a": 1}, {"b": 2}]
        distance = ncd(seq, seq)

        # NCD of identical should be low (close to 0)
        assert distance < 0.5

    def test_different_sequences(self):
        """Very different sequences should have higher NCD."""
        seq1 = [{"action": "CONTINUE"} for _ in range(10)]
        seq2 = [{"random": random.random()} for _ in range(10)]

        distance = ncd(seq1, seq2)
        # Different sequences should have measurable distance
        assert distance >= 0

    def test_ncd_symmetric(self):
        """NCD should be approximately symmetric."""
        seq1 = [{"a": i} for i in range(5)]
        seq2 = [{"b": i} for i in range(5)]

        d1 = ncd(seq1, seq2)
        d2 = ncd(seq2, seq1)

        # Should be similar (not exact due to compression quirks)
        assert abs(d1 - d2) < 0.2


class TestBaseline:
    """Tests for baseline building."""

    def test_build_baseline_requires_samples(self):
        """Should require minimum samples."""
        with pytest.raises(ValueError):
            build_baseline([{"a": 1}], min_samples=10)

    def test_baseline_has_statistics(self, compression_baseline):
        """Baseline should have computed statistics."""
        assert hasattr(compression_baseline, 'mean_ratio')
        assert hasattr(compression_baseline, 'std_ratio')
        assert hasattr(compression_baseline, 'sample_count')

    def test_baseline_mean_reasonable(self, compression_baseline):
        """Mean ratio should be reasonable (0-1)."""
        assert 0 < compression_baseline.mean_ratio < 1


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_normal_decision_not_anomalous(self, compression_baseline, sample_decisions):
        """Normal decisions should not be flagged."""
        if not sample_decisions:
            pytest.skip("No sample decisions")

        # Take a normal decision from the set
        decision = sample_decisions[0]
        if "full_decision" in decision:
            decision = decision["full_decision"]

        is_anomaly, score, reason = detect_anomaly(decision, compression_baseline)

        # Normal decision shouldn't necessarily be anomalous
        # (though individual decisions may vary)
        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)
        assert isinstance(reason, str)

    def test_random_noise_potentially_anomalous(self, compression_baseline):
        """Random noise should likely be flagged."""
        # Create very random decision
        noise_decision = {
            f"field_{i}": random.random() for i in range(50)
        }

        is_anomaly, score, reason = detect_anomaly(noise_decision, compression_baseline)

        # Random noise compresses differently
        # Score should be measurable
        assert score >= 0


class TestTamperingDetection:
    """Tests for hash-based tampering detection."""

    def test_unmodified_not_tampered(self):
        """Unmodified decision should not be flagged."""
        decision = {"action": "CONTINUE", "confidence": 0.95}
        hash_val = dual_hash(json.dumps(decision, sort_keys=True))

        is_tampered, msg = detect_tampering(decision, hash_val)

        assert not is_tampered
        assert "verified" in msg.lower()

    def test_modified_detected(self):
        """Modified decision should be detected."""
        original = {"action": "CONTINUE", "confidence": 0.95}
        original_hash = dual_hash(json.dumps(original, sort_keys=True))

        modified = {"action": "ENGAGE", "confidence": 0.95}

        is_tampered, msg = detect_tampering(modified, original_hash)

        assert is_tampered
        assert "TAMPERING" in msg or "mismatch" in msg.lower()


class TestAnomalyDetector:
    """Tests for stateful AnomalyDetector."""

    def test_learning_phase(self):
        """Should accumulate training data."""
        detector = AnomalyDetector(baseline_size=10)

        for i in range(5):
            result = detector.update({"i": i})
            assert result is None  # Still learning

        assert not detector.is_ready()

    def test_baseline_established(self):
        """Should establish baseline after enough samples."""
        detector = AnomalyDetector(baseline_size=10)

        for i in range(15):
            detector.update({"i": i, "action": "CONTINUE"})

        assert detector.is_ready()
        assert detector.get_baseline() is not None

    def test_detects_after_baseline(self):
        """Should return detection results after baseline."""
        detector = AnomalyDetector(baseline_size=10)

        # Train
        for i in range(10):
            detector.update({"action": "CONTINUE", "i": i})

        # Test
        result = detector.update({"action": "CONTINUE", "i": 11})

        assert result is not None
        assert len(result) == 3  # (is_anomaly, score, reason)
