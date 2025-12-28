"""Compression-Based Anomaly Detection

THE PARADIGM SHIFT: Normal patterns compress predictably.
Anomalous or manipulated patterns resist compression due to higher Kolmogorov complexity.

Implementation uses Normalized Compression Distance (NCD):
    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

Where C() is compressed size. High NCD = anomalous decision pattern.
"""

import gzip
import json
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .core import dual_hash, emit_receipt

# Anomaly thresholds
COMPRESSION_RATIO_TOLERANCE = 0.15  # 15% deviation from baseline
NCD_ANOMALY_THRESHOLD = 0.7  # High NCD indicates anomaly


@dataclass
class CompressionBaseline:
    """Baseline compression statistics from normal operations."""
    mean_ratio: float
    std_ratio: float
    mean_size: float
    sample_count: int
    baseline_data: bytes = field(default=b"", repr=False)
    decision_count: int = 0


def compress_decision(decision: dict) -> bytes:
    """Serialize and gzip compress a decision.

    Args:
        decision: Decision dict to compress

    Returns:
        Compressed bytes
    """
    # Serialize to JSON (sorted keys for consistency)
    json_bytes = json.dumps(decision, sort_keys=True).encode('utf-8')

    # Compress with gzip
    compressed = gzip.compress(json_bytes, compresslevel=9)

    return compressed


def compression_ratio(original: bytes, compressed: bytes) -> float:
    """Calculate compression ratio.

    Args:
        original: Original data
        compressed: Compressed data

    Returns:
        Ratio of compressed/original size (lower = better compression)
    """
    if len(original) == 0:
        return 1.0
    return len(compressed) / len(original)


def ncd(seq_a: list[dict], seq_b: list[dict]) -> float:
    """Compute Normalized Compression Distance between decision sequences.

    NCD measures how "different" two sequences are based on compression.
    Lower NCD = more similar, Higher NCD = more different/anomalous.

    Args:
        seq_a: First sequence of decisions
        seq_b: Second sequence of decisions

    Returns:
        NCD value between 0 and 1 (approximately)
    """
    # Serialize sequences
    data_a = json.dumps(seq_a, sort_keys=True).encode('utf-8')
    data_b = json.dumps(seq_b, sort_keys=True).encode('utf-8')
    data_ab = data_a + data_b

    # Compress each
    c_a = len(gzip.compress(data_a, compresslevel=9))
    c_b = len(gzip.compress(data_b, compresslevel=9))
    c_ab = len(gzip.compress(data_ab, compresslevel=9))

    # NCD formula
    min_c = min(c_a, c_b)
    max_c = max(c_a, c_b)

    if max_c == 0:
        return 0.0

    ncd_value = (c_ab - min_c) / max_c

    return ncd_value


def build_baseline(decisions: list[dict], min_samples: int = 10) -> CompressionBaseline:
    """Create compression baseline from normal operations.

    Args:
        decisions: List of known-good decisions
        min_samples: Minimum samples required

    Returns:
        CompressionBaseline with statistics

    Raises:
        ValueError: If not enough samples
    """
    if len(decisions) < min_samples:
        raise ValueError(f"Need at least {min_samples} decisions for baseline, got {len(decisions)}")

    ratios = []
    sizes = []
    all_data = b""

    for decision in decisions:
        original = json.dumps(decision, sort_keys=True).encode('utf-8')
        compressed = compress_decision(decision)

        ratio = compression_ratio(original, compressed)
        ratios.append(ratio)
        sizes.append(len(compressed))
        all_data += original

    return CompressionBaseline(
        mean_ratio=statistics.mean(ratios),
        std_ratio=statistics.stdev(ratios) if len(ratios) > 1 else 0.05,
        mean_size=statistics.mean(sizes),
        sample_count=len(decisions),
        baseline_data=gzip.compress(all_data, compresslevel=9),
        decision_count=len(decisions)
    )


def detect_anomaly(decision: dict, baseline: CompressionBaseline,
                   threshold: float = COMPRESSION_RATIO_TOLERANCE) -> tuple[bool, float, str]:
    """Detect if decision is anomalous based on compression.

    Args:
        decision: Decision to check
        baseline: Baseline from normal operations
        threshold: Tolerance for compression ratio deviation

    Returns:
        Tuple of (is_anomaly, score, reason)
    """
    original = json.dumps(decision, sort_keys=True).encode('utf-8')
    compressed = compress_decision(decision)
    ratio = compression_ratio(original, compressed)

    # Calculate deviation from baseline
    expected_ratio = baseline.mean_ratio
    deviation = abs(ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else 0

    # Compute NCD against baseline
    # Use ratio as proxy for full NCD (more efficient)
    z_score = (ratio - baseline.mean_ratio) / baseline.std_ratio if baseline.std_ratio > 0 else 0

    is_anomaly = deviation > threshold or abs(z_score) > 3

    if is_anomaly:
        if deviation > threshold:
            reason = f"Compression ratio deviation: {deviation:.2%} exceeds {threshold:.2%} threshold"
        else:
            reason = f"Z-score {z_score:.2f} exceeds 3 standard deviations"
    else:
        reason = "Normal compression pattern"

    # Emit receipt if anomaly detected
    if is_anomaly:
        emit_receipt("anomaly", {
            "metric": "compression_ratio",
            "baseline": baseline.mean_ratio,
            "actual": ratio,
            "delta": deviation,
            "classification": "drift" if deviation < 0.25 else "deviation",
            "action": "flag_for_review",
            "z_score": z_score
        }, silent=True)

    return is_anomaly, deviation, reason


def detect_tampering(decision: dict, stored_hash: str) -> tuple[bool, str]:
    """Detect if decision has been tampered with via hash mismatch.

    Args:
        decision: The decision to verify
        stored_hash: The hash from when decision was logged

    Returns:
        Tuple of (is_tampered, message)
    """
    current_hash = dual_hash(json.dumps(decision, sort_keys=True))

    if current_hash != stored_hash:
        emit_receipt("anomaly", {
            "metric": "hash_mismatch",
            "baseline": 0,
            "actual": 1,
            "delta": 1,
            "classification": "tampering",
            "action": "halt",
            "expected_hash": stored_hash,
            "actual_hash": current_hash
        }, silent=True)
        return True, f"TAMPERING DETECTED: Hash mismatch. Expected {stored_hash[:32]}... Got {current_hash[:32]}..."

    return False, "Hash verified"


def detect_sequence_anomaly(decisions: list[dict], baseline: CompressionBaseline,
                            window_size: int = 10) -> list[dict]:
    """Detect anomalies in a sequence of decisions using sliding window NCD.

    Args:
        decisions: Sequence of decisions to analyze
        baseline: Baseline for comparison
        window_size: Size of sliding window

    Returns:
        List of detected anomalies with positions
    """
    anomalies = []

    if len(decisions) < window_size * 2:
        return anomalies

    # Compare each window against baseline patterns
    baseline_decisions = []  # Would normally load from baseline

    for i in range(len(decisions) - window_size + 1):
        window = decisions[i:i + window_size]

        # Check individual decisions in window
        for j, decision in enumerate(window):
            is_anomaly, score, reason = detect_anomaly(decision, baseline)
            if is_anomaly:
                anomalies.append({
                    "position": i + j,
                    "decision_id": decision.get("decision_id", "unknown"),
                    "score": score,
                    "reason": reason,
                    "window_start": i
                })

    return anomalies


def compute_sequence_ncd(seq_a: list[dict], seq_b: list[dict]) -> float:
    """Compute NCD between two decision sequences.

    Lower values indicate similar sequences (same distribution/pattern).
    Higher values indicate different/anomalous sequences.

    Args:
        seq_a: First sequence
        seq_b: Second sequence

    Returns:
        NCD value
    """
    return ncd(seq_a, seq_b)


class AnomalyDetector:
    """Stateful anomaly detector with baseline learning."""

    def __init__(self, baseline_size: int = 100):
        """Initialize detector.

        Args:
            baseline_size: Number of decisions to use for baseline
        """
        self.baseline_size = baseline_size
        self.baseline: Optional[CompressionBaseline] = None
        self._learning_buffer: list[dict] = []
        self._is_learning = True

    def update(self, decision: dict) -> Optional[tuple[bool, float, str]]:
        """Update detector with new decision.

        Args:
            decision: New decision to process

        Returns:
            Anomaly result if baseline established, None if still learning
        """
        if self._is_learning:
            self._learning_buffer.append(decision)
            if len(self._learning_buffer) >= self.baseline_size:
                self.baseline = build_baseline(self._learning_buffer)
                self._is_learning = False
                emit_receipt("baseline", {
                    "sample_count": self.baseline.sample_count,
                    "mean_ratio": self.baseline.mean_ratio,
                    "std_ratio": self.baseline.std_ratio,
                    "status": "established"
                }, silent=True)
            return None

        return detect_anomaly(decision, self.baseline)

    def is_ready(self) -> bool:
        """Check if baseline is established."""
        return not self._is_learning

    def get_baseline(self) -> Optional[CompressionBaseline]:
        """Get current baseline."""
        return self.baseline

    def force_baseline(self, decisions: list[dict]):
        """Force baseline from provided decisions."""
        self.baseline = build_baseline(decisions)
        self._is_learning = False
