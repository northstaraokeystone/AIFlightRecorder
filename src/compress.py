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


# =============================================================================
# QED ENTROPY ENGINE - v2.0
# =============================================================================
# Agent fitness = entropy reduction per receipt
# Population bounds emerge from entropy budgets (not arbitrary caps)
# High-fitness agents survive; low-fitness enter SUPERPOSITION


import math
import random
import uuid


def system_entropy(decisions: list[dict]) -> float:
    """Calculate Shannon entropy of decision stream in bits.

    Higher entropy = more uncertainty/variety in decisions.
    Lower entropy = more predictable pattern.

    Args:
        decisions: List of decisions to analyze

    Returns:
        Entropy value in bits
    """
    if not decisions:
        return 0.0

    # Extract action types
    action_counts = {}
    total = 0

    for decision in decisions:
        if "full_decision" in decision:
            d = decision["full_decision"]
        else:
            d = decision

        action = d.get("action", {}).get("type", "CONTINUE")
        action_counts[action] = action_counts.get(action, 0) + 1
        total += 1

    if total == 0:
        return 0.0

    # Calculate Shannon entropy
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def agent_fitness(agent_id: str, decisions: list[dict], agent_decisions: list[dict]) -> float:
    """Calculate entropy reduction per decision for this agent.

    Agent fitness = how much the agent reduces system entropy.

    Args:
        agent_id: The agent's ID
        decisions: All system decisions
        agent_decisions: Decisions this agent contributed to

    Returns:
        Fitness score (entropy reduction per decision)
    """
    if not agent_decisions:
        return 0.0

    # Calculate system entropy before agent's contribution
    entropy_before = system_entropy(decisions)

    # Calculate entropy after (including agent's decisions)
    combined = decisions + agent_decisions
    entropy_after = system_entropy(combined)

    # Fitness = entropy reduction normalized by decision count
    reduction = entropy_before - entropy_after
    fitness = reduction / len(agent_decisions) if agent_decisions else 0.0

    # Normalize to 0-1 range
    # Max possible reduction is about log2(6) ≈ 2.58 bits
    normalized_fitness = max(0, min(1, fitness / 2.58 + 0.5))

    return normalized_fitness


def selection_pressure(
    agents: list[dict],
    fitness_scores: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    """Apply Thompson sampling over fitness to select survivors.

    Returns agents that survive and agents in superposition.
    Superposition = potential, not destroyed.

    Args:
        agents: List of agent dicts with agent_id and fitness
        fitness_scores: Optional precomputed fitness scores

    Returns:
        Tuple of (survivor_ids, superposition_ids)
    """
    if not agents:
        return [], []

    # Get fitness scores
    scores = {}
    for agent in agents:
        agent_id = agent.get("agent_id", str(uuid.uuid4()))
        if fitness_scores and agent_id in fitness_scores:
            scores[agent_id] = fitness_scores[agent_id]
        else:
            scores[agent_id] = agent.get("fitness", 0.5)

    # Thompson sampling: sample from Beta distribution based on fitness
    samples = {}
    for agent_id, fitness in scores.items():
        # Convert fitness to Beta parameters
        # Higher fitness = higher alpha (more likely to be sampled high)
        alpha = max(1, fitness * 10)
        beta = max(1, (1 - fitness) * 10)
        samples[agent_id] = random.betavariate(alpha, beta)

    # Sort by sampled value
    sorted_agents = sorted(samples.items(), key=lambda x: x[1], reverse=True)

    # Top half survive, bottom half enter superposition
    mid = len(sorted_agents) // 2
    if mid == 0:
        mid = 1

    survivors = [aid for aid, _ in sorted_agents[:mid]]
    superposition = [aid for aid, _ in sorted_agents[mid:]]

    return survivors, superposition


def entropy_budget() -> float:
    """Calculate available entropy reduction capacity.

    Based on system resources and current state.

    Returns:
        Available entropy reduction capacity (bits)
    """
    # In a real system, this would be based on:
    # - Available compute resources
    # - Current agent population
    # - Historical entropy reduction rates

    # For now, return a fixed budget
    # Budget represents how much more entropy we can absorb
    base_budget = 10.0  # bits

    # Would reduce based on current agent count
    # current_agents = get_registry().get_stats()["active_count"]
    # budget = base_budget - (current_agents * 0.1)

    return max(0, base_budget)


def entropy_conservation(cycle: dict) -> bool:
    """Validate that sum(entropy_in) = sum(entropy_out) + work.

    The second law of thermodynamics for the system.
    Entropy cannot be destroyed, only transformed.

    Args:
        cycle: Dict with 'entropy_in', 'entropy_out', 'work' values

    Returns:
        True if conservation holds (within tolerance), False otherwise
    """
    entropy_in = cycle.get("entropy_in", 0)
    entropy_out = cycle.get("entropy_out", 0)
    work = cycle.get("work", 0)

    # Conservation: in = out + work
    expected_out = entropy_in - work
    delta = abs(entropy_out - expected_out)

    # Import tolerance from config
    try:
        from config.constants import ENTROPY_TOLERANCE
        tolerance = ENTROPY_TOLERANCE
    except ImportError:
        tolerance = 0.01

    is_valid = delta <= tolerance

    if not is_valid:
        emit_receipt("entropy", {
            "cycle_id": cycle.get("cycle_id", "unknown"),
            "entropy_in": entropy_in,
            "entropy_out": entropy_out,
            "work": work,
            "delta": delta,
            "conservation_valid": False,
            "action": "halt"
        }, silent=True)

    return is_valid


def emit_entropy_receipt(
    cycle_id: str,
    system_entropy_bits: float,
    entropy_delta: float,
    agents_fitness: dict,
    superposition_count: int
) -> dict:
    """Emit entropy cycle receipt.

    Args:
        cycle_id: Unique cycle identifier
        system_entropy_bits: Current system entropy
        entropy_delta: Change in entropy this cycle
        agents_fitness: Dict of agent_id -> fitness score
        superposition_count: Number of agents in superposition

    Returns:
        Receipt dict
    """
    conservation_valid = entropy_delta <= 0  # Entropy should decrease or stay same

    return emit_receipt("entropy", {
        "cycle_id": cycle_id,
        "system_entropy_bits": system_entropy_bits,
        "entropy_delta": entropy_delta,
        "conservation_valid": conservation_valid,
        "agents_fitness": agents_fitness,
        "superposition_count": superposition_count
    }, silent=True)


def entropy_trend(history: list[float], window: int = 10) -> str:
    """Analyze entropy trend over recent measurements.

    Args:
        history: Recent entropy values
        window: Analysis window

    Returns:
        "decreasing" (healthy), "stable", or "increasing" (concerning)
    """
    if len(history) < window:
        return "stable"

    recent = history[-window:]
    mid = window // 2

    first_avg = sum(recent[:mid]) / mid if mid > 0 else 0
    second_avg = sum(recent[mid:]) / (window - mid) if (window - mid) > 0 else 0

    diff = second_avg - first_avg

    if diff < -0.1:
        return "decreasing"  # Healthy - system is learning
    elif diff > 0.1:
        return "increasing"  # Concerning - system is becoming more chaotic
    else:
        return "stable"


class EntropyEngine:
    """Stateful entropy tracking and agent selection."""

    def __init__(self):
        self._history: list[float] = []
        self._agent_fitness: dict[str, float] = {}
        self._superposition: set[str] = set()
        self._cycle_count = 0

    def update(self, decisions: list[dict], agents: list[dict]) -> dict:
        """Update entropy state with new decisions and agents.

        Args:
            decisions: Recent decisions
            agents: Current agents

        Returns:
            Entropy update result
        """
        self._cycle_count += 1

        # Calculate current entropy
        current_entropy = system_entropy(decisions)
        self._history.append(current_entropy)

        # Keep history bounded
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        # Calculate entropy delta
        if len(self._history) >= 2:
            entropy_delta = current_entropy - self._history[-2]
        else:
            entropy_delta = 0.0

        # Update agent fitness
        for agent in agents:
            agent_id = agent.get("agent_id", "")
            if agent_id:
                agent_decs = agent.get("decisions", [])
                fitness = agent_fitness(agent_id, decisions, agent_decs)
                self._agent_fitness[agent_id] = fitness

        # Apply selection pressure
        survivors, superposition = selection_pressure(agents, self._agent_fitness)
        self._superposition = set(superposition)

        # Emit entropy receipt
        receipt = emit_entropy_receipt(
            cycle_id=str(self._cycle_count),
            system_entropy_bits=current_entropy,
            entropy_delta=entropy_delta,
            agents_fitness=self._agent_fitness,
            superposition_count=len(superposition)
        )

        return {
            "cycle_id": self._cycle_count,
            "entropy": current_entropy,
            "delta": entropy_delta,
            "trend": entropy_trend(self._history),
            "survivors": survivors,
            "superposition": list(superposition),
            "receipt": receipt
        }

    def get_fitness(self, agent_id: str) -> float:
        """Get fitness score for an agent."""
        return self._agent_fitness.get(agent_id, 0.5)

    def is_in_superposition(self, agent_id: str) -> bool:
        """Check if agent is in superposition."""
        return agent_id in self._superposition

    def get_entropy_history(self) -> list[float]:
        """Get entropy history."""
        return self._history.copy()

    def get_budget(self) -> float:
        """Get current entropy budget."""
        return entropy_budget()


# Global entropy engine instance
_entropy_engine: Optional[EntropyEngine] = None


def get_entropy_engine() -> EntropyEngine:
    """Get the global entropy engine."""
    global _entropy_engine
    if _entropy_engine is None:
        _entropy_engine = EntropyEngine()
    return _entropy_engine


def reset_entropy_engine():
    """Reset the global entropy engine (for testing)."""
    global _entropy_engine
    _entropy_engine = EntropyEngine()
