"""7 Mandatory Validation Scenarios

No deployment without ALL scenarios passing.
Per monte_carlo.docx specification.
"""

from .sim import SimConfig

# SCENARIO 1: BASELINE
# Standard operation, establish baselines
BASELINE = SimConfig(
    name="BASELINE",
    n_cycles=1000,
    random_seed=42,
    stress_vectors={},
    success_criteria={
        "completion_rate": 0.999,  # 99.9% cycle completion
        "max_violations": 0,
        "chain_integrity": True
    },
    description="Standard operation, establish baselines"
)

# SCENARIO 2: STRESS
# High decision rate, resource constraints
STRESS = SimConfig(
    name="STRESS",
    n_cycles=500,
    random_seed=123,
    stress_vectors={
        "decision_rate_multiplier": 5,  # 50Hz instead of 10Hz
        "max_memory_mb": 256,  # Memory pressure
        "cpu_throttle": 0.5  # CPU throttling
    },
    success_criteria={
        "completion_rate": 0.95,
        "p95_latency_ms": 100,
        "max_memory_mb": 512,
        "max_violations": 0
    },
    description="High decision rate, resource constraints"
)

# SCENARIO 3: TOPOLOGY
# Validate pattern classification accuracy
TOPOLOGY = SimConfig(
    name="TOPOLOGY",
    n_cycles=100,
    random_seed=456,
    stress_vectors={
        "pattern_test": True  # Generate synthetic patterns
    },
    success_criteria={
        "completion_rate": 0.99,
        "classification_accuracy": 0.98  # 98% per class
    },
    description="Validate pattern classification accuracy"
)

# SCENARIO 4: CASCADE
# High-volume decision logging
CASCADE = SimConfig(
    name="CASCADE",
    n_cycles=100,
    random_seed=789,
    stress_vectors={
        "pattern_variants": 5  # Each pattern generates 5 variants
    },
    success_criteria={
        "completion_rate": 0.99,
        "max_violations": 0,
        "chain_integrity": True
    },
    description="High-volume decision logging"
)

# SCENARIO 5: COMPRESSION
# Validate anomaly detection via compression
COMPRESSION = SimConfig(
    name="COMPRESSION",
    n_cycles=200,
    random_seed=101112,
    stress_vectors={
        "anomaly_injection": 10  # Inject 10 anomalous decisions
    },
    success_criteria={
        "completion_rate": 0.99,
        "anomaly_detection_rate": 1.0,  # All anomalies detected
        "false_positive_rate": 0.0  # No false positives
    },
    description="Validate anomaly detection via compression"
)

# SCENARIO 6: SINGULARITY
# Long-run stability test
SINGULARITY = SimConfig(
    name="SINGULARITY",
    n_cycles=10000,  # ~17 minutes simulated
    random_seed=131415,
    stress_vectors={},
    success_criteria={
        "completion_rate": 0.999,
        "max_memory_mb": 512,  # No unbounded growth
        "max_violations": 0
    },
    description="Long-run stability test"
)

# SCENARIO 7: THERMODYNAMIC
# Hash integrity conservation
THERMODYNAMIC = SimConfig(
    name="THERMODYNAMIC",
    n_cycles=1000,
    random_seed=161718,
    stress_vectors={
        "verify_every_cycle": True
    },
    success_criteria={
        "completion_rate": 1.0,
        "max_violations": 0,
        "chain_integrity": True,
        "merkle_consistency": True
    },
    description="Hash integrity conservation"
)

# All scenarios list
ALL_SCENARIOS = [
    BASELINE,
    STRESS,
    TOPOLOGY,
    CASCADE,
    COMPRESSION,
    SINGULARITY,
    THERMODYNAMIC
]

# Quick scenarios for fast testing
QUICK_SCENARIOS = [
    SimConfig(
        name="QUICK_BASELINE",
        n_cycles=50,
        random_seed=42,
        success_criteria={
            "completion_rate": 0.99,
            "max_violations": 0
        }
    ),
    SimConfig(
        name="QUICK_INTEGRITY",
        n_cycles=50,
        random_seed=43,
        success_criteria={
            "chain_integrity": True,
            "max_violations": 0
        }
    )
]


def get_scenario_by_name(name: str) -> SimConfig:
    """Get scenario by name.

    Args:
        name: Scenario name

    Returns:
        SimConfig for the scenario

    Raises:
        ValueError: If scenario not found
    """
    for scenario in ALL_SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Scenario '{name}' not found")


def list_scenarios() -> list[str]:
    """List all scenario names.

    Returns:
        List of scenario names
    """
    return [s.name for s in ALL_SCENARIOS]
