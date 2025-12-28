"""8 Mandatory Validation Scenarios - v2.0

No deployment without ALL scenarios passing.
Per monte_carlo.docx specification + v2.0 agent spawning.
"""

from .sim import SimConfig

# SCENARIO 1: BASELINE (v2.0: Add agent spawning verification)
# Standard operation, establish baselines
BASELINE = SimConfig(
    name="BASELINE",
    n_cycles=1000,
    random_seed=42,
    stress_vectors={},
    success_criteria={
        "completion_rate": 0.999,  # 99.9% cycle completion
        "max_violations": 0,
        "chain_integrity": True,
        # v2.0 additions
        "min_green_learners_spawned": 1,  # At least 1 GREEN learner spawned
        "max_red_helpers_spawned": 0  # Zero RED helpers (no anomalies expected)
    },
    description="Standard operation, establish baselines, verify agent spawning"
)

# SCENARIO 2: STRESS (v2.0: Add population cap enforcement)
# High decision rate, resource constraints
STRESS = SimConfig(
    name="STRESS",
    n_cycles=500,
    random_seed=123,
    stress_vectors={
        "decision_rate_multiplier": 10,  # 100Hz for 10x stress
        "max_memory_mb": 256,  # Memory pressure
        "cpu_throttle": 0.5  # CPU throttling
    },
    success_criteria={
        "completion_rate": 0.95,
        "p95_latency_ms": 100,
        "max_memory_mb": 512,
        "max_violations": 0,
        # v2.0 additions
        "max_agent_population": 50,  # Never exceed 50 agents
        "max_agent_depth": 3  # No depth > 3 spawning
    },
    description="High decision rate, population cap enforcement"
)

# SCENARIO 3: TOPOLOGY (v2.0: Add classification accuracy + graduation)
# Validate pattern classification accuracy
TOPOLOGY = SimConfig(
    name="TOPOLOGY",
    n_cycles=100,
    random_seed=456,
    stress_vectors={
        "pattern_test": True,  # Generate synthetic patterns
        "inject_open_patterns": 10,  # Known OPEN patterns
        "inject_closed_patterns": 10,  # Known CLOSED patterns
        "inject_hybrid_patterns": 5  # Known HYBRID patterns
    },
    success_criteria={
        "completion_rate": 0.99,
        "classification_accuracy": 0.98,  # 98% correct classification
        # v2.0 additions
        "open_patterns_graduated": True,  # OPEN patterns must graduate
        "closed_patterns_pruned": True  # CLOSED patterns must be pruned
    },
    description="Validate topology classification with graduation/pruning"
)

# SCENARIO 4: BIRTHING (renamed from CASCADE, v2.0: Agent birthing)
# Test agent spawning and sibling pruning
BIRTHING = SimConfig(
    name="BIRTHING",
    n_cycles=100,
    random_seed=789,
    stress_vectors={
        "inject_red_gate_decisions": 10,  # Inject 10 RED-gate decisions
        "force_wound_threshold": True  # Force wound threshold triggers
    },
    success_criteria={
        "completion_rate": 0.99,
        "max_violations": 0,
        "chain_integrity": True,
        # v2.0 additions
        "correct_helper_count": True,  # Helpers spawned per wound formula
        "sibling_pruning_on_solution": True,  # First solution triggers sibling pruning
        "winner_graduates": True  # Winning helper graduates to permanent pattern
    },
    description="Agent birthing, sibling pruning, winner graduation"
)

# Keep CASCADE as alias for backwards compatibility
CASCADE = BIRTHING

# SCENARIO 5: COMPRESSION (v2.0: Add entropy conservation)
# Validate anomaly detection via compression + entropy
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
        "false_positive_rate": 0.0,  # No false positives
        # v2.0 additions
        "entropy_conservation": True,  # entropy_conservation() returns True every cycle
        "entropy_trend_negative": True  # System entropy trends negative over run
    },
    description="Anomaly detection + entropy conservation"
)

# SCENARIO 6: SINGULARITY (v2.0: Add self-improvement verification)
# Long-run stability test with self-improvement
SINGULARITY = SimConfig(
    name="SINGULARITY",
    n_cycles=10000,  # ~17 minutes simulated
    random_seed=131415,
    stress_vectors={},
    success_criteria={
        "completion_rate": 0.999,
        "max_memory_mb": 512,  # No unbounded growth
        "max_violations": 0,
        # v2.0 additions
        "graduated_patterns_reduce_spawning": True,  # Patterns reduce future spawns
        "superposition_not_empty": True  # Some patterns held in reserve
    },
    description="Long-run stability + self-improvement verification"
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

# SCENARIO 8: SELF_REFERENCE (v2.0 NEW)
# Verify L4 receipts inform L0 processing
SELF_REFERENCE = SimConfig(
    name="SELF_REFERENCE",
    n_cycles=1000,
    random_seed=192021,
    stress_vectors={
        "emit_meta_receipts": True,  # Emit receipts about receipts
        "enable_self_reference": True  # System references own receipts
    },
    success_criteria={
        "completion_rate": 0.99,
        "max_violations": 0,
        # v2.0 criteria
        "receipt_self_reference": True,  # System references own receipts in decisions
        "receipt_completeness": 0.95  # Receipt completeness score >= 0.95
    },
    description="Verify L4 receipts inform L0 processing (self-awareness)"
)

# All scenarios list (8 mandatory scenarios for v2.0)
ALL_SCENARIOS = [
    BASELINE,
    STRESS,
    TOPOLOGY,
    BIRTHING,  # Renamed from CASCADE
    COMPRESSION,
    SINGULARITY,
    THERMODYNAMIC,
    SELF_REFERENCE  # NEW in v2.0
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


# v2.0 Quick scenarios with agent spawning
QUICK_SCENARIOS_V2 = [
    SimConfig(
        name="QUICK_BASELINE_V2",
        n_cycles=50,
        random_seed=42,
        success_criteria={
            "completion_rate": 0.99,
            "max_violations": 0,
            "min_green_learners_spawned": 1
        }
    ),
    SimConfig(
        name="QUICK_SPAWNING",
        n_cycles=50,
        random_seed=44,
        stress_vectors={
            "inject_red_gate_decisions": 2
        },
        success_criteria={
            "max_agent_population": 50,
            "max_agent_depth": 3
        }
    )
]
