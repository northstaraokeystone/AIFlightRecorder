"""Monte Carlo Simulation Harness

Provides validation framework for AI Flight Recorder.
All 7 mandatory scenarios must pass before deployment.
"""

import gc
import json
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import dual_hash, emit_receipt, reset_receipt_counter
from src.drone import DroneState, run_cycle, run_mission
from src.logger import DecisionLogger
from src.anchor import MerkleTree
from src.compress import build_baseline, detect_anomaly, AnomalyDetector
from src.verify import verify_chain_integrity
from src.topology import analyze_pattern


@dataclass
class SimConfig:
    """Simulation configuration."""
    name: str
    n_cycles: int
    random_seed: int = 42
    stress_vectors: dict = field(default_factory=dict)
    success_criteria: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class SimState:
    """Simulation state at any point."""
    cycle: int = 0
    decisions: list = field(default_factory=list)
    decision_receipts: list = field(default_factory=list)
    merkle_tree: Optional[MerkleTree] = None
    receipts: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class SimResult:
    """Simulation result."""
    config: SimConfig
    state: SimState
    success: bool
    duration_ms: float
    metrics: dict


def execute_cycle(state: SimState, drone_state: DroneState,
                  seed: Optional[int] = None,
                  stress: Optional[dict] = None) -> tuple[SimState, DroneState]:
    """Execute one simulation cycle.

    Args:
        state: Current simulation state
        drone_state: Current drone state
        seed: Random seed
        stress: Stress parameters

    Returns:
        Tuple of (new_sim_state, new_drone_state)
    """
    cycle_start = time.perf_counter()

    # Apply stress if specified
    if stress:
        # Simulate CPU throttling by adding artificial delay
        if "cpu_throttle" in stress:
            time.sleep(stress["cpu_throttle"] * 0.001)

    # Run drone cycle
    new_drone_state, decision_receipt = run_cycle(drone_state, seed)

    # Add to Merkle tree
    if state.merkle_tree is None:
        state.merkle_tree = MerkleTree()

    state.merkle_tree.add_leaf(decision_receipt)

    # Update state
    state.cycle = drone_state.cycle_number + 1
    state.decision_receipts.append(decision_receipt)
    state.decisions.append(decision_receipt.get("full_decision", decision_receipt))

    # Track metrics
    cycle_time_ms = (time.perf_counter() - cycle_start) * 1000
    if "cycle_times" not in state.metrics:
        state.metrics["cycle_times"] = []
    state.metrics["cycle_times"].append(cycle_time_ms)

    return state, new_drone_state


def run_simulation(config: SimConfig) -> SimResult:
    """Execute a full simulation scenario.

    Args:
        config: Simulation configuration

    Returns:
        SimResult with outcomes
    """
    start_time = time.perf_counter()

    # Initialize
    state = SimState()
    state.merkle_tree = MerkleTree()
    drone_state = DroneState()
    reset_receipt_counter()

    # Start memory tracking
    tracemalloc.start()
    gc.collect()

    # Run cycles
    try:
        for i in range(config.n_cycles):
            cycle_seed = config.random_seed + i

            # Apply stress vectors
            stress = {}
            if "decision_rate_multiplier" in config.stress_vectors:
                # Run multiple sub-cycles
                mult = config.stress_vectors["decision_rate_multiplier"]
                for _ in range(int(mult)):
                    state, drone_state = execute_cycle(
                        state, drone_state, cycle_seed, stress
                    )
            else:
                state, drone_state = execute_cycle(
                    state, drone_state, cycle_seed, stress
                )

            # Memory check
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / 1024 / 1024

            if "max_memory_mb" in config.stress_vectors:
                if peak_mb > config.stress_vectors["max_memory_mb"]:
                    state.violations.append({
                        "type": "memory_exceeded",
                        "limit_mb": config.stress_vectors["max_memory_mb"],
                        "actual_mb": peak_mb,
                        "cycle": i
                    })

            state.metrics["peak_memory_mb"] = peak_mb

    except Exception as e:
        state.error = str(e)
        state.success = False

    tracemalloc.stop()

    # Verify success criteria
    success = validate_criteria(state, config.success_criteria)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Compute final metrics
    state.metrics["total_decisions"] = len(state.decisions)
    state.metrics["merkle_root"] = state.merkle_tree.get_root() if state.merkle_tree else None
    state.metrics["violations"] = len(state.violations)

    if state.metrics.get("cycle_times"):
        times = state.metrics["cycle_times"]
        state.metrics["avg_cycle_ms"] = sum(times) / len(times)
        state.metrics["p95_cycle_ms"] = sorted(times)[int(len(times) * 0.95)] if times else 0

    return SimResult(
        config=config,
        state=state,
        success=success and state.success,
        duration_ms=duration_ms,
        metrics=state.metrics
    )


def validate_criteria(state: SimState, criteria: dict) -> bool:
    """Validate success criteria.

    Args:
        state: Final simulation state
        criteria: Success criteria dict

    Returns:
        True if all criteria met
    """
    if not criteria:
        return True

    # Check completion rate
    if "completion_rate" in criteria:
        expected = criteria["completion_rate"]
        actual = len(state.decisions) / max(1, state.cycle)
        if actual < expected:
            state.violations.append({
                "type": "completion_rate",
                "expected": expected,
                "actual": actual
            })
            return False

    # Check violations
    if "max_violations" in criteria:
        if len(state.violations) > criteria["max_violations"]:
            return False

    # Check latency
    if "p95_latency_ms" in criteria:
        times = state.metrics.get("cycle_times", [])
        if times:
            p95 = sorted(times)[int(len(times) * 0.95)]
            if p95 > criteria["p95_latency_ms"]:
                state.violations.append({
                    "type": "latency_exceeded",
                    "expected": criteria["p95_latency_ms"],
                    "actual": p95
                })
                return False

    # Check memory
    if "max_memory_mb" in criteria:
        peak = state.metrics.get("peak_memory_mb", 0)
        if peak > criteria["max_memory_mb"]:
            state.violations.append({
                "type": "memory_exceeded",
                "expected": criteria["max_memory_mb"],
                "actual": peak
            })
            return False

    # Check chain integrity
    if criteria.get("chain_integrity", False):
        is_valid, violations = verify_chain_integrity(state.decision_receipts)
        if not is_valid:
            state.violations.extend([
                {"type": "chain_integrity", **v.__dict__}
                for v in violations
            ])
            return False

    return True


def run_all_scenarios(scenarios: list[SimConfig]) -> dict:
    """Run all scenarios and return summary.

    Args:
        scenarios: List of scenario configs

    Returns:
        Summary dict with all results
    """
    results = {}
    all_passed = True

    for scenario in scenarios:
        print(f"Running {scenario.name}...", end=" ", flush=True)
        result = run_simulation(scenario)
        results[scenario.name] = {
            "success": result.success,
            "duration_ms": result.duration_ms,
            "metrics": result.metrics,
            "violations": result.state.violations
        }
        if result.success:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            all_passed = False

    return {
        "all_passed": all_passed,
        "scenarios": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Convenience function for testing
def quick_test(n_cycles: int = 10) -> bool:
    """Run a quick test simulation.

    Args:
        n_cycles: Number of cycles

    Returns:
        True if passed
    """
    config = SimConfig(
        name="quick_test",
        n_cycles=n_cycles,
        success_criteria={
            "completion_rate": 0.99,
            "max_violations": 0
        }
    )
    result = run_simulation(config)
    return result.success
