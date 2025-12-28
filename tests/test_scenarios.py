"""Tests for Monte Carlo validation scenarios."""

import pytest
import time

from sim.sim import SimConfig, run_simulation, quick_test
from sim.scenarios import (
    BASELINE, STRESS, TOPOLOGY, CASCADE,
    COMPRESSION, SINGULARITY, THERMODYNAMIC,
    QUICK_SCENARIOS, get_scenario_by_name, list_scenarios
)


class TestQuickTest:
    """Tests for quick validation."""

    def test_quick_test_passes(self):
        """Quick test with minimal cycles should pass."""
        assert quick_test(n_cycles=10)


class TestSimConfig:
    """Tests for simulation configuration."""

    def test_config_has_name(self):
        """Config should have name."""
        config = SimConfig(name="test", n_cycles=10)
        assert config.name == "test"

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = SimConfig(name="test", n_cycles=10)
        assert config.random_seed == 42
        assert config.stress_vectors == {}
        assert config.success_criteria == {}


class TestScenarioList:
    """Tests for scenario utilities."""

    def test_list_scenarios(self):
        """Should list all scenario names."""
        names = list_scenarios()
        assert "BASELINE" in names
        assert "STRESS" in names
        assert len(names) == 7

    def test_get_scenario_by_name(self):
        """Should retrieve scenario by name."""
        scenario = get_scenario_by_name("BASELINE")
        assert scenario.name == "BASELINE"

    def test_unknown_scenario_raises(self):
        """Unknown name should raise ValueError."""
        with pytest.raises(ValueError):
            get_scenario_by_name("NONEXISTENT")


class TestBaselineScenario:
    """Tests for BASELINE scenario."""

    def test_baseline_config(self):
        """BASELINE should have correct config."""
        assert BASELINE.n_cycles == 1000
        assert BASELINE.success_criteria.get("completion_rate") == 0.999

    def test_baseline_quick_run(self):
        """Quick BASELINE run should complete."""
        quick_baseline = SimConfig(
            name="QUICK_BASELINE",
            n_cycles=50,
            random_seed=42,
            success_criteria={
                "completion_rate": 0.99,
                "max_violations": 0
            }
        )
        result = run_simulation(quick_baseline)
        assert result.success


class TestStressScenario:
    """Tests for STRESS scenario."""

    def test_stress_has_constraints(self):
        """STRESS should have resource constraints."""
        assert "decision_rate_multiplier" in STRESS.stress_vectors


class TestQuickScenarios:
    """Tests for quick validation scenarios."""

    def test_quick_scenarios_exist(self):
        """Quick scenarios should be defined."""
        assert len(QUICK_SCENARIOS) >= 2

    def test_quick_scenarios_pass(self):
        """Quick scenarios should all pass."""
        from sim.sim import run_all_scenarios
        results = run_all_scenarios(QUICK_SCENARIOS)

        assert results["all_passed"], f"Quick scenarios failed: {results}"


class TestSimulationResult:
    """Tests for simulation result structure."""

    def test_result_has_metrics(self):
        """Result should include metrics."""
        config = SimConfig(name="test", n_cycles=20)
        result = run_simulation(config)

        assert "total_decisions" in result.metrics
        assert "merkle_root" in result.metrics

    def test_result_has_duration(self):
        """Result should report duration."""
        config = SimConfig(name="test", n_cycles=10)
        result = run_simulation(config)

        assert result.duration_ms > 0

    def test_result_has_state(self):
        """Result should include final state."""
        config = SimConfig(name="test", n_cycles=10)
        result = run_simulation(config)

        assert result.state is not None
        assert len(result.state.decisions) == 10


class TestValidationCriteria:
    """Tests for success criteria validation."""

    def test_completion_rate_check(self):
        """Should check completion rate."""
        config = SimConfig(
            name="test",
            n_cycles=10,
            success_criteria={"completion_rate": 0.99}
        )
        result = run_simulation(config)

        # Should pass - all cycles complete
        assert result.success

    def test_memory_check(self):
        """Should check memory constraint."""
        config = SimConfig(
            name="test",
            n_cycles=10,
            success_criteria={"max_memory_mb": 1000}  # Very generous
        )
        result = run_simulation(config)

        # Should pass - we won't use 1GB
        assert result.success
