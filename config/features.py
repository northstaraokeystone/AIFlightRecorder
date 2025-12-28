"""AI Flight Recorder v2.0 Feature Flags

All flags start FALSE (shadow mode first).
Deployment sequence: shadow → yellow-only → full → recursive → self-aware
"""

import os
from typing import Optional

# =============================================================================
# FEATURE FLAGS - All start FALSE
# =============================================================================

# Core gating
FEATURE_GATE_ENABLED = False
FEATURE_GATE_YELLOW_ONLY = False  # Start with monitoring only

# Monte Carlo variance reduction
FEATURE_MONTE_CARLO_ENABLED = False

# Agent spawning (enable in order)
FEATURE_AGENT_SPAWNING_ENABLED = False
FEATURE_GREEN_LEARNERS_ENABLED = False     # 1st: lowest risk
FEATURE_YELLOW_WATCHERS_ENABLED = False    # 2nd: monitoring with spawned agents
FEATURE_RED_HELPERS_ENABLED = False        # 3rd: active problem solving

# Recursive gating (full depth)
FEATURE_RECURSIVE_GATING_ENABLED = False

# Self-improvement
FEATURE_TOPOLOGY_CLASSIFICATION_ENABLED = False
FEATURE_PATTERN_GRADUATION_ENABLED = False
FEATURE_ENTROPY_SELECTION_ENABLED = False

# Self-awareness (final stage)
FEATURE_HUNTER_ENABLED = False
FEATURE_SHEPHERD_ENABLED = False


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled.

    Supports environment variable override: FLIGHT_RECORDER_{FEATURE_NAME}=1

    Args:
        feature_name: Name of the feature flag

    Returns:
        True if enabled, False otherwise
    """
    # Check environment override first
    env_var = f"FLIGHT_RECORDER_{feature_name.upper()}"
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value.lower() in ("1", "true", "yes", "on")

    # Check module-level variable
    return globals().get(feature_name, False)


def enable_feature(feature_name: str):
    """Enable a feature flag at runtime.

    Args:
        feature_name: Name of the feature flag
    """
    if feature_name in globals():
        globals()[feature_name] = True


def disable_feature(feature_name: str):
    """Disable a feature flag at runtime.

    Args:
        feature_name: Name of the feature flag
    """
    if feature_name in globals():
        globals()[feature_name] = False


def get_deployment_stage() -> str:
    """Get current deployment stage based on enabled features.

    Returns:
        Stage name: shadow, yellow_only, monte_carlo, green_learners,
                   yellow_watchers, red_helpers, recursive, topology,
                   entropy, self_aware
    """
    if FEATURE_HUNTER_ENABLED and FEATURE_SHEPHERD_ENABLED:
        return "self_aware"
    if FEATURE_ENTROPY_SELECTION_ENABLED:
        return "entropy"
    if FEATURE_TOPOLOGY_CLASSIFICATION_ENABLED or FEATURE_PATTERN_GRADUATION_ENABLED:
        return "topology"
    if FEATURE_RECURSIVE_GATING_ENABLED:
        return "recursive"
    if FEATURE_RED_HELPERS_ENABLED:
        return "red_helpers"
    if FEATURE_YELLOW_WATCHERS_ENABLED:
        return "yellow_watchers"
    if FEATURE_GREEN_LEARNERS_ENABLED:
        return "green_learners"
    if FEATURE_MONTE_CARLO_ENABLED:
        return "monte_carlo"
    if FEATURE_GATE_ENABLED and FEATURE_GATE_YELLOW_ONLY:
        return "yellow_only"
    if FEATURE_GATE_ENABLED:
        return "gating"
    return "shadow"


def get_all_features() -> dict:
    """Get all feature flags and their current state.

    Returns:
        Dict of feature_name -> enabled
    """
    return {
        "FEATURE_GATE_ENABLED": FEATURE_GATE_ENABLED,
        "FEATURE_GATE_YELLOW_ONLY": FEATURE_GATE_YELLOW_ONLY,
        "FEATURE_MONTE_CARLO_ENABLED": FEATURE_MONTE_CARLO_ENABLED,
        "FEATURE_AGENT_SPAWNING_ENABLED": FEATURE_AGENT_SPAWNING_ENABLED,
        "FEATURE_GREEN_LEARNERS_ENABLED": FEATURE_GREEN_LEARNERS_ENABLED,
        "FEATURE_YELLOW_WATCHERS_ENABLED": FEATURE_YELLOW_WATCHERS_ENABLED,
        "FEATURE_RED_HELPERS_ENABLED": FEATURE_RED_HELPERS_ENABLED,
        "FEATURE_RECURSIVE_GATING_ENABLED": FEATURE_RECURSIVE_GATING_ENABLED,
        "FEATURE_TOPOLOGY_CLASSIFICATION_ENABLED": FEATURE_TOPOLOGY_CLASSIFICATION_ENABLED,
        "FEATURE_PATTERN_GRADUATION_ENABLED": FEATURE_PATTERN_GRADUATION_ENABLED,
        "FEATURE_ENTROPY_SELECTION_ENABLED": FEATURE_ENTROPY_SELECTION_ENABLED,
        "FEATURE_HUNTER_ENABLED": FEATURE_HUNTER_ENABLED,
        "FEATURE_SHEPHERD_ENABLED": FEATURE_SHEPHERD_ENABLED
    }


# =============================================================================
# DEPLOYMENT SEQUENCE DOCUMENTATION
# =============================================================================
#
# 1. All OFF — shadow mode, log what WOULD happen
# 2. GATE_ENABLED + GATE_YELLOW_ONLY — start with monitoring only
# 3. MONTE_CARLO_ENABLED — add variance reduction
# 4. GREEN_LEARNERS — lowest risk spawning
# 5. YELLOW_WATCHERS — monitoring with spawned agents
# 6. RED_HELPERS — active problem solving
# 7. RECURSIVE_GATING — full depth
# 8. TOPOLOGY + GRADUATION — self-improvement loop
# 9. ENTROPY_SELECTION — physics-based population
# 10. HUNTER + SHEPHERD — full self-awareness
#
# =============================================================================
