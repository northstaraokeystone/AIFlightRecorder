"""AI Flight Recorder v2.2 Feature Flags

All flags start FALSE (shadow mode first).
Deployment sequence: shadow → gating → proof → memory → crag → agents → governance → mcp → training
"""

import os
from typing import Optional

# =============================================================================
# v2.0 FEATURE FLAGS - Agent/Gate Features
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

# =============================================================================
# v2.1 FEATURE FLAGS - Governance Features
# =============================================================================

# RACI accountability matrix
FEATURE_RACI_ENABLED = False

# Provenance tracking (model/policy versioning)
FEATURE_PROVENANCE_ENABLED = False

# Reason codes for interventions
FEATURE_REASON_CODES_ENABLED = False

# Training data production from corrections
FEATURE_TRAINING_PRODUCTION_ENABLED = False

# Enforce structured logging (no print statements)
FEATURE_STRUCTURED_LOGGING_ENFORCED = False

# Compliance report generation
FEATURE_COMPLIANCE_REPORTS_ENABLED = False

# Feedback loop for fine-tuning
FEATURE_FEEDBACK_LOOP_ENABLED = False

# =============================================================================
# v2.2 FEATURE FLAGS - Consolidation/New Features
# =============================================================================

# Consolidated proof module (BRIEF/PACKET/DETECT modes)
FEATURE_PROOF_CONSOLIDATED_ENABLED = False

# Temporal knowledge graph for decision memory
FEATURE_TEMPORAL_MEMORY_ENABLED = False

# CRAG (Corrective RAG) fallback pattern
FEATURE_CRAG_FALLBACK_ENABLED = False

# MCP server for external orchestrator integration
FEATURE_MCP_SERVER_ENABLED = False


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
        # v2.0 Agent/Gate
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
        "FEATURE_SHEPHERD_ENABLED": FEATURE_SHEPHERD_ENABLED,
        # v2.1 Governance
        "FEATURE_RACI_ENABLED": FEATURE_RACI_ENABLED,
        "FEATURE_PROVENANCE_ENABLED": FEATURE_PROVENANCE_ENABLED,
        "FEATURE_REASON_CODES_ENABLED": FEATURE_REASON_CODES_ENABLED,
        "FEATURE_TRAINING_PRODUCTION_ENABLED": FEATURE_TRAINING_PRODUCTION_ENABLED,
        "FEATURE_STRUCTURED_LOGGING_ENFORCED": FEATURE_STRUCTURED_LOGGING_ENFORCED,
        "FEATURE_COMPLIANCE_REPORTS_ENABLED": FEATURE_COMPLIANCE_REPORTS_ENABLED,
        "FEATURE_FEEDBACK_LOOP_ENABLED": FEATURE_FEEDBACK_LOOP_ENABLED,
        # v2.2 Consolidation/New
        "FEATURE_PROOF_CONSOLIDATED_ENABLED": FEATURE_PROOF_CONSOLIDATED_ENABLED,
        "FEATURE_TEMPORAL_MEMORY_ENABLED": FEATURE_TEMPORAL_MEMORY_ENABLED,
        "FEATURE_CRAG_FALLBACK_ENABLED": FEATURE_CRAG_FALLBACK_ENABLED,
        "FEATURE_MCP_SERVER_ENABLED": FEATURE_MCP_SERVER_ENABLED
    }


def get_deployment_stage_v2() -> str:
    """Get current deployment stage for v2.2.

    Returns:
        Stage name
    """
    # Check v2.2 features first (highest tier)
    if FEATURE_MCP_SERVER_ENABLED:
        return "mcp_integration"
    if FEATURE_CRAG_FALLBACK_ENABLED:
        return "crag_fallback"
    if FEATURE_TEMPORAL_MEMORY_ENABLED:
        return "temporal_memory"
    if FEATURE_PROOF_CONSOLIDATED_ENABLED:
        return "proof_consolidated"

    # Check v2.1 governance features
    if FEATURE_TRAINING_PRODUCTION_ENABLED:
        return "training_production"
    if FEATURE_COMPLIANCE_REPORTS_ENABLED:
        return "compliance"
    if FEATURE_RACI_ENABLED or FEATURE_PROVENANCE_ENABLED:
        return "governance"

    # Fall back to v2.0 stages
    return get_deployment_stage()


# =============================================================================
# DEPLOYMENT SEQUENCE DOCUMENTATION
# =============================================================================
#
# v2.0 Deployment Sequence:
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
# v2.1 Governance Additions (cumulative):
# 11. RACI_ENABLED — accountability matrix
# 12. PROVENANCE_ENABLED — model/policy versioning
# 13. REASON_CODES_ENABLED — intervention classification
# 14. COMPLIANCE_REPORTS_ENABLED — audit trail generation
# 15. TRAINING_PRODUCTION_ENABLED — correction → training data
# 16. FEEDBACK_LOOP_ENABLED — fine-tuning queue
#
# v2.2 Consolidation/New (cumulative):
# 17. PROOF_CONSOLIDATED_ENABLED — unified proof module
# 18. TEMPORAL_MEMORY_ENABLED — decision graph
# 19. CRAG_FALLBACK_ENABLED — knowledge augmentation
# 20. MCP_SERVER_ENABLED — external integration
#
# =============================================================================
