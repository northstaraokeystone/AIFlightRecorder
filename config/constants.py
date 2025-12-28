"""AI Flight Recorder v2.2 Constants

Single source of truth for all thresholds and configuration.
No magic numbers in module code.
"""

# =============================================================================
# CONFIDENCE GATE THRESHOLDS
# =============================================================================

# Gate tier thresholds
GATE_GREEN_THRESHOLD = 0.9    # Confidence >= 0.9 → GREEN
GATE_YELLOW_THRESHOLD = 0.7   # Confidence >= 0.7 and < 0.9 → YELLOW
                               # Confidence < 0.7 → RED

# Wound detection (confidence drops)
WOUND_DROP_THRESHOLD = 0.15   # 15% drop triggers wound

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

MONTE_CARLO_SIMS = 100        # Number of simulations per decision
MONTE_CARLO_LATENCY_MS = 200  # Max latency for 100 sims
MONTE_CARLO_NOISE = 0.05      # Noise factor for perturbations
MONTE_CARLO_VARIANCE_THRESHOLD = 0.2  # Variance threshold for stability

# =============================================================================
# AGENT SPAWNING
# =============================================================================

# Population limits
MAX_AGENT_DEPTH = 3           # Maximum spawning depth
MAX_AGENT_POPULATION = 50     # Maximum total agents

# Default TTLs (seconds)
DEFAULT_TTL_SECONDS = 300     # 5 minutes default
GREEN_LEARNER_TTL = 60        # Success learners: 60s
YELLOW_WATCHER_TTL_EXTRA = 30 # Watchers: action_duration + 30s
RED_HELPER_TTL = 300          # Helpers: 5 minutes

# Spawn formula: (wound_count // 2) + 1, clamped
MIN_HELPER_SPAWN = 1
MAX_HELPER_SPAWN = 6

# Wound threshold for spawning
WOUND_SPAWN_THRESHOLD = 5     # 5 wounds triggers spawn

# =============================================================================
# ESCAPE VELOCITY THRESHOLDS (per domain)
# =============================================================================

ESCAPE_VELOCITY = {
    "navigation": 0.90,
    "threat_detection": 0.95,
    "target_acquisition": 0.88,
    "anomaly_response": 0.85,
    "drone_navigation": 0.90,  # Legacy name support
    "default": 0.85
}

# Topology classification thresholds
AUTONOMY_THRESHOLD = 0.75     # 75% for autonomous operation
TRANSFER_THRESHOLD = 0.70     # 70% for cross-domain transfer
EFFECTIVENESS_OPEN_THRESHOLD = 0.85  # Below this = closed topology

# =============================================================================
# ENTROPY CONSERVATION
# =============================================================================

ENTROPY_TOLERANCE = 0.01      # Conservation delta tolerance
ENTROPY_VIOLATION_ACTION = "halt"  # What to do on violation

# =============================================================================
# SLO THRESHOLDS (v2.0 additions)
# =============================================================================

SLO_GATE_DECISION_MS = 50     # Gate decision latency
SLO_MONTE_CARLO_MS = 200      # Monte Carlo completion
SLO_AGENT_SPAWN_MS = 50       # Agent spawn latency
SLO_AGENT_COORDINATION_MS = 100  # Agent coordination
SLO_GRADUATION_EVAL_MS = 200  # Graduation evaluation

# Existing SLOs (from v1.0)
SLO_HASH_LATENCY_MS = 10
SLO_MERKLE_PROOF_MS = 50
SLO_DECISION_LOG_P95_MS = 100
SLO_TAMPER_DETECTION_MS = 500
SLO_SYNC_VERIFICATION_MS = 2000
SLO_EDGE_MEMORY_MB = 512

# =============================================================================
# HITL (Human-In-The-Loop) GATES
# =============================================================================

HITL_AUTO_APPROVE_CONFIDENCE = 0.8  # Auto-approve if confidence > 0.8
HITL_AUTO_APPROVE_RISK = "low"      # Only auto-approve low risk
HITL_ESCALATION_DAYS = 7            # Days before escalation
HITL_AUTO_DECLINE_DAYS = 14         # Days before auto-decline

# =============================================================================
# COMPRESSION / ANOMALY DETECTION
# =============================================================================

COMPRESSION_RATIO_TOLERANCE = 0.15  # 15% deviation from baseline
NCD_ANOMALY_THRESHOLD = 0.7         # High NCD indicates anomaly
BASELINE_MIN_SAMPLES = 10           # Minimum samples for baseline

# =============================================================================
# AGENT TYPES
# =============================================================================

AGENT_TYPES = {
    "success_learner": {"color": "green", "ttl": GREEN_LEARNER_TTL},
    "drift_watcher": {"color": "yellow", "ttl": None},  # Uses YELLOW_WATCHER_TTL_EXTRA
    "wound_watcher": {"color": "yellow", "ttl": None},
    "success_watcher": {"color": "yellow", "ttl": None},
    "helper": {"color": "red", "ttl": RED_HELPER_TTL}
}

# Immortal agents (cannot be spawned/pruned)
IMMORTAL_AGENTS = ["HUNTER", "SHEPHERD"]

# =============================================================================
# TOPOLOGY STATES
# =============================================================================

TOPOLOGY_OPEN = "open"        # Continues to iterate/improve
TOPOLOGY_CLOSED = "closed"    # Stable, converged
TOPOLOGY_HYBRID = "hybrid"    # Can transfer across domains

# Agent lifecycle states
AGENT_STATE_SPAWNED = "spawned"
AGENT_STATE_ACTIVE = "active"
AGENT_STATE_GRADUATED = "graduated"
AGENT_STATE_PRUNED = "pruned"
AGENT_STATE_SUPERPOSITION = "superposition"  # Potential, not destroyed

# Pruning reasons
PRUNE_TTL_EXPIRED = "TTL_EXPIRED"
PRUNE_SIBLING_SOLVED = "SIBLING_SOLVED"
PRUNE_DEPTH_LIMIT = "DEPTH_LIMIT"
PRUNE_RESOURCE_CAP = "RESOURCE_CAP"
PRUNE_LOW_EFFECTIVENESS = "LOW_EFFECTIVENESS"

# =============================================================================
# GATE COLORS
# =============================================================================

GATE_GREEN = "GREEN"
GATE_YELLOW = "YELLOW"
GATE_RED = "RED"

# Gate triggers for spawning
TRIGGER_GREEN_GATE = "GREEN_GATE"
TRIGGER_YELLOW_GATE = "YELLOW_GATE"
TRIGGER_RED_GATE = "RED_GATE"
TRIGGER_WOUND_THRESHOLD = "WOUND_THRESHOLD"

# =============================================================================
# v2.1 GOVERNANCE THRESHOLDS
# =============================================================================

# RACI coverage requirements
RACI_COVERAGE_MIN = 1.0           # 100% of decisions must have RACI
PROVENANCE_COVERAGE_MIN = 1.0     # 100% of decisions must have provenance

# Provenance capture latency
SLO_PROVENANCE_CAPTURE_MS = 5     # Provenance capture < 5ms
SLO_RACI_LOOKUP_MS = 2            # RACI lookup < 2ms

# Training data production
SLO_TRAINING_EXTRACTION_MS = 100  # Training extraction < 100ms
TRAINING_MIN_QUALITY_SCORE = 0.6  # Minimum quality for training examples
TRAINING_BATCH_SIZE = 100         # Fine-tuning batch size

# Audit trail
SLO_AUDIT_TRAIL_10K_S = 5         # Audit trail generation for 10k < 5s

# =============================================================================
# v2.2 NEW SLO THRESHOLDS
# =============================================================================

# Proof module (any mode)
SLO_PROOF_MS = 150                # Proof operation < 150ms

# Temporal graph operations
SLO_TEMPORAL_GRAPH_ADD_MS = 20    # Add episode < 20ms
SLO_TEMPORAL_GRAPH_QUERY_MS = 50  # Query relevant < 50ms
TEMPORAL_DECAY_TAU = 0.1          # Default temporal decay constant

# CRAG fallback
SLO_CRAG_ASSESSMENT_MS = 100      # Knowledge sufficiency check < 100ms
SLO_CRAG_EXTERNAL_MS = 500        # External fallback < 500ms
CRAG_SUFFICIENCY_THRESHOLD = 0.7  # When to trigger external fallback

# MCP server
SLO_MCP_TOOL_MS = 200             # MCP tool invocation < 200ms

# Scenario pass rate
SCENARIO_PASS_RATE_MIN = 1.0      # 100% scenario pass required

# =============================================================================
# KNOWLEDGE SOURCES
# =============================================================================

# Default external sources for CRAG
CRAG_DEFAULT_SOURCES = ["ground_control", "reference_db"]

# =============================================================================
# ESCALATION TIMEOUTS (minutes)
# =============================================================================

ESCALATION_L1_TIMEOUT = 5         # Operator level
ESCALATION_L2_TIMEOUT = 15        # Supervisor level
ESCALATION_L3_TIMEOUT = 30        # Safety officer level
ESCALATION_L4_TIMEOUT = 60        # Executive level
