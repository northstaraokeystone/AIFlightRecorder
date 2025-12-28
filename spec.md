# AI Flight Recorder Specification v2.2

## Purpose
Decision provenance infrastructure for autonomous systems. Captures cryptographically
verifiable records of AI decisions at execution time - not post-hoc explanations, but proofs.

**Version Evolution:**
- **v1.0**: Core receipts, merkle trees, compression-based anomaly detection
- **v2.0**: Agent birthing, confidence gating, Monte Carlo, entropy engine (ProofPack v3.0)
- **v2.1**: RACI accountability, provenance tracking, reason codes, training data production
- **v2.2**: Module consolidation (proof.py), MCP interface, temporal knowledge graph, CRAG fallback

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AI FLIGHT RECORDER v2.2                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   CORE      │    │   PROOF     │    │   MEMORY    │    │  KNOWLEDGE  │   │
│  │  (v1.0)     │    │   (v2.2)    │    │   (v2.2)    │    │   (v2.2)    │   │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤   │
│  │ dual_hash   │    │ BRIEF mode  │    │ temporal.py │    │  crag.py    │   │
│  │ emit_receipt│    │ PACKET mode │    │ episodes    │    │ sufficiency │   │
│  │ merkle_*    │    │ DETECT mode │    │ decay       │    │ fallback    │   │
│  │ StopRule    │    │ synthesize  │    │ lineage     │    │ fusion      │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ GOVERNANCE  │    │  TRAINING   │    │ COMPLIANCE  │    │    MCP      │   │
│  │   (v2.1)    │    │   (v2.1)    │    │   (v2.1)    │    │   (v2.2)    │   │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤   │
│  │ raci.py     │    │ extractor   │    │ audit_trail │    │ server.py   │   │
│  │ provenance  │    │ exporter    │    │ provenance_ │    │ tools.py    │   │
│  │ reason_codes│    │ feedback_   │    │   report    │    │ resources   │   │
│  │ escalation  │    │   loop      │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── core.py              # Foundation: dual_hash, emit_receipt, merkle_*, StopRule
├── proof.py             # v2.2: Unified BRIEF/PACKET/DETECT modes
├── compress.py          # Compression-based anomaly detection
├── verify.py            # Chain integrity verification
├── drone.py             # Simulated autonomous agent
├── anchor.py            # Merkle tree anchoring
├── logger.py            # Decision logging with receipts
├── spawner.py           # v2.0: Agent birthing (BIRTHRIGHT pattern)
├── gating.py            # v2.0: Confidence gates (RED/YELLOW/GREEN)
├── entropy.py           # v2.0: Entropy engine for anomaly detection
├── agents/              # v2.0: Agent types
│   ├── hunter.py        # HUNTER: Anomaly investigation
│   └── healer.py        # HEALER: Pattern remediation
├── memory/              # v2.2: Temporal knowledge
│   └── temporal.py      # Graphiti-inspired temporal graph
├── knowledge/           # v2.2: CRAG pattern
│   └── crag.py          # Corrective RAG fallback
├── mcp/                 # v2.2: Model Context Protocol
│   ├── server.py        # MCP server implementation
│   ├── tools.py         # 5 tools for external orchestrators
│   └── resources.py     # 8 resources for data access
├── governance/          # v2.1: Enterprise governance
│   ├── raci.py          # RACI accountability matrix
│   ├── provenance.py    # Model/policy provenance tracking
│   ├── reason_codes.py  # 14 intervention reason codes
│   └── escalation.py    # Decision escalation routing
├── training/            # v2.1: Training data production
│   ├── extractor.py     # Example extraction from interventions
│   ├── exporter.py      # JSONL/Parquet/HF export
│   └── feedback_loop.py # Fine-tuning job queue
└── compliance/          # v2.1: Regulatory compliance
    ├── audit_trail.py   # Audit trail generation
    └── provenance_report.py  # Provenance drift detection
```

---

## Inputs

| Input | Type | Source | Validation |
|-------|------|--------|------------|
| Telemetry | dict | Drone sensors | GPS bounds, battery 0-100%, velocity limits |
| Perception | dict | Sensor fusion | Obstacle/target/threat schema validation |
| Mission | dict | Ground control | Mission ID, waypoints, constraints |
| Previous Hash | str | Local chain | SHA256:BLAKE3 format |
| MCP Request | dict | External orchestrator | Tool/resource name, inputs, caller_id |
| Intervention | dict | Human operator | Reason code, correction, decision_id |

---

## Outputs

| Output | Type | Destination | Validation |
|--------|------|-------------|------------|
| decision_receipt | dict | receipts.jsonl | Schema compliance, hash present |
| log_receipt | dict | receipts.jsonl | Chain linkage valid |
| anchor_receipt | dict | receipts.jsonl | Merkle root verifiable |
| anomaly_receipt | dict | receipts.jsonl | Detection classification valid |
| proof_receipt | dict | receipts.jsonl | Mode valid, confidence 0-1 |
| memory_receipt | dict | receipts.jsonl | Operation valid, graph stats |
| crag_receipt | dict | receipts.jsonl | Sources used, fusion confidence |
| mcp_receipt | dict | receipts.jsonl | Request/response valid |
| raci_receipt | dict | receipts.jsonl | All roles assigned |
| provenance_receipt | dict | receipts.jsonl | All hashes present |
| intervention_receipt | dict | receipts.jsonl | Reason code valid |
| training_export_receipt | dict | receipts.jsonl | Format valid, count > 0 |
| audit_trail_receipt | dict | receipts.jsonl | Compliance status valid |

---

## Receipt Types (26 Total)

### Core Receipts (v1.0)

| Receipt Type | Purpose | Key Fields |
|--------------|---------|------------|
| `decision` | Decision capture | decision_id, action, confidence |
| `decision_log` | Chain logging | decision_hash, prev_hash, merkle_position |
| `anchor` | Merkle anchoring | merkle_root, batch_size, tree_size |

### Agent/Gate Receipts (v2.0)

| Receipt Type | Purpose | Key Fields |
|--------------|---------|------------|
| `spawn` | Agent creation | parent_agent_id, child_agents[], trigger |
| `wound` | Agent damage | wound_type, severity, agent_id |
| `remediation` | Damage repair | wound_id, strategy, outcome |
| `pattern_graduation` | Pattern maturation | pattern_id, effectiveness, transfer_score |
| `gate` | Confidence gating | gate_tier, required_confidence, outcome |
| `monte_carlo` | Simulation result | simulation_id, paths_explored, convergence |
| `entropy` | System entropy | entropy_level, anomaly_indicators |
| `anomaly_alert` | Anomaly detection | anomaly_id, severity, classification |

### Governance Receipts (v2.1)

| Receipt Type | Purpose | Key Fields |
|--------------|---------|------------|
| `raci` | Accountability | responsible, accountable, consulted, informed |
| `provenance` | Version tracking | model_version, model_hash, policy_version |
| `intervention` | Human override | reason_code, correction, operator_id |
| `training_example` | Training capture | original, corrected, quality_score |
| `training_export` | Batch export | format, count, output_path |
| `finetune` | Fine-tune job | job_id, examples_count, priority |
| `rollback` | Model rollback | from_version, to_version, reason |
| `escalation` | Decision escalation | current_level, escalation_reasons |
| `escalation_resolved` | Escalation closed | resolution, resolved_by |
| `audit_trail` | Audit generation | compliance_status, finding_count |
| `provenance_report` | Provenance report | drift_detected, drift_types |
| `system_event` | System events | event_type, severity |

### Consolidated Receipts (v2.2)

| Receipt Type | Purpose | Key Fields |
|--------------|---------|------------|
| `proof` | Unified proof | mode, input_receipts, output, confidence |
| `memory` | Temporal graph | operation, decision_ids_affected, graph_stats |
| `crag` | CRAG operation | internal_sufficiency, external_queried, fusion_confidence |
| `mcp` | MCP interaction | request_type, tool/resource, caller_id |

---

## MCP Interface (v2.2)

### Tools (5)

| Tool | Description | Inputs |
|------|-------------|--------|
| `verify_chain` | Verify decision chain integrity | start_time?, end_time? |
| `query_decisions` | Search decision history | action_type?, confidence_min?, limit? |
| `get_audit_trail` | Generate audit report | report_type, start_time?, end_time? |
| `inject_intervention` | Record human override | decision_id, reason_code, correction |
| `spawn_investigator` | Trigger agent spawn | anomaly_id, investigation_type |

### Resources (8)

| Resource URI | Description |
|--------------|-------------|
| `flight://decisions/stream` | Live decision feed |
| `flight://decisions/*` | Individual decision by ID |
| `flight://agents/active` | Currently active agents |
| `flight://agents/*` | Individual agent by ID |
| `flight://metrics/entropy` | System entropy metrics |
| `flight://metrics/slo` | SLO compliance metrics |
| `flight://patterns/graduated` | Graduated patterns |
| `flight://receipts/recent` | Recent receipts (last 100) |

---

## Proof Module Modes (v2.2)

### BRIEF Mode
Synthesizes multiple receipts into coherent evidence summary.
- Used by: `audit_trail.py`
- Output: `Evidence` with dialectical record (pro/con/gaps)

### PACKET Mode
Cryptographically binds external claims to receipt chain.
- Used by: `drone.py` for sensor data binding
- Output: `BoundPacket` with binding hash

### DETECT Mode
Entropy-based anomaly detection (HUNTER's core mechanism).
- Used by: `verify_chain`, entropy monitoring
- Output: `AnomalyResult` with classification

---

## Temporal Knowledge Graph (v2.2)

Graphiti-inspired memory for decision history:

- **Episodes**: Decision events with temporal context
- **Edges**: Causal relationships between decisions
- **Decay**: Older decisions weighted less (configurable tau)
- **Lineage**: Trace root cause or blast radius

```python
# Add decision to graph
episode_id = add_episode(decision, context, caused_by=["prev_decision_id"])

# Query relevant past decisions
relevant = query_relevant({"action_type": "AVOID"}, limit=10)

# Trace causality
lineage = get_decision_lineage("decision_id", direction="backward", max_depth=5)
```

---

## CRAG Fallback Pattern (v2.2)

Corrective RAG happens BEFORE spawning helpers:

1. **Assess sufficiency**: Score internal knowledge 0-1
2. **Fallback if needed**: Query external sources (ground control)
3. **Fuse knowledge**: Combine internal + external
4. **Decide**: If fused confidence >= 0.7, resolve without helpers

```python
result = perform_crag(
    query={"action_type": "AVOID"},
    decision_id="d1",
    internal_results=temporal_results,
    sufficiency_threshold=0.7
)
# result.resolved = True if no helpers needed
```

---

## Governance (v2.1)

### RACI Matrix
Every decision type has assigned accountability:
- **Responsible**: Who executes (usually `ai_system`)
- **Accountable**: Who approves (operator, safety_officer, etc.)
- **Consulted**: Who advises (path_planner, sensor_fusion, etc.)
- **Informed**: Who is notified (telemetry, audit_log, etc.)

### Reason Codes (14)
Standardized intervention classifications:

| Category | Codes |
|----------|-------|
| Safety | SAFETY_CRITICAL, IMMINENT_DANGER, SENSOR_MALFUNCTION |
| Operational | MISSION_CHANGE, WEATHER_OVERRIDE, AIRSPACE_RESTRICTION |
| Model | MODEL_ERROR, CONFIDENCE_OVERRIDE, CONTEXT_MISSING |
| Policy | POLICY_VIOLATION, REGULATORY_COMPLIANCE |
| Testing | TESTING, CALIBRATION |
| Other | OTHER (requires notes) |

### Escalation Levels (4)
1. **Operator**: Primary/backup operator (5min timeout)
2. **Supervisor**: Shift supervisor/ops manager (15min timeout)
3. **Safety Officer**: Safety officer/chief safety (30min timeout)
4. **Executive**: Operations director/CTO (60min timeout)

---

## SLOs (Service Level Objectives)

### Core SLOs (v1.0)

| SLO | Threshold | Action on Violation |
|-----|-----------|---------------------|
| Hash computation latency | < 10ms | emit_violation |
| Merkle proof generation | < 50ms | emit_violation |
| Decision logging latency | < 100ms p95 | emit_violation |
| Compression ratio stability | within 5% of baseline | flag_anomaly |
| Tamper detection latency | < 500ms | emit_violation |
| Sync verification | < 2s | emit_violation |
| Memory consumption (edge) | < 512MB | HALT |
| Scenario pass rate | 100% | block_deploy |

### Governance SLOs (v2.1)

| SLO | Threshold | Action on Violation |
|-----|-----------|---------------------|
| RACI assignment latency | < 5ms | emit_violation |
| Provenance capture latency | < 10ms | emit_violation |
| Training extraction latency | < 5ms | emit_violation |
| Audit trail (10k receipts) | < 5s | emit_violation |
| RACI coverage | >= 95% | review_required |
| Provenance coverage | >= 95% | review_required |

### v2.2 SLOs

| SLO | Threshold | Action on Violation |
|-----|-----------|---------------------|
| Proof operation (any mode) | < 50ms | emit_violation |
| Temporal graph add | < 10ms | emit_violation |
| Temporal graph query | < 20ms | emit_violation |
| CRAG assessment | < 10ms | emit_violation |
| CRAG external fallback | < 500ms | emit_violation |
| MCP tool invocation | < 50ms | emit_violation |

---

## Validation Scenarios (10)

| Scenario | Description | Key Assertions |
|----------|-------------|----------------|
| NORMAL_MISSION | Standard flight path | Decision count, compression baseline |
| OBSTACLE_AVOIDANCE | Reactive avoidance | AVOID action emitted, latency SLO |
| THREAT_DETECTION | Threat response | Confidence gating, anomaly detection |
| COMMUNICATION_LOSS | Offline operation | Autonomous decisions, sync on reconnect |
| BATTERY_CRITICAL | Emergency RTB | ABORT/RTB emitted, high confidence |
| PATTERN_LEARNING | Pattern graduation | Topology extraction, graduation receipt |
| DECISION_STREAM | High-volume | 1000 decisions, compression variance |
| TAMPERING_DETECTED | Integrity violation | Anomaly detection, HALT action |
| GOVERNANCE | RACI + provenance | All roles assigned, version tracked |
| CRAG_FALLBACK | Knowledge sufficiency | External queried when needed, fusion |

---

## Stoprules

1. **integrity_violation**: Chain hash mismatch - HALT, emit anomaly_receipt
2. **merkle_inconsistency**: Tree derivation fails - HALT, rehydrate
3. **compression_anomaly**: Ratio > 15% deviation - FLAG, continue
4. **memory_exceeded**: > 512MB on edge - HALT immediately
5. **decision_timeout**: > 100ms logging - emit_violation, continue
6. **raci_missing**: No accountability assigned - FLAG, escalate
7. **provenance_drift**: Model hash changed unexpectedly - FLAG, review

---

## Rollback Procedures

1. **Chain corruption**: Restore from last verified anchor point
2. **Sync failure**: Retry with exponential backoff (2s, 4s, 8s, 16s)
3. **Anomaly cascade**: Halt new decisions, verify last 100, resume from verified point
4. **Model rollback**: Emit rollback_receipt, restore previous model version

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/features.py` | Feature flags for progressive deployment |
| `config/constants.py` | SLO thresholds and system constants |
| `config/raci_matrix.json` | RACI assignments per decision type |
| `config/reason_codes.json` | Intervention reason code definitions |
| `ledger_schema.json` | Receipt type schemas and validation |

---

## Constraints

- All decisions deterministic given same random seed
- Merkle proofs O(log N) space complexity
- Edge device: 512MB RAM, 1 CPU core max
- Decision rate: 10Hz sustained, 50Hz burst
- Offline operation: unlimited duration with local logging
- All receipts must have `tenant_id` and `payload_hash`
- Dual hash format: `SHA256:BLAKE3`

---

Hash: COMPUTE_ON_SAVE
Version: 2.2
Gate: t24h
