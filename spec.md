# AI Flight Recorder Specification v1.0

## Purpose
Decision provenance infrastructure for autonomous systems. Captures cryptographically
verifiable records of AI decisions at execution time - not post-hoc explanations, but proofs.

## Inputs

| Input | Type | Source | Validation |
|-------|------|--------|------------|
| Telemetry | dict | Drone sensors | GPS bounds, battery 0-100%, velocity limits |
| Perception | dict | Sensor fusion | Obstacle/target/threat schema validation |
| Mission | dict | Ground control | Mission ID, waypoints, constraints |
| Previous Hash | str | Local chain | SHA256:BLAKE3 format |

## Outputs

| Output | Type | Destination | Validation |
|--------|------|-------------|------------|
| decision_receipt | dict | receipts.jsonl | Schema compliance, hash present |
| log_receipt | dict | receipts.jsonl | Chain linkage valid |
| anchor_receipt | dict | receipts.jsonl | Merkle root verifiable |
| anomaly_receipt | dict | receipts.jsonl | Detection classification valid |
| sync_receipt | dict | receipts.jsonl | Custody chain intact |
| topology_receipt | dict | receipts.jsonl | Pattern classification valid |

## Receipt Types

### decision_receipt
```json
{
    "receipt_type": "decision",
    "ts": "ISO8601",
    "tenant_id": "edge-device-001",
    "decision_id": "uuid",
    "action_type": "CONTINUE|AVOID|ENGAGE|ABORT|HOVER|RTB",
    "confidence": "float 0-1",
    "model_version": "str",
    "payload_hash": "sha256:blake3"
}
```

### log_receipt
```json
{
    "receipt_type": "decision_log",
    "ts": "ISO8601",
    "tenant_id": "str",
    "decision_id": "uuid",
    "decision_hash": "sha256:blake3",
    "prev_hash": "sha256:blake3",
    "merkle_position": "int",
    "local_tree_root": "str",
    "payload_hash": "sha256:blake3"
}
```

### anchor_receipt
```json
{
    "receipt_type": "anchor",
    "ts": "ISO8601",
    "tenant_id": "str",
    "merkle_root": "hex",
    "hash_algos": ["SHA256", "BLAKE3"],
    "batch_size": "int",
    "tree_size": "int",
    "payload_hash": "sha256:blake3"
}
```

### anomaly_receipt
```json
{
    "receipt_type": "anomaly",
    "ts": "ISO8601",
    "tenant_id": "str",
    "metric": "compression_ratio|ncd|hash_mismatch",
    "baseline": "float",
    "actual": "float",
    "delta": "float",
    "classification": "drift|tampering|deviation",
    "action": "alert|halt|flag_for_review",
    "payload_hash": "sha256:blake3"
}
```

### sync_receipt
```json
{
    "receipt_type": "sync",
    "ts": "ISO8601",
    "tenant_id": "str",
    "edge_device_id": "str",
    "local_tree_size": "int",
    "local_root_hash": "str",
    "cloud_tree_size": "int",
    "cloud_root_hash": "str",
    "consistency_verified": "bool",
    "decisions_synced": "int",
    "payload_hash": "sha256:blake3"
}
```

### topology_receipt
```json
{
    "receipt_type": "topology",
    "ts": "ISO8601",
    "tenant_id": "str",
    "pattern_id": "uuid",
    "pattern_type": "navigation|threat|target|anomaly",
    "topology": "open|closed|hybrid",
    "effectiveness": "float",
    "escape_velocity": "float",
    "autonomy_score": "float",
    "transfer_score": "float",
    "can_graduate": "bool",
    "payload_hash": "sha256:blake3"
}
```

## SLOs (Service Level Objectives)

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

## Stoprules

1. **integrity_violation**: Chain hash mismatch - HALT, emit anomaly_receipt
2. **merkle_inconsistency**: Tree derivation fails - HALT, rehydrate
3. **compression_anomaly**: Ratio > 15% deviation - FLAG, continue
4. **memory_exceeded**: > 512MB on edge - HALT immediately
5. **decision_timeout**: > 100ms logging - emit_violation, continue

## Rollback Procedures

1. **Chain corruption**: Restore from last verified anchor point
2. **Sync failure**: Retry with exponential backoff (2s, 4s, 8s, 16s)
3. **Anomaly cascade**: Halt new decisions, verify last 100, resume from verified point

## Constraints

- All decisions deterministic given same random seed
- Merkle proofs O(log N) space complexity
- Edge device: 512MB RAM, 1 CPU core max
- Decision rate: 10Hz sustained, 50Hz burst
- Offline operation: unlimited duration with local logging

---
Hash: COMPUTE_ON_SAVE
Version: 1.0
Gate: t2h
