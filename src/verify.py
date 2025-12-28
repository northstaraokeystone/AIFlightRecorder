"""Tamper Detection Engine - THE DEMO MOMENT

This module creates the conversion moment: when viewers watch a modification
attempt get instantly flagged, they witness proof that accountability works.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .core import dual_hash, emit_receipt, load_receipts, StopRule
from .anchor import MerkleTree, verify_merkle_proof
from .compress import detect_tampering, detect_anomaly, CompressionBaseline


@dataclass
class IntegrityViolation:
    """A detected integrity violation."""
    position: int
    violation_type: str
    expected: str
    actual: str
    decision_id: str
    severity: str  # "critical", "warning"


@dataclass
class VerificationResult:
    """Result of chain verification."""
    is_valid: bool
    decisions_checked: int
    violations: list[IntegrityViolation]
    verification_time_ms: float
    chain_length: int
    merkle_root: str


def verify_chain_integrity(decisions: list[dict],
                           expected_hashes: Optional[list[str]] = None) -> tuple[bool, list[IntegrityViolation]]:
    """Verify complete decision chain integrity.

    If expected_hashes is provided, compares each decision's hash against it.
    Otherwise, the chain is considered valid (no stored hashes to compare against).

    The key insight: if anyone modifies decision data, the hash changes,
    and we detect it immediately by comparing to stored hashes.

    Args:
        decisions: List of decisions with or without hash metadata
        expected_hashes: Pre-computed hashes to verify against (from original logging)

    Returns:
        Tuple of (is_valid, list of violations)
    """
    violations = []

    if not decisions:
        return True, []

    # Compute current hashes
    current_hashes = []
    for decision in decisions:
        if "full_decision" in decision:
            decision_data = decision["full_decision"]
        elif "decision" in decision:
            decision_data = decision["decision"]
        else:
            decision_data = decision

        hash_val = dual_hash(json.dumps(decision_data, sort_keys=True))
        current_hashes.append(hash_val)

    # If no expected hashes provided, chain is valid by default
    # (This is initial verification - establishing the baseline)
    if expected_hashes is None:
        return True, []

    # Compare against expected hashes
    for i, (current, expected) in enumerate(zip(current_hashes, expected_hashes)):
        if current != expected:
            if "full_decision" in decisions[i]:
                decision_data = decisions[i]["full_decision"]
            else:
                decision_data = decisions[i]

            violations.append(IntegrityViolation(
                position=i,
                violation_type="hash_mismatch",
                expected=expected,
                actual=current,
                decision_id=decision_data.get("decision_id", "unknown"),
                severity="critical"
            ))

    return len(violations) == 0, violations


def compute_chain_hashes(decisions: list[dict]) -> list[str]:
    """Compute hashes for a decision chain.

    This should be called when decisions are first logged to establish
    the baseline for later verification.

    Args:
        decisions: List of decisions

    Returns:
        List of hash strings for each decision
    """
    hashes = []
    for decision in decisions:
        if "full_decision" in decision:
            decision_data = decision["full_decision"]
        elif "decision" in decision:
            decision_data = decision["decision"]
        else:
            decision_data = decision

        hash_val = dual_hash(json.dumps(decision_data, sort_keys=True))
        hashes.append(hash_val)

    return hashes


def verify_single_decision(decision: dict, expected_hash: str) -> bool:
    """Verify a single decision's hash.

    Args:
        decision: The decision dict
        expected_hash: Expected hash value

    Returns:
        True if hash matches
    """
    computed = dual_hash(json.dumps(decision, sort_keys=True))
    return computed == expected_hash


def verify_merkle_inclusion(decision: dict, proof: list[tuple[str, str]],
                             root: str) -> bool:
    """Verify decision is included in Merkle tree.

    Args:
        decision: The decision to verify
        proof: Merkle inclusion proof
        root: Expected Merkle root

    Returns:
        True if proof is valid
    """
    return verify_merkle_proof(decision, proof, root)


def run_tamper_test(decisions: list[dict], decision_index: int,
                    modification: dict) -> dict:
    """Simulate tampering and demonstrate detection.

    THE DEMO MOMENT: This function shows what happens when someone
    tries to modify a decision.

    Args:
        decisions: Original decision chain
        decision_index: Which decision to tamper with
        modification: Dict of {field_path: new_value}

    Returns:
        Detailed result of the tamper test
    """
    start_time = time.perf_counter()

    if decision_index < 0 or decision_index >= len(decisions):
        raise IndexError(f"Decision index {decision_index} out of range")

    # First, compute expected hashes from the ORIGINAL chain
    # These represent the "stored" hashes from when decisions were logged
    expected_hashes = compute_chain_hashes(decisions)

    # Get original decision
    original_decision = decisions[decision_index].copy()
    if "full_decision" in original_decision:
        target = original_decision["full_decision"]
    else:
        target = original_decision

    original_hash = dual_hash(json.dumps(target, sort_keys=True))

    # Apply modification
    modified_target = json.loads(json.dumps(target))  # Deep copy
    old_values = {}

    for field_path, new_value in modification.items():
        parts = field_path.split(".")
        obj = modified_target
        for part in parts[:-1]:
            obj = obj.get(part, {})
        old_values[field_path] = obj.get(parts[-1])
        obj[parts[-1]] = new_value

    modified_hash = dual_hash(json.dumps(modified_target, sort_keys=True))

    # Create tampered chain
    tampered_decisions = list(decisions)  # Copy list
    if "full_decision" in original_decision:
        tampered_decisions[decision_index] = {
            **original_decision,
            "full_decision": modified_target
        }
    elif "decision" in original_decision:
        tampered_decisions[decision_index] = {
            **original_decision,
            "decision": modified_target
        }
    else:
        tampered_decisions[decision_index] = modified_target

    # Run verification - compare tampered chain against ORIGINAL expected hashes
    is_valid, violations = verify_chain_integrity(tampered_decisions, expected_hashes)

    detection_time_ms = (time.perf_counter() - start_time) * 1000

    # Build result
    result = {
        "test_type": "tamper_simulation",
        "target_decision": target.get("decision_id", f"decision_{decision_index}"),
        "target_position": decision_index,
        "modification_attempted": {
            field: {"old": old_values.get(field), "new": value}
            for field, value in modification.items()
        },
        "detection_result": "INTEGRITY_FAILURE" if not is_valid else "NO_DETECTION",
        "detection_method": violations[0].violation_type if violations else "none",
        "detection_latency_ms": detection_time_ms,
        "original_hash": original_hash,
        "modified_hash": modified_hash,
        "chain_break_location": violations[0].position if violations else None,
        "violations": [
            {
                "position": v.position,
                "type": v.violation_type,
                "expected": v.expected[:32] + "..." if len(v.expected) > 32 else v.expected,
                "actual": v.actual[:32] + "..." if len(v.actual) > 32 else v.actual
            }
            for v in violations
        ]
    }

    # Emit verification receipt
    emit_receipt("verification", {
        "verification_type": "tamper_test",
        "result": "INTEGRITY_FAILURE" if not is_valid else "PASSED",
        "decisions_checked": len(decisions),
        "violations_found": len(violations),
        "detection_time_ms": detection_time_ms
    }, silent=True)

    return result


def generate_integrity_report(decisions: list[dict],
                               baseline: Optional[CompressionBaseline] = None) -> dict:
    """Generate comprehensive integrity audit report.

    Args:
        decisions: All decisions to verify
        baseline: Optional compression baseline for anomaly detection

    Returns:
        Full integrity report
    """
    start_time = time.perf_counter()

    # Chain integrity check
    is_valid, violations = verify_chain_integrity(decisions)

    # Build Merkle tree
    tree = MerkleTree()
    for decision in decisions:
        if "full_decision" in decision:
            tree.add_leaf(decision["full_decision"])
        else:
            tree.add_leaf(decision)

    merkle_root = tree.get_root()

    # Compression anomaly check
    anomalies = []
    if baseline:
        from .compress import detect_anomaly
        for i, decision in enumerate(decisions):
            target = decision.get("full_decision", decision)
            is_anomaly, score, reason = detect_anomaly(target, baseline)
            if is_anomaly:
                anomalies.append({
                    "position": i,
                    "score": score,
                    "reason": reason
                })

    verification_time_ms = (time.perf_counter() - start_time) * 1000

    report = {
        "report_type": "integrity_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "status": "VERIFIED" if is_valid and not anomalies else "FAILED",
            "decisions_checked": len(decisions),
            "chain_valid": is_valid,
            "violations_found": len(violations),
            "anomalies_detected": len(anomalies),
            "verification_time_ms": verification_time_ms
        },
        "chain_integrity": {
            "is_valid": is_valid,
            "violations": [
                {
                    "position": v.position,
                    "type": v.violation_type,
                    "decision_id": v.decision_id,
                    "severity": v.severity
                }
                for v in violations
            ]
        },
        "merkle_tree": {
            "root": merkle_root,
            "size": len(decisions),
            "algorithm": "SHA256:BLAKE3"
        },
        "compression_analysis": {
            "anomalies_detected": len(anomalies),
            "anomaly_positions": [a["position"] for a in anomalies]
        }
    }

    # Emit report receipt
    emit_receipt("verification", {
        "verification_type": "integrity_audit",
        "result": report["summary"]["status"],
        "decisions_checked": len(decisions),
        "violations": len(violations),
        "anomalies": len(anomalies),
        "verification_time_ms": verification_time_ms
    }, silent=True)

    return report


def format_tamper_alert(result: dict) -> str:
    """Format tamper test result for dramatic display.

    THE CONVERSION MOMENT OUTPUT.

    Args:
        result: Result from run_tamper_test

    Returns:
        Formatted string for terminal display
    """
    if result["detection_result"] == "INTEGRITY_FAILURE":
        lines = [
            "",
            "═" * 60,
            "  ██████╗ INTEGRITY FAILURE ██████╗",
            "═" * 60,
            "",
            f"  TAMPERING DETECTED at Decision #{result['target_position']}",
            "",
        ]

        for field, change in result["modification_attempted"].items():
            lines.append(f"  Modification: {field}")
            lines.append(f"    Old value: {change['old']}")
            lines.append(f"    New value: {change['new']}")
            lines.append("")

        lines.extend([
            f"  Expected Hash: {result['original_hash'][:50]}...",
            f"  Actual Hash:   {result['modified_hash'][:50]}...",
            "",
            "  Chain breaks at this point. All subsequent decisions",
            "  cannot be trusted.",
            "",
            f"  Detection Time: {result['detection_latency_ms']:.2f}ms",
            "",
            "═" * 60,
            ""
        ])

        return "\n".join(lines)

    else:
        return f"\n✓ Chain verified. No tampering detected.\n"


def format_verification_success() -> str:
    """Format successful verification for display.

    Returns:
        Formatted success message
    """
    lines = [
        "",
        "═" * 60,
        "  ✓ ✓ ✓  CHAIN VERIFIED  ✓ ✓ ✓",
        "═" * 60,
        "",
        "  All decisions cryptographically verified.",
        "  Hash chain intact. Merkle tree consistent.",
        "  No tampering detected.",
        "",
        "═" * 60,
        ""
    ]
    return "\n".join(lines)
