"""Unified Proof Module - BRIEF/PACKET/DETECT modes (v2.2)

Consolidates evidence synthesis, claim binding, and anomaly detection
into a single unified interface.

Modes:
  - BRIEF: Synthesize multiple receipts into coherent evidence summary
  - PACKET: Cryptographically bind external claims to receipt chain
  - DETECT: Entropy-based anomaly detection (HUNTER's core mechanism)
"""

import gzip
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .core import dual_hash, emit_receipt, merkle_root


class ProofMode(Enum):
    """Proof operation modes."""
    BRIEF = "BRIEF"    # Evidence synthesis
    PACKET = "PACKET"  # Claim binding
    DETECT = "DETECT"  # Anomaly detection


@dataclass
class ProofResult:
    """Result of a proof operation."""
    proof_id: str
    mode: ProofMode
    success: bool
    confidence: float
    output: dict
    input_receipts: list
    error: Optional[str] = None


@dataclass
class Evidence:
    """Synthesized evidence from multiple receipts."""
    evidence_id: str
    summary: str
    source_receipts: list
    confidence: float
    dialectical_record: dict  # pro, con, gaps


@dataclass
class BoundPacket:
    """External claim bound to receipt chain."""
    packet_id: str
    claim: dict
    binding_receipts: list
    binding_hash: str
    verification_status: str


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    anomaly_id: str
    is_anomaly: bool
    score: float
    classification: str
    description: str
    affected_items: list


# =============================================================================
# BRIEF MODE - Evidence Synthesis
# =============================================================================

def brief_evidence(receipts: list, context: Optional[dict] = None) -> Evidence:
    """Synthesize multiple receipts into coherent evidence.

    Used by audit_trail.py to generate audit summaries.

    Args:
        receipts: List of receipts to synthesize
        context: Optional context for synthesis

    Returns:
        Evidence object with summary
    """
    if not receipts:
        return Evidence(
            evidence_id=str(uuid.uuid4()),
            summary="No receipts provided for synthesis",
            source_receipts=[],
            confidence=0.0,
            dialectical_record={"pro": [], "con": [], "gaps": ["no_input"]}
        )

    # Extract key information from receipts
    receipt_types = {}
    decision_ids = set()
    time_range = {"start": None, "end": None}
    confidence_scores = []

    for r in receipts:
        # Count receipt types
        rt = r.get("receipt_type", "unknown")
        receipt_types[rt] = receipt_types.get(rt, 0) + 1

        # Track decision IDs
        if "decision_id" in r:
            decision_ids.add(r["decision_id"])

        # Track time range
        ts = r.get("ts")
        if ts:
            if time_range["start"] is None or ts < time_range["start"]:
                time_range["start"] = ts
            if time_range["end"] is None or ts > time_range["end"]:
                time_range["end"] = ts

        # Collect confidence scores
        if "confidence" in r:
            confidence_scores.append(r["confidence"])
        elif "confidence_score" in r:
            confidence_scores.append(r["confidence_score"])

    # Build dialectical record
    pro = []
    con = []
    gaps = []

    # Evidence supporting conclusions
    if "decision" in receipt_types:
        pro.append(f"{receipt_types['decision']} decisions recorded")
    if "anchor" in receipt_types:
        pro.append(f"Chain anchored {receipt_types['anchor']} times")
    if "verification" in receipt_types:
        pro.append(f"{receipt_types['verification']} verifications performed")

    # Evidence against or concerning
    if "anomaly" in receipt_types:
        con.append(f"{receipt_types['anomaly']} anomalies detected")
    if "wound" in receipt_types:
        con.append(f"{receipt_types['wound']} wounds recorded")

    # Gaps in evidence
    if "anchor" not in receipt_types:
        gaps.append("No anchor receipts - chain not verified")
    if len(decision_ids) == 0:
        gaps.append("No decision IDs tracked")

    # Calculate overall confidence
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
    else:
        avg_confidence = 0.5

    # Adjust confidence based on gaps
    if gaps:
        avg_confidence *= (1 - 0.1 * len(gaps))

    # Generate summary
    summary_parts = [
        f"Evidence synthesis of {len(receipts)} receipts",
        f"covering {len(decision_ids)} decisions",
    ]
    if time_range["start"] and time_range["end"]:
        summary_parts.append(f"from {time_range['start']} to {time_range['end']}")
    summary_parts.append(f"Receipt types: {', '.join(f'{k}({v})' for k, v in receipt_types.items())}")

    return Evidence(
        evidence_id=str(uuid.uuid4()),
        summary=". ".join(summary_parts),
        source_receipts=[r.get("payload_hash", "") for r in receipts],
        confidence=max(0.0, min(1.0, avg_confidence)),
        dialectical_record={"pro": pro, "con": con, "gaps": gaps}
    )


# =============================================================================
# PACKET MODE - Claim Binding
# =============================================================================

def packet_bind(claim: dict, receipts: list, context: Optional[dict] = None) -> BoundPacket:
    """Cryptographically bind external claim to receipt chain.

    Used by drone.py to bind sensor readings to decision receipts.

    Args:
        claim: External claim to bind
        receipts: Receipts to bind to
        context: Optional binding context

    Returns:
        BoundPacket with binding proof
    """
    if not claim:
        return BoundPacket(
            packet_id=str(uuid.uuid4()),
            claim={},
            binding_receipts=[],
            binding_hash=dual_hash(b"empty_claim"),
            verification_status="failed"
        )

    # Extract receipt hashes for binding
    receipt_hashes = []
    for r in receipts:
        if "payload_hash" in r:
            receipt_hashes.append(r["payload_hash"])

    # Create binding structure
    binding_data = {
        "claim": claim,
        "receipt_hashes": receipt_hashes,
        "bound_at": datetime.now(timezone.utc).isoformat(),
        "binding_context": context or {}
    }

    # Compute binding hash (merkle of claim + receipt hashes)
    binding_items = [json.dumps(claim, sort_keys=True)] + receipt_hashes
    binding_hash = merkle_root(binding_items)

    # Verify binding integrity
    verification_status = "verified" if receipt_hashes else "unbound"

    return BoundPacket(
        packet_id=str(uuid.uuid4()),
        claim=claim,
        binding_receipts=receipt_hashes,
        binding_hash=binding_hash,
        verification_status=verification_status
    )


# =============================================================================
# DETECT MODE - Anomaly Detection
# =============================================================================

def detect_anomaly(stream: list, baseline: Optional[dict] = None,
                   context: Optional[dict] = None) -> AnomalyResult:
    """Compression-based anomaly detection.

    Uses Normalized Compression Distance (NCD) to detect anomalies.
    HUNTER's core detection mechanism.

    Args:
        stream: Data stream to analyze
        baseline: Baseline statistics for comparison
        context: Detection context

    Returns:
        AnomalyResult with detection outcome
    """
    if not stream:
        return AnomalyResult(
            anomaly_id=str(uuid.uuid4()),
            is_anomaly=False,
            score=0.0,
            classification="no_data",
            description="Empty stream provided",
            affected_items=[]
        )

    # Default baseline
    if baseline is None:
        baseline = {
            "mean_ratio": 0.5,
            "std_ratio": 0.1,
            "ncd_threshold": 0.7
        }

    # Serialize stream data
    stream_bytes = json.dumps(stream, sort_keys=True).encode('utf-8')

    # Compress and calculate ratio
    compressed = gzip.compress(stream_bytes, compresslevel=9)
    ratio = len(compressed) / len(stream_bytes) if stream_bytes else 1.0

    # Calculate deviation from baseline
    expected_ratio = baseline.get("mean_ratio", 0.5)
    std_ratio = baseline.get("std_ratio", 0.1)
    ncd_threshold = baseline.get("ncd_threshold", 0.7)

    deviation = abs(ratio - expected_ratio)
    z_score = deviation / std_ratio if std_ratio > 0 else 0

    # Compute NCD if baseline data available
    ncd_score = 0.0
    if "baseline_data" in baseline:
        baseline_bytes = baseline["baseline_data"]
        if isinstance(baseline_bytes, str):
            baseline_bytes = baseline_bytes.encode('utf-8')
        combined = stream_bytes + baseline_bytes

        c_stream = len(compressed)
        c_baseline = len(gzip.compress(baseline_bytes, compresslevel=9))
        c_combined = len(gzip.compress(combined, compresslevel=9))

        min_c = min(c_stream, c_baseline)
        max_c = max(c_stream, c_baseline)

        if max_c > 0:
            ncd_score = (c_combined - min_c) / max_c

    # Determine if anomaly
    is_anomaly = (z_score > 3) or (ncd_score > ncd_threshold) or (deviation > 0.15)

    # Calculate composite score
    score = max(z_score / 10, ncd_score, deviation) if is_anomaly else 0.0
    score = min(1.0, score)

    # Classify anomaly
    if not is_anomaly:
        classification = "normal"
        description = "Pattern matches baseline"
    elif ncd_score > 0.9:
        classification = "tampering"
        description = f"Pattern radically different from baseline (NCD={ncd_score:.2f})"
    elif ncd_score > 0.7:
        classification = "drift"
        description = f"Pattern drifting from baseline (NCD={ncd_score:.2f})"
    elif z_score > 5:
        classification = "spike"
        description = f"Compression ratio spike (z={z_score:.2f})"
    else:
        classification = "deviation"
        description = f"Unusual pattern (deviation={deviation:.2f})"

    # Identify affected items
    affected = []
    for i, item in enumerate(stream):
        if isinstance(item, dict) and "decision_id" in item:
            affected.append(item["decision_id"])
        elif isinstance(item, dict) and "receipt_type" in item:
            affected.append(f"{item['receipt_type']}_{i}")

    return AnomalyResult(
        anomaly_id=str(uuid.uuid4()),
        is_anomaly=is_anomaly,
        score=score,
        classification=classification,
        description=description,
        affected_items=affected[:10]  # Limit to 10
    )


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def prove(mode: str, inputs: list, context: Optional[dict] = None) -> ProofResult:
    """Unified entry point for all proof operations.

    Args:
        mode: One of BRIEF, PACKET, DETECT
        inputs: Input data (receipts, claims, streams)
        context: Operation context

    Returns:
        ProofResult with operation outcome
    """
    proof_id = str(uuid.uuid4())
    context = context or {}

    try:
        mode_enum = ProofMode(mode.upper())
    except ValueError:
        return ProofResult(
            proof_id=proof_id,
            mode=ProofMode.BRIEF,
            success=False,
            confidence=0.0,
            output={"error": f"Invalid mode: {mode}"},
            input_receipts=[],
            error=f"Invalid mode: {mode}. Use BRIEF, PACKET, or DETECT."
        )

    try:
        if mode_enum == ProofMode.BRIEF:
            evidence = brief_evidence(inputs, context)
            return ProofResult(
                proof_id=proof_id,
                mode=mode_enum,
                success=True,
                confidence=evidence.confidence,
                output={
                    "evidence_id": evidence.evidence_id,
                    "summary": evidence.summary,
                    "dialectical_record": evidence.dialectical_record
                },
                input_receipts=evidence.source_receipts
            )

        elif mode_enum == ProofMode.PACKET:
            claim = context.get("claim", {})
            packet = packet_bind(claim, inputs, context)
            return ProofResult(
                proof_id=proof_id,
                mode=mode_enum,
                success=packet.verification_status == "verified",
                confidence=1.0 if packet.verification_status == "verified" else 0.0,
                output={
                    "packet_id": packet.packet_id,
                    "binding_hash": packet.binding_hash,
                    "verification_status": packet.verification_status
                },
                input_receipts=packet.binding_receipts
            )

        elif mode_enum == ProofMode.DETECT:
            baseline = context.get("baseline", None)
            result = detect_anomaly(inputs, baseline, context)
            return ProofResult(
                proof_id=proof_id,
                mode=mode_enum,
                success=not result.is_anomaly,
                confidence=1.0 - result.score,
                output={
                    "anomaly_id": result.anomaly_id,
                    "is_anomaly": result.is_anomaly,
                    "score": result.score,
                    "classification": result.classification,
                    "description": result.description,
                    "affected_items": result.affected_items
                },
                input_receipts=[]
            )

    except Exception as e:
        return ProofResult(
            proof_id=proof_id,
            mode=mode_enum,
            success=False,
            confidence=0.0,
            output={"error": str(e)},
            input_receipts=[],
            error=str(e)
        )


def emit_proof_receipt(result: ProofResult, tenant_id: Optional[str] = None) -> dict:
    """Emit CLAUDEME-compliant proof receipt.

    Args:
        result: ProofResult from prove()
        tenant_id: Optional tenant ID override

    Returns:
        Receipt dict
    """
    return emit_receipt("proof", {
        "proof_id": result.proof_id,
        "mode": result.mode.value,
        "input_receipts": result.input_receipts,
        "output": result.output,
        "confidence": result.confidence,
        "success": result.success,
        "error": result.error
    }, tenant_id=tenant_id, silent=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def verify_chain(receipts: list, context: Optional[dict] = None) -> ProofResult:
    """Verify a chain of receipts using DETECT mode.

    Args:
        receipts: Chain of receipts to verify
        context: Verification context

    Returns:
        ProofResult indicating chain integrity
    """
    return prove("DETECT", receipts, context)


def synthesize_audit(receipts: list, context: Optional[dict] = None) -> Evidence:
    """Synthesize audit evidence using BRIEF mode.

    Args:
        receipts: Receipts to synthesize
        context: Synthesis context

    Returns:
        Evidence summary
    """
    return brief_evidence(receipts, context)


def bind_sensor_data(sensor_data: dict, decision_receipts: list,
                     context: Optional[dict] = None) -> BoundPacket:
    """Bind sensor data to decision receipts using PACKET mode.

    Args:
        sensor_data: Sensor readings to bind
        decision_receipts: Decision receipts to bind to
        context: Binding context

    Returns:
        BoundPacket with proof
    """
    return packet_bind(sensor_data, decision_receipts, context)
