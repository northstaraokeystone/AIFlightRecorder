"""Audit Trail Generation (v2.1)

Generates comprehensive audit trails for regulatory compliance.
Uses proof.py BRIEF mode for evidence synthesis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, load_receipts
from ..proof import prove, synthesize_audit, Evidence


@dataclass
class AuditTrail:
    """Complete audit trail for a time period."""
    trail_id: str
    start_time: str
    end_time: str
    generated_at: str
    decision_count: int
    intervention_count: int
    anomaly_count: int
    evidence_summary: Evidence
    raci_coverage: float
    provenance_coverage: float
    findings: List[dict]
    compliance_status: str  # "compliant", "non_compliant", "review_required"


def generate_audit_trail(start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         include_details: bool = True) -> AuditTrail:
    """Generate audit trail for a time period.

    Args:
        start_time: Start of audit period (ISO8601)
        end_time: End of audit period (ISO8601)
        include_details: Whether to include detailed findings

    Returns:
        AuditTrail
    """
    import uuid

    # Load all receipts
    all_receipts = load_receipts()

    # Filter by time range
    receipts = []
    for r in all_receipts:
        ts = r.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        receipts.append(r)

    # Count by type
    decisions = [r for r in receipts if r.get("receipt_type") in ("decision", "decision_log")]
    interventions = [r for r in receipts if r.get("receipt_type") == "intervention"]
    anomalies = [r for r in receipts if r.get("receipt_type") in ("anomaly", "anomaly_alert")]
    raci_receipts = [r for r in receipts if r.get("receipt_type") == "raci"]
    provenance_receipts = [r for r in receipts if r.get("receipt_type") == "provenance"]

    # Calculate coverage
    raci_coverage = len(raci_receipts) / max(1, len(decisions))
    provenance_coverage = len(provenance_receipts) / max(1, len(decisions))

    # Synthesize evidence using proof.py BRIEF mode
    evidence = synthesize_audit(receipts)

    # Generate findings
    findings = []
    if include_details:
        findings = _generate_findings(receipts, decisions, interventions, anomalies)

    # Determine compliance status
    compliance_status = _determine_compliance(
        raci_coverage, provenance_coverage, findings
    )

    trail = AuditTrail(
        trail_id=str(uuid.uuid4()),
        start_time=start_time or "beginning",
        end_time=end_time or datetime.now(timezone.utc).isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        decision_count=len(decisions),
        intervention_count=len(interventions),
        anomaly_count=len(anomalies),
        evidence_summary=evidence,
        raci_coverage=raci_coverage,
        provenance_coverage=provenance_coverage,
        findings=findings,
        compliance_status=compliance_status
    )

    # Emit receipt
    emit_audit_trail_receipt(trail)

    return trail


def _generate_findings(receipts: list, decisions: list,
                       interventions: list, anomalies: list) -> List[dict]:
    """Generate audit findings.

    Args:
        receipts: All receipts
        decisions: Decision receipts
        interventions: Intervention receipts
        anomalies: Anomaly receipts

    Returns:
        List of findings
    """
    findings = []

    # Check for unresolved anomalies
    unresolved_anomalies = []
    for a in anomalies:
        anomaly_id = a.get("anomaly_id")
        # Check if there's a corresponding resolution
        resolved = False
        for r in receipts:
            if r.get("receipt_type") == "remediation" and r.get("anomaly_id") == anomaly_id:
                if r.get("status") == "completed":
                    resolved = True
                    break
        if not resolved:
            unresolved_anomalies.append(anomaly_id)

    if unresolved_anomalies:
        findings.append({
            "type": "unresolved_anomalies",
            "severity": "medium",
            "count": len(unresolved_anomalies),
            "description": f"{len(unresolved_anomalies)} anomalies without resolution",
            "affected_ids": unresolved_anomalies[:10]  # Limit
        })

    # Check for decisions without RACI
    raci_decision_ids = set()
    for r in receipts:
        if r.get("receipt_type") == "raci":
            raci_decision_ids.add(r.get("decision_id"))

    missing_raci = []
    for d in decisions:
        if d.get("decision_id") not in raci_decision_ids:
            missing_raci.append(d.get("decision_id"))

    if missing_raci:
        findings.append({
            "type": "missing_raci",
            "severity": "high",
            "count": len(missing_raci),
            "description": f"{len(missing_raci)} decisions without RACI assignment",
            "affected_ids": missing_raci[:10]
        })

    # Check for interventions requiring reports
    reportable_interventions = []
    for i in interventions:
        code_info = i.get("reason_code_info", {})
        if code_info.get("requires_report", False):
            reportable_interventions.append(i.get("decision_id"))

    if reportable_interventions:
        findings.append({
            "type": "reportable_interventions",
            "severity": "info",
            "count": len(reportable_interventions),
            "description": f"{len(reportable_interventions)} interventions require regulatory reporting",
            "affected_ids": reportable_interventions[:10]
        })

    # Check for high-severity anomalies
    high_severity = [a for a in anomalies if a.get("severity", 0) > 0.8]
    if high_severity:
        findings.append({
            "type": "high_severity_anomalies",
            "severity": "high",
            "count": len(high_severity),
            "description": f"{len(high_severity)} high-severity anomalies detected",
            "affected_ids": [a.get("anomaly_id") for a in high_severity[:10]]
        })

    return findings


def _determine_compliance(raci_coverage: float,
                          provenance_coverage: float,
                          findings: list) -> str:
    """Determine overall compliance status.

    Args:
        raci_coverage: RACI coverage percentage
        provenance_coverage: Provenance coverage percentage
        findings: Audit findings

    Returns:
        Compliance status
    """
    # Check for critical findings
    high_severity_findings = [f for f in findings if f.get("severity") == "high"]

    if high_severity_findings:
        return "non_compliant"

    # Check coverage thresholds
    if raci_coverage < 0.95 or provenance_coverage < 0.95:
        return "review_required"

    # Check for medium findings
    medium_findings = [f for f in findings if f.get("severity") == "medium"]
    if len(medium_findings) > 3:
        return "review_required"

    return "compliant"


def emit_audit_trail_receipt(trail: AuditTrail) -> dict:
    """Emit audit trail receipt.

    Args:
        trail: AuditTrail

    Returns:
        Receipt dict
    """
    return emit_receipt("audit_trail", {
        "trail_id": trail.trail_id,
        "start_time": trail.start_time,
        "end_time": trail.end_time,
        "generated_at": trail.generated_at,
        "decision_count": trail.decision_count,
        "intervention_count": trail.intervention_count,
        "anomaly_count": trail.anomaly_count,
        "raci_coverage": trail.raci_coverage,
        "provenance_coverage": trail.provenance_coverage,
        "finding_count": len(trail.findings),
        "compliance_status": trail.compliance_status,
        "evidence_confidence": trail.evidence_summary.confidence
    }, silent=True)


def get_audit_summary(start_time: Optional[str] = None,
                      end_time: Optional[str] = None) -> dict:
    """Get a quick audit summary without full trail generation.

    Args:
        start_time: Start time
        end_time: End time

    Returns:
        Summary dict
    """
    all_receipts = load_receipts()

    # Filter by time
    receipts = []
    for r in all_receipts:
        ts = r.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        receipts.append(r)

    # Count by type
    type_counts = {}
    for r in receipts:
        rt = r.get("receipt_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    return {
        "total_receipts": len(receipts),
        "by_type": type_counts,
        "time_range": {
            "start": start_time or "beginning",
            "end": end_time or "now"
        }
    }
