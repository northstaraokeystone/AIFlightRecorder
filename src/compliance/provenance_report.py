"""Provenance Report Generation (v2.1)

Generates provenance reports showing model/policy version history
and detecting drift.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..core import emit_receipt, load_receipts


@dataclass
class ProvenanceReport:
    """Provenance report for a time period."""
    report_id: str
    start_time: str
    end_time: str
    generated_at: str
    model_versions: List[dict]
    policy_versions: List[dict]
    config_versions: List[dict]
    drift_events: List[dict]
    coverage: float


def generate_provenance_report(start_time: Optional[str] = None,
                               end_time: Optional[str] = None) -> ProvenanceReport:
    """Generate provenance report for a time period.

    Args:
        start_time: Start time (ISO8601)
        end_time: End time (ISO8601)

    Returns:
        ProvenanceReport
    """
    import uuid

    all_receipts = load_receipts()

    # Filter by time and type
    provenance_receipts = []
    decision_count = 0

    for r in all_receipts:
        ts = r.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue

        if r.get("receipt_type") == "provenance":
            provenance_receipts.append(r)
        elif r.get("receipt_type") in ("decision", "decision_log"):
            decision_count += 1

    # Track version history
    model_versions = []
    policy_versions = []
    config_versions = []

    seen_model = set()
    seen_policy = set()
    seen_config = set()

    for p in provenance_receipts:
        mv = p.get("model_version")
        mh = p.get("model_hash", "")[:16]
        if mv and (mv, mh) not in seen_model:
            seen_model.add((mv, mh))
            model_versions.append({
                "version": mv,
                "hash": p.get("model_hash"),
                "first_seen": p.get("ts")
            })

        pv = p.get("policy_version")
        ph = p.get("policy_hash", "")[:16]
        if pv and (pv, ph) not in seen_policy:
            seen_policy.add((pv, ph))
            policy_versions.append({
                "version": pv,
                "hash": p.get("policy_hash"),
                "first_seen": p.get("ts")
            })

        cv = p.get("config_version")
        ch = p.get("config_hash", "")[:16]
        if cv and (cv, ch) not in seen_config:
            seen_config.add((cv, ch))
            config_versions.append({
                "version": cv,
                "hash": p.get("config_hash"),
                "first_seen": p.get("ts")
            })

    # Detect drift events
    drift_events = detect_provenance_drift(provenance_receipts)

    # Calculate coverage
    coverage = len(provenance_receipts) / max(1, decision_count)

    report = ProvenanceReport(
        report_id=str(uuid.uuid4()),
        start_time=start_time or "beginning",
        end_time=end_time or datetime.now(timezone.utc).isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_versions=model_versions,
        policy_versions=policy_versions,
        config_versions=config_versions,
        drift_events=drift_events,
        coverage=coverage
    )

    # Emit receipt
    emit_receipt("provenance_report", {
        "report_id": report.report_id,
        "start_time": report.start_time,
        "end_time": report.end_time,
        "model_version_count": len(model_versions),
        "policy_version_count": len(policy_versions),
        "config_version_count": len(config_versions),
        "drift_event_count": len(drift_events),
        "coverage": coverage
    }, silent=True)

    return report


def detect_provenance_drift(provenance_receipts: List[dict]) -> List[dict]:
    """Detect provenance drift events.

    Args:
        provenance_receipts: List of provenance receipts

    Returns:
        List of drift events
    """
    if len(provenance_receipts) < 2:
        return []

    # Sort by timestamp
    sorted_receipts = sorted(provenance_receipts, key=lambda x: x.get("ts", ""))

    drift_events = []
    prev = None

    for curr in sorted_receipts:
        if prev is None:
            prev = curr
            continue

        # Check for model drift
        if prev.get("model_version") != curr.get("model_version"):
            drift_events.append({
                "drift_type": "model_version",
                "from_version": prev.get("model_version"),
                "to_version": curr.get("model_version"),
                "detected_at": curr.get("ts"),
                "decision_id": curr.get("decision_id")
            })

        if prev.get("model_hash") != curr.get("model_hash"):
            # Hash changed but version same = sneaky update
            if prev.get("model_version") == curr.get("model_version"):
                drift_events.append({
                    "drift_type": "model_hash_silent",
                    "version": curr.get("model_version"),
                    "from_hash": prev.get("model_hash", "")[:16],
                    "to_hash": curr.get("model_hash", "")[:16],
                    "detected_at": curr.get("ts"),
                    "severity": "high"
                })

        # Check for policy drift
        if prev.get("policy_version") != curr.get("policy_version"):
            drift_events.append({
                "drift_type": "policy_version",
                "from_version": prev.get("policy_version"),
                "to_version": curr.get("policy_version"),
                "detected_at": curr.get("ts")
            })

        # Check for config drift
        if prev.get("config_version") != curr.get("config_version"):
            drift_events.append({
                "drift_type": "config_version",
                "from_version": prev.get("config_version"),
                "to_version": curr.get("config_version"),
                "detected_at": curr.get("ts")
            })

        prev = curr

    return drift_events


def get_current_provenance() -> dict:
    """Get the most recent provenance information.

    Returns:
        Current provenance dict
    """
    all_receipts = load_receipts()

    # Find most recent provenance receipt
    provenance_receipts = [
        r for r in all_receipts
        if r.get("receipt_type") == "provenance"
    ]

    if not provenance_receipts:
        return {
            "available": False,
            "message": "No provenance data available"
        }

    latest = provenance_receipts[-1]

    return {
        "available": True,
        "model_version": latest.get("model_version"),
        "model_hash": latest.get("model_hash"),
        "policy_version": latest.get("policy_version"),
        "policy_hash": latest.get("policy_hash"),
        "config_version": latest.get("config_version"),
        "config_hash": latest.get("config_hash"),
        "as_of": latest.get("ts")
    }
