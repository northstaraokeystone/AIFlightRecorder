"""HUNTER - Proprioception Agent

HUNTER is not scanning FOR the flight recorder.
HUNTER IS the flight recorder being aware.

The system's capacity to feel anomalies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash
from src.compress import ncd, system_entropy
from .immortal import ImmortalAgent


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str
    anomaly_type: str
    severity: float       # 0-1
    decision_id: str
    description: str
    detected_at: str
    ncd_score: Optional[float] = None
    compression_score: Optional[float] = None


class Hunter(ImmortalAgent):
    """HUNTER - The system's proprioception.

    Scans for compression anomalies that indicate:
    - Tampering
    - Drift
    - Unusual patterns
    - System degradation
    """

    def __init__(self):
        super().__init__(name="HUNTER")
        self._baseline_decisions: List[dict] = []
        self._baseline_established = False
        self._ncd_threshold = 0.7
        self._severity_threshold = 0.5

    def scan(self, decisions: List[dict]) -> List[Anomaly]:
        """Scan decisions for compression anomalies.

        Args:
            decisions: Decisions to scan

        Returns:
            List of detected anomalies
        """
        if not self.is_enabled():
            return []

        anomalies = []

        # Build baseline if not established
        if not self._baseline_established and len(decisions) >= 100:
            self._baseline_decisions = decisions[:100]
            self._baseline_established = True

        # Skip if no baseline
        if not self._baseline_established:
            return []

        # Scan recent decisions against baseline
        for decision in decisions[-10:]:  # Last 10 decisions
            # Skip our own receipts
            if self.should_ignore(decision):
                continue

            anomaly = self._check_decision(decision)
            if anomaly:
                anomalies.append(anomaly)
                self._emit_alert(anomaly)

        return anomalies

    def _check_decision(self, decision: dict) -> Optional[Anomaly]:
        """Check a single decision for anomalies.

        Args:
            decision: Decision to check

        Returns:
            Anomaly if detected, None otherwise
        """
        import uuid

        # Get decision data
        if "full_decision" in decision:
            d = decision["full_decision"]
        else:
            d = decision

        decision_id = d.get("decision_id", str(uuid.uuid4()))

        # Calculate NCD against baseline
        ncd_score = ncd([d], self._baseline_decisions[:10])

        # Score severity based on NCD
        severity = self._score_severity(ncd_score, d)

        if severity >= self._severity_threshold:
            # Determine anomaly type
            if ncd_score > 0.9:
                anomaly_type = "tampering"
                description = f"Decision pattern radically different from baseline (NCD={ncd_score:.2f})"
            elif ncd_score > 0.7:
                anomaly_type = "drift"
                description = f"Decision pattern drifting from baseline (NCD={ncd_score:.2f})"
            else:
                anomaly_type = "deviation"
                description = f"Unusual decision pattern detected (severity={severity:.2f})"

            return Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type=anomaly_type,
                severity=severity,
                decision_id=decision_id,
                description=description,
                detected_at=datetime.now(timezone.utc).isoformat(),
                ncd_score=ncd_score
            )

        return None

    def score_severity(self, anomaly: dict) -> float:
        """Score the severity of an anomaly.

        Public interface for external callers.

        Args:
            anomaly: Anomaly dict

        Returns:
            Severity score 0-1
        """
        ncd_score = anomaly.get("ncd_score", 0.5)
        return self._score_severity(ncd_score, anomaly)

    def _score_severity(self, ncd_score: float, decision: dict) -> float:
        """Calculate severity based on NCD and decision properties.

        Args:
            ncd_score: NCD score against baseline
            decision: Decision dict

        Returns:
            Severity score 0-1
        """
        # Base severity from NCD
        if ncd_score > 0.9:
            severity = 1.0
        elif ncd_score > 0.7:
            severity = 0.7 + (ncd_score - 0.7) * 1.5
        elif ncd_score > 0.5:
            severity = 0.4 + (ncd_score - 0.5) * 1.5
        else:
            severity = ncd_score * 0.8

        # Adjust for action type
        if "full_decision" in decision:
            d = decision["full_decision"]
        else:
            d = decision

        action = d.get("action", {}).get("type", "CONTINUE")
        if action in ["ABORT", "RTB"]:
            severity *= 1.2  # Safety-critical actions get higher severity

        # Adjust for confidence
        confidence = d.get("confidence", 0.5)
        if confidence < 0.5:
            severity *= 1.1  # Low confidence + anomaly = more concerning

        return min(1.0, max(0.0, severity))

    def _emit_alert(self, anomaly: Anomaly) -> dict:
        """Emit an anomaly alert receipt.

        Args:
            anomaly: The anomaly to alert on

        Returns:
            Alert receipt
        """
        return emit_receipt("anomaly_alert", {
            "anomaly_id": anomaly.anomaly_id,
            "anomaly_type": anomaly.anomaly_type,
            "severity": anomaly.severity,
            "decision_id": anomaly.decision_id,
            "description": anomaly.description,
            "detected_at": anomaly.detected_at,
            "ncd_score": anomaly.ncd_score,
            "hunter_id": self.self_id,
            "agent_id": self.self_id  # For self-ignore
        }, silent=True)

    def get_baseline_stats(self) -> dict:
        """Get baseline statistics.

        Returns:
            Baseline stats dict
        """
        if not self._baseline_established:
            return {"established": False}

        entropy = system_entropy(self._baseline_decisions)

        return {
            "established": True,
            "decision_count": len(self._baseline_decisions),
            "entropy": entropy
        }


def scan_for_anomalies(decisions: List[dict]) -> List[Anomaly]:
    """Convenience function to scan decisions for anomalies.

    Args:
        decisions: Decisions to scan

    Returns:
        List of anomalies
    """
    from .immortal import get_hunter
    hunter = get_hunter()
    return hunter.scan(decisions)
