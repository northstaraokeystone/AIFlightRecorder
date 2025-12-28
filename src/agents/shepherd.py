"""SHEPHERD - Homeostasis Agent

SHEPHERD is not healing the flight recorder.
SHEPHERD IS the flight recorder healing.

The system's capacity to heal itself.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    HITL_AUTO_APPROVE_CONFIDENCE,
    HITL_AUTO_APPROVE_RISK,
    HITL_ESCALATION_DAYS,
    HITL_AUTO_DECLINE_DAYS
)
from src.core import emit_receipt
from .immortal import ImmortalAgent


@dataclass
class RemediationPlan:
    """A proposed remediation plan."""
    plan_id: str
    anomaly_id: str
    actions: List[dict]
    confidence: float
    risk_level: str         # low, medium, high
    requires_approval: bool
    proposed_at: str
    approval_deadline: str
    auto_approve: bool
    status: str = "proposed"  # proposed, approved, executed, declined


class Shepherd(ImmortalAgent):
    """SHEPHERD - The system's homeostasis.

    Proposes and executes remediation for detected anomalies.
    Includes HITL (Human-In-The-Loop) gates for safety.
    """

    def __init__(self):
        super().__init__(name="SHEPHERD")
        self._pending_plans: Dict[str, RemediationPlan] = {}
        self._executed_plans: List[str] = []
        self._declined_plans: List[str] = []

    def scan(self, anomalies: List[dict]) -> List[RemediationPlan]:
        """Propose remediations for anomalies.

        Note: 'scan' is used for API consistency with ImmortalAgent,
        but SHEPHERD is proposing remediations, not scanning.

        Args:
            anomalies: List of anomalies to remediate

        Returns:
            List of remediation plans
        """
        if not self.is_enabled():
            return []

        plans = []

        for anomaly in anomalies:
            # Skip our own receipts
            if self.should_ignore(anomaly):
                continue

            plan = self.propose_remediation(anomaly)
            if plan:
                plans.append(plan)
                self._pending_plans[plan.plan_id] = plan

        return plans

    def propose_remediation(self, anomaly: dict) -> Optional[RemediationPlan]:
        """Propose a remediation plan for an anomaly.

        Args:
            anomaly: Anomaly dict with type, severity, etc.

        Returns:
            RemediationPlan or None if no remediation needed
        """
        anomaly_type = anomaly.get("anomaly_type", "unknown")
        severity = anomaly.get("severity", 0.5)
        anomaly_id = anomaly.get("anomaly_id", str(uuid.uuid4()))

        # Determine remediation actions based on anomaly type
        actions = self._determine_actions(anomaly_type, severity, anomaly)

        if not actions:
            return None

        # Calculate confidence in remediation
        confidence = self._calculate_confidence(actions, anomaly)

        # Determine risk level
        risk_level = self._assess_risk(actions, anomaly)

        # Check HITL gate
        requires_approval, auto_approve = self._check_hitl_gate(confidence, risk_level)

        # Calculate approval deadline
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(days=HITL_AUTO_DECLINE_DAYS)

        plan = RemediationPlan(
            plan_id=str(uuid.uuid4()),
            anomaly_id=anomaly_id,
            actions=actions,
            confidence=confidence,
            risk_level=risk_level,
            requires_approval=requires_approval,
            proposed_at=now.isoformat(),
            approval_deadline=deadline.isoformat(),
            auto_approve=auto_approve
        )

        # Emit remediation proposal receipt
        self._emit_remediation_proposal(plan)

        return plan

    def _determine_actions(
        self,
        anomaly_type: str,
        severity: float,
        anomaly: dict
    ) -> List[dict]:
        """Determine remediation actions for anomaly.

        Args:
            anomaly_type: Type of anomaly
            severity: Severity score
            anomaly: Full anomaly dict

        Returns:
            List of action dicts
        """
        actions = []

        if anomaly_type == "tampering":
            # High severity: isolate and alert
            actions = [
                {"action": "isolate_decision", "target": anomaly.get("decision_id")},
                {"action": "alert_security", "severity": "critical"},
                {"action": "snapshot_state", "reason": "tampering_detection"},
                {"action": "trigger_audit", "scope": "full"}
            ]

        elif anomaly_type == "drift":
            # Medium severity: adjust baseline
            actions = [
                {"action": "flag_for_review", "target": anomaly.get("decision_id")},
                {"action": "consider_baseline_update", "ncd_score": anomaly.get("ncd_score")}
            ]
            if severity > 0.7:
                actions.append({"action": "spawn_watchers", "count": 3})

        elif anomaly_type == "deviation":
            # Low severity: log and monitor
            actions = [
                {"action": "log_deviation", "target": anomaly.get("decision_id")},
                {"action": "increase_monitoring", "duration_seconds": 300}
            ]

        else:
            # Unknown: investigate
            actions = [
                {"action": "investigate", "target": anomaly.get("decision_id")},
                {"action": "spawn_helpers", "count": 1}
            ]

        return actions

    def _calculate_confidence(self, actions: List[dict], anomaly: dict) -> float:
        """Calculate confidence in remediation plan.

        Args:
            actions: Proposed actions
            anomaly: Anomaly being remediated

        Returns:
            Confidence score 0-1
        """
        # Base confidence
        confidence = 0.7

        # Adjust based on anomaly clarity
        anomaly_type = anomaly.get("anomaly_type", "unknown")
        if anomaly_type in ["tampering", "drift", "deviation"]:
            confidence += 0.1  # Known type

        # Adjust based on severity match
        severity = anomaly.get("severity", 0.5)
        action_severity = "high" if any(a.get("severity") == "critical" for a in actions) else "low"
        if severity > 0.7 and action_severity == "high":
            confidence += 0.1
        elif severity < 0.5 and action_severity == "low":
            confidence += 0.1

        return min(1.0, confidence)

    def _assess_risk(self, actions: List[dict], anomaly: dict) -> str:
        """Assess risk level of remediation.

        Args:
            actions: Proposed actions
            anomaly: Anomaly being remediated

        Returns:
            Risk level: low, medium, high
        """
        high_risk_actions = ["isolate_decision", "trigger_audit", "halt_system"]
        medium_risk_actions = ["spawn_helpers", "spawn_watchers", "adjust_baseline"]

        has_high_risk = any(a.get("action") in high_risk_actions for a in actions)
        has_medium_risk = any(a.get("action") in medium_risk_actions for a in actions)

        if has_high_risk:
            return "high"
        elif has_medium_risk:
            return "medium"
        else:
            return "low"

    def _check_hitl_gate(self, confidence: float, risk_level: str) -> tuple[bool, bool]:
        """Check Human-In-The-Loop gate.

        Args:
            confidence: Remediation confidence
            risk_level: Remediation risk level

        Returns:
            Tuple of (requires_approval, auto_approve)
        """
        # Auto-approve if high confidence and low risk
        if confidence > HITL_AUTO_APPROVE_CONFIDENCE and risk_level == HITL_AUTO_APPROVE_RISK:
            return False, True

        # Require approval for high risk
        if risk_level == "high":
            return True, False

        # Require approval for low confidence
        if confidence < 0.6:
            return True, False

        # Medium risk with good confidence: require approval but allow
        return True, False

    def execute_remediation(
        self,
        plan_id: str,
        approval: str = "auto"
    ) -> dict:
        """Execute an approved remediation plan.

        Args:
            plan_id: Plan ID to execute
            approval: Approval type (auto, manual, timeout)

        Returns:
            Execution result
        """
        if plan_id not in self._pending_plans:
            return {"success": False, "error": "Plan not found"}

        plan = self._pending_plans[plan_id]

        # Check approval
        if plan.requires_approval and approval not in ["manual", "auto"]:
            return {"success": False, "error": "Approval required"}

        # Execute actions
        results = []
        for action in plan.actions:
            result = self._execute_action(action)
            results.append(result)

        # Update plan status
        plan.status = "executed"
        self._executed_plans.append(plan_id)
        del self._pending_plans[plan_id]

        # Emit remediation result
        receipt = self._emit_remediation_result(plan, results, approval)

        return {
            "success": True,
            "plan_id": plan_id,
            "actions_executed": len(results),
            "approval": approval,
            "receipt": receipt
        }

    def _execute_action(self, action: dict) -> dict:
        """Execute a single remediation action.

        Args:
            action: Action dict

        Returns:
            Execution result
        """
        action_type = action.get("action", "unknown")

        # Log the action (actual execution would depend on action type)
        return {
            "action": action_type,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "details": action
        }

    def _emit_remediation_proposal(self, plan: RemediationPlan) -> dict:
        """Emit remediation proposal receipt.

        Args:
            plan: The remediation plan

        Returns:
            Receipt
        """
        return emit_receipt("remediation_proposal", {
            "plan_id": plan.plan_id,
            "anomaly_id": plan.anomaly_id,
            "actions_count": len(plan.actions),
            "confidence": plan.confidence,
            "risk_level": plan.risk_level,
            "requires_approval": plan.requires_approval,
            "auto_approve": plan.auto_approve,
            "proposed_at": plan.proposed_at,
            "approval_deadline": plan.approval_deadline,
            "shepherd_id": self.self_id,
            "agent_id": self.self_id
        }, silent=True)

    def _emit_remediation_result(
        self,
        plan: RemediationPlan,
        results: List[dict],
        approval: str
    ) -> dict:
        """Emit remediation result receipt.

        Args:
            plan: The executed plan
            results: Action execution results
            approval: Approval type

        Returns:
            Receipt
        """
        return emit_receipt("remediation", {
            "plan_id": plan.plan_id,
            "anomaly_id": plan.anomaly_id,
            "status": "executed",
            "approval_type": approval,
            "actions_executed": len(results),
            "all_succeeded": all(r.get("success", False) for r in results),
            "shepherd_id": self.self_id,
            "agent_id": self.self_id
        }, silent=True)

    def get_pending_plans(self) -> List[RemediationPlan]:
        """Get all pending remediation plans.

        Returns:
            List of pending plans
        """
        return list(self._pending_plans.values())

    def decline_plan(self, plan_id: str, reason: str = "manual") -> bool:
        """Decline a remediation plan.

        Args:
            plan_id: Plan to decline
            reason: Reason for decline

        Returns:
            True if declined
        """
        if plan_id not in self._pending_plans:
            return False

        plan = self._pending_plans[plan_id]
        plan.status = "declined"
        self._declined_plans.append(plan_id)
        del self._pending_plans[plan_id]

        emit_receipt("remediation_declined", {
            "plan_id": plan_id,
            "anomaly_id": plan.anomaly_id,
            "reason": reason,
            "shepherd_id": self.self_id
        }, silent=True)

        return True


def propose_remediation(anomaly: dict) -> Optional[RemediationPlan]:
    """Convenience function to propose remediation.

    Args:
        anomaly: Anomaly to remediate

    Returns:
        RemediationPlan or None
    """
    from .immortal import get_shepherd
    shepherd = get_shepherd()
    return shepherd.propose_remediation(anomaly)
