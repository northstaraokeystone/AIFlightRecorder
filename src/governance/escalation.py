"""Escalation Routing - Decision Escalation (v2.1)

Routes decisions and interventions to appropriate handlers
based on severity, confidence, and policy.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from ..core import emit_receipt


@dataclass
class EscalationLevel:
    """Escalation level definition."""
    level: int
    name: str
    handlers: List[str]
    timeout_minutes: int
    auto_escalate: bool


@dataclass
class EscalationPath:
    """Escalation path for a decision."""
    decision_id: str
    current_level: int
    path: List[EscalationLevel]
    reason: str
    escalated_at: str
    resolved: bool = False


# Default escalation levels
DEFAULT_ESCALATION_LEVELS = [
    EscalationLevel(
        level=1,
        name="operator",
        handlers=["primary_operator", "backup_operator"],
        timeout_minutes=5,
        auto_escalate=True
    ),
    EscalationLevel(
        level=2,
        name="supervisor",
        handlers=["shift_supervisor", "operations_manager"],
        timeout_minutes=15,
        auto_escalate=True
    ),
    EscalationLevel(
        level=3,
        name="safety_officer",
        handlers=["safety_officer", "chief_safety_officer"],
        timeout_minutes=30,
        auto_escalate=True
    ),
    EscalationLevel(
        level=4,
        name="executive",
        handlers=["operations_director", "cto"],
        timeout_minutes=60,
        auto_escalate=False
    )
]


class EscalationRouter:
    """Routes decisions to appropriate escalation handlers.

    Determines escalation path based on:
    - Decision severity
    - Confidence level
    - Gate tier (RED = higher escalation)
    - Policy requirements
    """

    def __init__(self, levels: Optional[List[EscalationLevel]] = None):
        """Initialize escalation router.

        Args:
            levels: Custom escalation levels or None for defaults
        """
        self._levels = levels or DEFAULT_ESCALATION_LEVELS
        self._active_escalations: Dict[str, EscalationPath] = {}
        self._handlers: Dict[str, Callable] = {}

    def register_handler(self, handler_id: str, handler: Callable):
        """Register an escalation handler.

        Args:
            handler_id: Handler identifier
            handler: Handler function
        """
        self._handlers[handler_id] = handler

    def determine_initial_level(self, decision: dict) -> int:
        """Determine initial escalation level for a decision.

        Args:
            decision: Decision dict

        Returns:
            Initial escalation level (1-4)
        """
        # Start at level 1 by default
        level = 1

        # Check severity
        severity = decision.get("severity", 0)
        if severity > 0.9:
            level = max(level, 3)
        elif severity > 0.7:
            level = max(level, 2)

        # Check confidence
        confidence = decision.get("confidence", 1.0)
        if confidence < 0.5:
            level = max(level, 2)

        # Check gate tier
        gate_tier = decision.get("gate_tier", "GREEN")
        if gate_tier == "RED":
            level = max(level, 2)

        # Check for specific action types
        action_type = decision.get("action", {}).get("type", "")
        if action_type in ("ABORT", "EMERGENCY"):
            level = max(level, 3)

        # Check for anomaly
        if decision.get("is_anomaly", False):
            level = max(level, 2)

        return min(level, len(self._levels))

    def route(self, decision_id: str, decision: dict,
              reason: str) -> EscalationPath:
        """Route a decision for escalation.

        Args:
            decision_id: Decision identifier
            decision: Decision dict
            reason: Reason for escalation

        Returns:
            EscalationPath
        """
        initial_level = self.determine_initial_level(decision)

        # Build path from initial level up
        path = [
            level for level in self._levels
            if level.level >= initial_level
        ]

        escalation = EscalationPath(
            decision_id=decision_id,
            current_level=initial_level,
            path=path,
            reason=reason,
            escalated_at=datetime.now(timezone.utc).isoformat()
        )

        self._active_escalations[decision_id] = escalation

        # Emit receipt
        emit_escalation_receipt(escalation)

        # Notify handlers at current level
        self._notify_handlers(escalation)

        return escalation

    def _notify_handlers(self, escalation: EscalationPath):
        """Notify handlers at current escalation level.

        Args:
            escalation: Current escalation
        """
        current = None
        for level in escalation.path:
            if level.level == escalation.current_level:
                current = level
                break

        if not current:
            return

        for handler_id in current.handlers:
            if handler_id in self._handlers:
                try:
                    self._handlers[handler_id](escalation)
                except Exception:
                    pass  # Don't let handler failure block escalation

    def escalate(self, decision_id: str) -> Optional[EscalationPath]:
        """Escalate to next level.

        Args:
            decision_id: Decision to escalate

        Returns:
            Updated EscalationPath or None if at max level
        """
        if decision_id not in self._active_escalations:
            return None

        escalation = self._active_escalations[decision_id]

        # Find next level
        next_level = None
        for level in escalation.path:
            if level.level > escalation.current_level:
                next_level = level
                break

        if not next_level:
            return None  # Already at max level

        escalation.current_level = next_level.level

        # Emit receipt
        emit_escalation_receipt(escalation)

        # Notify handlers
        self._notify_handlers(escalation)

        return escalation

    def resolve(self, decision_id: str, resolution: dict) -> bool:
        """Resolve an escalation.

        Args:
            decision_id: Decision to resolve
            resolution: Resolution details

        Returns:
            True if resolved, False if not found
        """
        if decision_id not in self._active_escalations:
            return False

        escalation = self._active_escalations[decision_id]
        escalation.resolved = True

        # Emit resolution receipt
        emit_receipt("escalation_resolved", {
            "decision_id": decision_id,
            "final_level": escalation.current_level,
            "resolution": resolution
        }, silent=True)

        # Remove from active
        del self._active_escalations[decision_id]

        return True

    def get_active(self) -> List[EscalationPath]:
        """Get all active escalations.

        Returns:
            List of active escalations
        """
        return list(self._active_escalations.values())

    def get_escalation(self, decision_id: str) -> Optional[EscalationPath]:
        """Get escalation for a decision.

        Args:
            decision_id: Decision ID

        Returns:
            EscalationPath or None
        """
        return self._active_escalations.get(decision_id)


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_escalation_router: Optional[EscalationRouter] = None


def get_escalation_router() -> EscalationRouter:
    """Get the global escalation router."""
    global _escalation_router
    if _escalation_router is None:
        _escalation_router = EscalationRouter()
    return _escalation_router


def route_escalation(decision_id: str, decision: dict,
                     reason: str) -> EscalationPath:
    """Route a decision for escalation.

    Args:
        decision_id: Decision ID
        decision: Decision dict
        reason: Escalation reason

    Returns:
        EscalationPath
    """
    router = get_escalation_router()
    return router.route(decision_id, decision, reason)


def get_escalation_path(decision_id: str) -> Optional[EscalationPath]:
    """Get escalation path for a decision.

    Args:
        decision_id: Decision ID

    Returns:
        EscalationPath or None
    """
    router = get_escalation_router()
    return router.get_escalation(decision_id)


def emit_escalation_receipt(escalation: EscalationPath) -> dict:
    """Emit escalation receipt.

    Args:
        escalation: Escalation path

    Returns:
        Receipt dict
    """
    current_level = None
    for level in escalation.path:
        if level.level == escalation.current_level:
            current_level = level
            break

    return emit_receipt("escalation", {
        "decision_id": escalation.decision_id,
        "current_level": escalation.current_level,
        "level_name": current_level.name if current_level else "unknown",
        "handlers": current_level.handlers if current_level else [],
        "reason": escalation.reason,
        "escalated_at": escalation.escalated_at,
        "timeout_minutes": current_level.timeout_minutes if current_level else 0,
        "auto_escalate": current_level.auto_escalate if current_level else False
    }, silent=True)
