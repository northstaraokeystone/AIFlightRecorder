"""RACI Matrix - Accountability Assignment (v2.1)

Assigns Responsible, Accountable, Consulted, Informed roles
to every decision for enterprise governance.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..core import emit_receipt


@dataclass
class RACIAssignment:
    """RACI assignment for a decision."""
    decision_id: str
    responsible: str      # Who does the work
    accountable: str      # Who approves/owns
    consulted: List[str]  # Who provides input
    informed: List[str]   # Who needs to know
    decision_type: str
    timestamp: str


class RACIMatrix:
    """RACI matrix manager for decision accountability.

    Loads RACI configuration from raci_matrix.json and assigns
    accountability based on decision type and context.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize RACI matrix.

        Args:
            config_path: Path to raci_matrix.json
        """
        self._matrix: Dict[str, Dict] = {}
        self._default_raci = {
            "responsible": "ai_system",
            "accountable": "operator",
            "consulted": [],
            "informed": ["audit_log"]
        }

        # Load configuration
        if config_path:
            self._load_config(config_path)
        else:
            # Try default location
            default_path = Path(__file__).parent.parent.parent / "config" / "raci_matrix.json"
            if default_path.exists():
                self._load_config(str(default_path))
            else:
                self._setup_default_matrix()

    def _load_config(self, config_path: str):
        """Load RACI configuration from file.

        Args:
            config_path: Path to configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._matrix = config.get("decision_types", {})
                if "default" in config:
                    self._default_raci = config["default"]
        except (FileNotFoundError, json.JSONDecodeError):
            self._setup_default_matrix()

    def _setup_default_matrix(self):
        """Set up default RACI assignments."""
        self._matrix = {
            "navigation": {
                "responsible": "ai_system",
                "accountable": "flight_controller",
                "consulted": ["path_planner"],
                "informed": ["telemetry", "audit_log"]
            },
            "avoidance": {
                "responsible": "ai_system",
                "accountable": "safety_officer",
                "consulted": ["sensor_fusion", "threat_detection"],
                "informed": ["ground_control", "audit_log"]
            },
            "abort": {
                "responsible": "ai_system",
                "accountable": "safety_officer",
                "consulted": ["ground_control"],
                "informed": ["operator", "regulatory", "audit_log"]
            },
            "rtb": {
                "responsible": "ai_system",
                "accountable": "flight_controller",
                "consulted": [],
                "informed": ["ground_control", "audit_log"]
            },
            "sensor_override": {
                "responsible": "operator",
                "accountable": "safety_officer",
                "consulted": ["ai_system", "sensor_fusion"],
                "informed": ["audit_log", "regulatory"]
            },
            "human_intervention": {
                "responsible": "operator",
                "accountable": "safety_officer",
                "consulted": [],
                "informed": ["ai_system", "audit_log", "regulatory"]
            }
        }

    def get_assignment(self, decision_type: str,
                       context: Optional[dict] = None) -> dict:
        """Get RACI assignment for a decision type.

        Args:
            decision_type: Type of decision
            context: Optional context for dynamic assignment

        Returns:
            RACI assignment dict
        """
        # Normalize decision type
        dt_lower = decision_type.lower().replace(" ", "_")

        # Look up in matrix
        if dt_lower in self._matrix:
            base_raci = self._matrix[dt_lower].copy()
        else:
            base_raci = self._default_raci.copy()

        # Apply context-based adjustments
        if context:
            base_raci = self._apply_context(base_raci, context)

        return base_raci

    def _apply_context(self, raci: dict, context: dict) -> dict:
        """Apply context-based RACI adjustments.

        Args:
            raci: Base RACI assignment
            context: Decision context

        Returns:
            Adjusted RACI
        """
        result = raci.copy()

        # High severity escalates accountability
        severity = context.get("severity", 0)
        if severity > 0.8:
            if "safety_officer" not in result.get("consulted", []):
                result.setdefault("consulted", []).append("safety_officer")
            if "regulatory" not in result.get("informed", []):
                result.setdefault("informed", []).append("regulatory")

        # Low confidence adds consultants
        confidence = context.get("confidence", 1.0)
        if confidence < 0.7:
            if "ground_control" not in result.get("consulted", []):
                result.setdefault("consulted", []).append("ground_control")

        # Anomaly adds audit trail
        if context.get("is_anomaly", False):
            if "incident_manager" not in result.get("informed", []):
                result.setdefault("informed", []).append("incident_manager")

        return result

    def validate_coverage(self, decision: dict) -> tuple[bool, list]:
        """Validate that decision has complete RACI coverage.

        Args:
            decision: Decision to validate

        Returns:
            Tuple of (is_valid, missing_roles)
        """
        raci = decision.get("raci", {})
        missing = []

        # Required roles
        if not raci.get("responsible"):
            missing.append("responsible")
        if not raci.get("accountable"):
            missing.append("accountable")

        # informed and consulted can be empty but must exist
        if "consulted" not in raci:
            missing.append("consulted")
        if "informed" not in raci:
            missing.append("informed")

        return len(missing) == 0, missing


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_raci_matrix: Optional[RACIMatrix] = None


def get_raci_matrix() -> RACIMatrix:
    """Get the global RACI matrix."""
    global _raci_matrix
    if _raci_matrix is None:
        _raci_matrix = RACIMatrix()
    return _raci_matrix


def assign_accountability(decision_id: str, decision_type: str,
                          context: Optional[dict] = None) -> RACIAssignment:
    """Assign RACI accountability to a decision.

    Args:
        decision_id: Decision identifier
        decision_type: Type of decision
        context: Decision context

    Returns:
        RACIAssignment
    """
    matrix = get_raci_matrix()
    raci = matrix.get_assignment(decision_type, context)

    assignment = RACIAssignment(
        decision_id=decision_id,
        responsible=raci.get("responsible", "ai_system"),
        accountable=raci.get("accountable", "operator"),
        consulted=raci.get("consulted", []),
        informed=raci.get("informed", []),
        decision_type=decision_type,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    # Emit receipt
    emit_raci_receipt(assignment)

    return assignment


def get_raci_for_decision(decision: dict) -> dict:
    """Get RACI assignment for a decision dict.

    Args:
        decision: Decision dict

    Returns:
        RACI assignment dict
    """
    decision_type = decision.get("action", {}).get("type", "unknown")
    context = {
        "confidence": decision.get("confidence", 0.5),
        "severity": decision.get("severity", 0),
        "is_anomaly": decision.get("is_anomaly", False)
    }

    matrix = get_raci_matrix()
    return matrix.get_assignment(decision_type, context)


def validate_raci_coverage(decision: dict) -> tuple[bool, list]:
    """Validate RACI coverage for a decision.

    Args:
        decision: Decision to validate

    Returns:
        Tuple of (is_valid, missing_roles)
    """
    matrix = get_raci_matrix()
    return matrix.validate_coverage(decision)


def emit_raci_receipt(assignment: RACIAssignment) -> dict:
    """Emit RACI assignment receipt.

    Args:
        assignment: RACI assignment

    Returns:
        Receipt dict
    """
    return emit_receipt("raci", {
        "decision_id": assignment.decision_id,
        "decision_type": assignment.decision_type,
        "responsible": assignment.responsible,
        "accountable": assignment.accountable,
        "consulted": assignment.consulted,
        "informed": assignment.informed
    }, silent=True)
