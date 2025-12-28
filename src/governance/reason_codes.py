"""Reason Codes - Intervention Classification (v2.1)

Standardized reason codes for human interventions.
Every intervention must have a valid reason code.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..core import emit_receipt


# Standard reason codes
REASON_CODES = {
    # Safety overrides
    "SAFETY_CRITICAL": {
        "code": "S001",
        "category": "safety",
        "description": "Safety-critical override by human operator",
        "requires_report": True,
        "auto_training": True
    },
    "IMMINENT_DANGER": {
        "code": "S002",
        "category": "safety",
        "description": "Imminent danger detected, emergency intervention",
        "requires_report": True,
        "auto_training": True
    },
    "SENSOR_MALFUNCTION": {
        "code": "S003",
        "category": "safety",
        "description": "Sensor malfunction suspected or confirmed",
        "requires_report": True,
        "auto_training": False
    },

    # Operational adjustments
    "MISSION_CHANGE": {
        "code": "O001",
        "category": "operational",
        "description": "Mission parameters changed",
        "requires_report": False,
        "auto_training": False
    },
    "WEATHER_OVERRIDE": {
        "code": "O002",
        "category": "operational",
        "description": "Weather conditions require manual override",
        "requires_report": False,
        "auto_training": True
    },
    "AIRSPACE_RESTRICTION": {
        "code": "O003",
        "category": "operational",
        "description": "Airspace restriction requires path change",
        "requires_report": True,
        "auto_training": True
    },

    # Model corrections
    "MODEL_ERROR": {
        "code": "M001",
        "category": "model",
        "description": "Model made incorrect decision",
        "requires_report": True,
        "auto_training": True
    },
    "CONFIDENCE_OVERRIDE": {
        "code": "M002",
        "category": "model",
        "description": "Operator disagrees with AI confidence assessment",
        "requires_report": False,
        "auto_training": True
    },
    "CONTEXT_MISSING": {
        "code": "M003",
        "category": "model",
        "description": "AI missing relevant context that operator has",
        "requires_report": False,
        "auto_training": True
    },

    # Policy enforcement
    "POLICY_VIOLATION": {
        "code": "P001",
        "category": "policy",
        "description": "Decision would violate policy",
        "requires_report": True,
        "auto_training": False
    },
    "REGULATORY_COMPLIANCE": {
        "code": "P002",
        "category": "policy",
        "description": "Regulatory compliance requires override",
        "requires_report": True,
        "auto_training": False
    },

    # Testing and maintenance
    "TESTING": {
        "code": "T001",
        "category": "testing",
        "description": "Test intervention, not production",
        "requires_report": False,
        "auto_training": False
    },
    "CALIBRATION": {
        "code": "T002",
        "category": "testing",
        "description": "System calibration intervention",
        "requires_report": False,
        "auto_training": False
    },

    # Other
    "OTHER": {
        "code": "X001",
        "category": "other",
        "description": "Other reason (requires notes)",
        "requires_report": True,
        "auto_training": False
    }
}


def validate_reason_code(code: str) -> tuple[bool, str]:
    """Validate that a reason code is valid.

    Args:
        code: Reason code to validate

    Returns:
        Tuple of (is_valid, message)
    """
    # Check by code name
    if code in REASON_CODES:
        return True, f"Valid reason code: {code}"

    # Check by code value (e.g., "S001")
    for name, info in REASON_CODES.items():
        if info["code"] == code:
            return True, f"Valid reason code: {name}"

    return False, f"Invalid reason code: {code}"


def get_reason_code_info(code: str) -> Optional[dict]:
    """Get information about a reason code.

    Args:
        code: Reason code

    Returns:
        Reason code info or None
    """
    # Check by code name
    if code in REASON_CODES:
        return {
            "name": code,
            **REASON_CODES[code]
        }

    # Check by code value
    for name, info in REASON_CODES.items():
        if info["code"] == code:
            return {
                "name": name,
                **info
            }

    return None


def get_codes_by_category(category: str) -> List[dict]:
    """Get all reason codes in a category.

    Args:
        category: Category name

    Returns:
        List of reason codes
    """
    return [
        {"name": name, **info}
        for name, info in REASON_CODES.items()
        if info["category"] == category
    ]


def get_training_eligible_codes() -> List[str]:
    """Get reason codes that should generate training data.

    Returns:
        List of code names
    """
    return [
        name for name, info in REASON_CODES.items()
        if info.get("auto_training", False)
    ]


def get_reportable_codes() -> List[str]:
    """Get reason codes that require regulatory reporting.

    Returns:
        List of code names
    """
    return [
        name for name, info in REASON_CODES.items()
        if info.get("requires_report", False)
    ]


def emit_intervention_receipt(decision_id: str, reason_code: str,
                              correction: dict, operator_id: str = "unknown",
                              notes: Optional[str] = None) -> dict:
    """Emit intervention receipt.

    Args:
        decision_id: Decision being overridden
        reason_code: Reason for intervention
        correction: Corrected decision data
        operator_id: ID of operator making intervention
        notes: Optional notes

    Returns:
        Receipt dict
    """
    # Validate reason code
    is_valid, message = validate_reason_code(reason_code)
    if not is_valid:
        # Still emit but mark as invalid
        pass

    code_info = get_reason_code_info(reason_code)

    receipt_data = {
        "decision_id": decision_id,
        "reason_code": reason_code,
        "reason_code_info": code_info,
        "correction": correction,
        "operator_id": operator_id,
        "valid_code": is_valid,
        "requires_report": code_info.get("requires_report", False) if code_info else False,
        "auto_training": code_info.get("auto_training", False) if code_info else False
    }

    if notes:
        receipt_data["notes"] = notes

    return emit_receipt("intervention", receipt_data, silent=True)


class InterventionTracker:
    """Tracks interventions for reporting and training."""

    def __init__(self):
        self._interventions: List[dict] = []

    def record(self, decision_id: str, reason_code: str,
               correction: dict, operator_id: str = "unknown",
               notes: Optional[str] = None) -> dict:
        """Record an intervention.

        Args:
            decision_id: Decision overridden
            reason_code: Reason code
            correction: Correction data
            operator_id: Operator ID
            notes: Optional notes

        Returns:
            Intervention record
        """
        record = {
            "decision_id": decision_id,
            "reason_code": reason_code,
            "correction": correction,
            "operator_id": operator_id,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self._interventions.append(record)
        emit_intervention_receipt(decision_id, reason_code, correction,
                                  operator_id, notes)

        return record

    def get_reportable(self) -> List[dict]:
        """Get interventions that require reporting.

        Returns:
            List of reportable interventions
        """
        reportable_codes = set(get_reportable_codes())
        return [
            i for i in self._interventions
            if i["reason_code"] in reportable_codes
        ]

    def get_training_eligible(self) -> List[dict]:
        """Get interventions eligible for training data.

        Returns:
            List of training-eligible interventions
        """
        training_codes = set(get_training_eligible_codes())
        return [
            i for i in self._interventions
            if i["reason_code"] in training_codes
        ]

    def get_by_category(self, category: str) -> List[dict]:
        """Get interventions by category.

        Args:
            category: Category name

        Returns:
            List of interventions
        """
        category_codes = set(
            name for name, info in REASON_CODES.items()
            if info["category"] == category
        )
        return [
            i for i in self._interventions
            if i["reason_code"] in category_codes
        ]

    def get_stats(self) -> dict:
        """Get intervention statistics.

        Returns:
            Stats dict
        """
        by_code = {}
        by_category = {}

        for i in self._interventions:
            code = i["reason_code"]
            by_code[code] = by_code.get(code, 0) + 1

            info = get_reason_code_info(code)
            if info:
                cat = info.get("category", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total": len(self._interventions),
            "by_code": by_code,
            "by_category": by_category,
            "reportable_count": len(self.get_reportable()),
            "training_eligible_count": len(self.get_training_eligible())
        }
