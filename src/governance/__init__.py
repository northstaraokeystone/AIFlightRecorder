"""Governance Module - Enterprise Patterns (v2.1)

RACI accountability, provenance tracking, reason codes, and escalation.
"""

from .raci import (
    RACIMatrix,
    assign_accountability,
    get_raci_for_decision,
    validate_raci_coverage,
    emit_raci_receipt
)

from .provenance import (
    capture_provenance,
    get_model_version,
    get_policy_version,
    emit_provenance_receipt
)

from .reason_codes import (
    REASON_CODES,
    validate_reason_code,
    get_reason_code_info,
    emit_intervention_receipt
)

from .escalation import (
    EscalationRouter,
    route_escalation,
    get_escalation_path,
    emit_escalation_receipt
)

__all__ = [
    "RACIMatrix",
    "assign_accountability",
    "get_raci_for_decision",
    "validate_raci_coverage",
    "emit_raci_receipt",
    "capture_provenance",
    "get_model_version",
    "get_policy_version",
    "emit_provenance_receipt",
    "REASON_CODES",
    "validate_reason_code",
    "get_reason_code_info",
    "emit_intervention_receipt",
    "EscalationRouter",
    "route_escalation",
    "get_escalation_path",
    "emit_escalation_receipt"
]
