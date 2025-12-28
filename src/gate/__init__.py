"""Confidence Gate Module - v2.0

Three-tier confidence gating for decision flow control:
- GREEN (>0.9): High confidence, proceed with learning
- YELLOW (0.7-0.9): Medium confidence, proceed with monitoring
- RED (<0.7): Low confidence, block and investigate
"""

from .confidence import calculate_confidence, ConfidenceScore
from .decision import decide, GateDecision
from .drift import measure_drift, DriftScore

__all__ = [
    "calculate_confidence",
    "ConfidenceScore",
    "decide",
    "GateDecision",
    "measure_drift",
    "DriftScore"
]
