"""Training Example Extractor (v2.1)

Extracts, labels, and scores training examples from human interventions.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash


@dataclass
class TrainingExample:
    """A training example extracted from an intervention."""
    example_id: str
    decision_id: str
    original_decision: dict
    corrected_decision: dict
    reason_code: str
    context: dict
    label: str           # "correct", "incorrect", "ambiguous"
    quality_score: float # 0-1
    created_at: str
    features: Dict[str, Any] = field(default_factory=dict)


# Training queue
_training_queue: List[TrainingExample] = []


def extract_training_example(intervention: dict) -> TrainingExample:
    """Extract a training example from an intervention.

    Args:
        intervention: Intervention receipt data

    Returns:
        TrainingExample
    """
    decision_id = intervention.get("decision_id", str(uuid.uuid4()))
    example_id = str(uuid.uuid4())

    # Build example
    example = TrainingExample(
        example_id=example_id,
        decision_id=decision_id,
        original_decision=intervention.get("original_decision", {}),
        corrected_decision=intervention.get("correction", {}),
        reason_code=intervention.get("reason_code", "UNKNOWN"),
        context=intervention.get("context", {}),
        label="",  # Will be set by label_example
        quality_score=0.0,  # Will be set by score_example
        created_at=datetime.now(timezone.utc).isoformat()
    )

    # Extract features
    example.features = _extract_features(example)

    # Auto-label based on reason code
    example = label_example(example)

    # Score quality
    example = score_example(example)

    # Add to queue
    _training_queue.append(example)

    # Emit receipt
    emit_training_example_receipt(example)

    return example


def _extract_features(example: TrainingExample) -> Dict[str, Any]:
    """Extract features from a training example.

    Args:
        example: Training example

    Returns:
        Feature dict
    """
    features = {}

    # Original decision features
    orig = example.original_decision
    if orig:
        features["original_action"] = orig.get("action", {}).get("type", "")
        features["original_confidence"] = orig.get("confidence", 0.5)
        features["original_severity"] = orig.get("severity", 0)

    # Corrected decision features
    corr = example.corrected_decision
    if corr:
        features["corrected_action"] = corr.get("action", {}).get("type", "")
        features["corrected_confidence"] = corr.get("confidence", 0.5)

    # Compute diff features
    if orig and corr:
        features["action_changed"] = (
            features.get("original_action") != features.get("corrected_action")
        )
        features["confidence_delta"] = (
            features.get("corrected_confidence", 0) -
            features.get("original_confidence", 0)
        )

    # Context features
    ctx = example.context
    if ctx:
        features["has_sensor_data"] = "sensors" in ctx
        features["has_mission_data"] = "mission" in ctx
        features["context_size"] = len(ctx)

    return features


def label_example(example: TrainingExample) -> TrainingExample:
    """Apply label to training example.

    Labels:
    - "correct": AI made wrong decision, human correction is correct
    - "incorrect": Human made error (rare, for audit)
    - "ambiguous": Unclear which is correct

    Args:
        example: Example to label

    Returns:
        Labeled example
    """
    reason_code = example.reason_code

    # Safety overrides are always labeled as AI incorrect
    if reason_code in ("SAFETY_CRITICAL", "IMMINENT_DANGER", "MODEL_ERROR"):
        example.label = "correct"

    # Testing/calibration shouldn't be used for training
    elif reason_code in ("TESTING", "CALIBRATION"):
        example.label = "ambiguous"

    # Operational changes may not indicate AI error
    elif reason_code in ("MISSION_CHANGE", "WEATHER_OVERRIDE"):
        example.label = "ambiguous"

    # Policy/regulatory are external constraints
    elif reason_code in ("POLICY_VIOLATION", "REGULATORY_COMPLIANCE"):
        example.label = "correct"

    # Confidence overrides suggest AI calibration issues
    elif reason_code in ("CONFIDENCE_OVERRIDE", "CONTEXT_MISSING"):
        example.label = "correct"

    # Sensor malfunction is not AI error
    elif reason_code == "SENSOR_MALFUNCTION":
        example.label = "ambiguous"

    # Default
    else:
        example.label = "correct"

    return example


def score_example(example: TrainingExample) -> TrainingExample:
    """Score training example quality.

    Higher scores = better training examples.

    Args:
        example: Example to score

    Returns:
        Scored example
    """
    score = 0.5  # Base score

    # Label affects quality
    if example.label == "correct":
        score += 0.2
    elif example.label == "ambiguous":
        score -= 0.2

    # Complete data improves quality
    if example.original_decision:
        score += 0.1
    if example.corrected_decision:
        score += 0.1
    if example.context:
        score += 0.1

    # Specific reason codes improve quality
    if example.reason_code in ("MODEL_ERROR", "CONFIDENCE_OVERRIDE"):
        score += 0.1

    # Action changed is more informative
    if example.features.get("action_changed", False):
        score += 0.1

    example.quality_score = max(0.0, min(1.0, score))
    return example


def emit_training_example_receipt(example: TrainingExample) -> dict:
    """Emit training example receipt.

    Args:
        example: Training example

    Returns:
        Receipt dict
    """
    return emit_receipt("training_example", {
        "example_id": example.example_id,
        "decision_id": example.decision_id,
        "reason_code": example.reason_code,
        "label": example.label,
        "quality_score": example.quality_score,
        "features": example.features,
        "example_hash": dual_hash(json.dumps({
            "original": example.original_decision,
            "corrected": example.corrected_decision
        }, sort_keys=True))
    }, silent=True)


def get_training_queue() -> List[TrainingExample]:
    """Get all queued training examples.

    Returns:
        List of training examples
    """
    return _training_queue.copy()


def filter_training_queue(min_quality: float = 0.5,
                          labels: Optional[List[str]] = None) -> List[TrainingExample]:
    """Filter training queue by quality and label.

    Args:
        min_quality: Minimum quality score
        labels: Labels to include (None for all)

    Returns:
        Filtered examples
    """
    result = []
    for example in _training_queue:
        if example.quality_score < min_quality:
            continue
        if labels and example.label not in labels:
            continue
        result.append(example)
    return result


def clear_training_queue():
    """Clear the training queue."""
    global _training_queue
    _training_queue = []


class TrainingExampleBuffer:
    """Buffered training example extraction."""

    def __init__(self, max_size: int = 1000):
        self._buffer: List[TrainingExample] = []
        self._max_size = max_size

    def add(self, intervention: dict) -> TrainingExample:
        """Add intervention to buffer.

        Args:
            intervention: Intervention data

        Returns:
            Created training example
        """
        example = extract_training_example(intervention)
        self._buffer.append(example)

        # Trim if over max size
        if len(self._buffer) > self._max_size:
            self._buffer = self._buffer[-self._max_size:]

        return example

    def get_all(self) -> List[TrainingExample]:
        """Get all buffered examples."""
        return self._buffer.copy()

    def get_high_quality(self, min_score: float = 0.7) -> List[TrainingExample]:
        """Get high-quality examples."""
        return [e for e in self._buffer if e.quality_score >= min_score]

    def flush(self) -> List[TrainingExample]:
        """Flush and return all examples."""
        result = self._buffer.copy()
        self._buffer = []
        return result
