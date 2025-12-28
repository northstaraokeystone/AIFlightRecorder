"""Wound Tracking Module

Track confidence drops as "wounds" that trigger helper spawning.
A wound = significant confidence drop (>15%)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import (
    WOUND_DROP_THRESHOLD,
    WOUND_SPAWN_THRESHOLD,
    MIN_HELPER_SPAWN,
    MAX_HELPER_SPAWN
)
from src.core import emit_receipt


@dataclass
class Wound:
    """A detected wound (confidence drop)."""
    wound_id: str
    decision_id: str
    confidence_before: float
    confidence_after: float
    drop_magnitude: float
    wound_index: int
    detected_at: str
    triggers_spawn: bool


def detect_wound(
    confidence_before: float,
    confidence_after: float,
    threshold: float = WOUND_DROP_THRESHOLD
) -> bool:
    """Detect if a confidence drop constitutes a wound.

    Args:
        confidence_before: Confidence before
        confidence_after: Confidence after
        threshold: Drop threshold (default 0.15 = 15%)

    Returns:
        True if drop exceeds threshold (wound detected)
    """
    drop = confidence_before - confidence_after
    return drop > threshold


def track_wounds(decision_stream: List[dict]) -> int:
    """Count wounds in a decision stream.

    Args:
        decision_stream: List of decisions with confidence

    Returns:
        Number of wounds detected
    """
    wound_count = 0
    prev_confidence = None

    for decision in decision_stream:
        # Extract confidence
        if "full_decision" in decision:
            confidence = decision["full_decision"].get("confidence", 0.5)
        else:
            confidence = decision.get("confidence", 0.5)

        # Check for wound
        if prev_confidence is not None:
            if detect_wound(prev_confidence, confidence):
                wound_count += 1

        prev_confidence = confidence

    return wound_count


def spawn_threshold_check(wound_count: int) -> bool:
    """Check if wound count triggers spawning.

    Args:
        wound_count: Current wound count

    Returns:
        True if wounds >= spawn threshold
    """
    return wound_count >= WOUND_SPAWN_THRESHOLD


def calculate_helper_count(wound_count: int) -> int:
    """Calculate number of helpers to spawn based on wound count.

    Formula: (wound_count // 2) + 1, clamped to [1, 6]

    Args:
        wound_count: Current wound count

    Returns:
        Number of helpers to spawn
    """
    count = (wound_count // 2) + 1
    return max(MIN_HELPER_SPAWN, min(MAX_HELPER_SPAWN, count))


def emit_wound_receipt(
    drop: float,
    decision_id: str,
    confidence_before: float,
    confidence_after: float,
    wound_index: int,
    triggers_spawn: bool = False
) -> dict:
    """Emit a wound detection receipt.

    Args:
        drop: Drop magnitude
        decision_id: Decision where wound occurred
        confidence_before: Confidence before drop
        confidence_after: Confidence after drop
        wound_index: Index in wound sequence
        triggers_spawn: Whether this wound triggers spawning

    Returns:
        Wound receipt
    """
    return emit_receipt("wound", {
        "decision_id": decision_id,
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
        "drop_magnitude": drop,
        "wound_index": wound_index,
        "triggers_spawn": triggers_spawn
    }, silent=True)


class WoundTracker:
    """Stateful wound tracking across decision stream."""

    def __init__(self, window_size: int = 100):
        """Initialize wound tracker.

        Args:
            window_size: Size of rolling window for wound tracking
        """
        self._window_size = window_size
        self._wounds: List[Wound] = []
        self._recent_confidences: List[float] = []
        self._total_wounds = 0
        self._spawn_triggered_count = 0

    def update(self, decision: dict) -> Optional[Wound]:
        """Update tracker with new decision.

        Args:
            decision: New decision

        Returns:
            Wound if detected, None otherwise
        """
        # Extract confidence
        if "full_decision" in decision:
            d = decision["full_decision"]
        else:
            d = decision

        confidence = d.get("confidence", 0.5)
        decision_id = d.get("decision_id", str(uuid.uuid4()))

        # Store confidence
        self._recent_confidences.append(confidence)
        if len(self._recent_confidences) > self._window_size:
            self._recent_confidences.pop(0)

        # Check for wound
        if len(self._recent_confidences) < 2:
            return None

        prev_confidence = self._recent_confidences[-2]

        if detect_wound(prev_confidence, confidence):
            self._total_wounds += 1

            # Check if triggers spawn
            triggers_spawn = spawn_threshold_check(self.get_recent_wound_count())
            if triggers_spawn:
                self._spawn_triggered_count += 1

            wound = Wound(
                wound_id=str(uuid.uuid4()),
                decision_id=decision_id,
                confidence_before=prev_confidence,
                confidence_after=confidence,
                drop_magnitude=prev_confidence - confidence,
                wound_index=self._total_wounds,
                detected_at=datetime.now(timezone.utc).isoformat(),
                triggers_spawn=triggers_spawn
            )

            self._wounds.append(wound)

            # Keep wounds bounded
            if len(self._wounds) > self._window_size:
                self._wounds.pop(0)

            # Emit wound receipt
            emit_wound_receipt(
                drop=wound.drop_magnitude,
                decision_id=decision_id,
                confidence_before=prev_confidence,
                confidence_after=confidence,
                wound_index=self._total_wounds,
                triggers_spawn=triggers_spawn
            )

            return wound

        return None

    def get_recent_wound_count(self) -> int:
        """Get wound count in recent window.

        Returns:
            Number of wounds in window
        """
        return len(self._wounds)

    def get_total_wound_count(self) -> int:
        """Get total wound count since tracker creation.

        Returns:
            Total wound count
        """
        return self._total_wounds

    def should_spawn_helpers(self) -> bool:
        """Check if helpers should be spawned based on wounds.

        Returns:
            True if spawn threshold reached
        """
        return spawn_threshold_check(self.get_recent_wound_count())

    def get_helper_count_to_spawn(self) -> int:
        """Get number of helpers to spawn.

        Returns:
            Helper count (0 if threshold not reached)
        """
        if not self.should_spawn_helpers():
            return 0
        return calculate_helper_count(self.get_recent_wound_count())

    def get_wound_trend(self) -> str:
        """Analyze wound trend.

        Returns:
            "increasing", "stable", or "decreasing"
        """
        if len(self._wounds) < 10:
            return "stable"

        # Compare first half to second half
        mid = len(self._wounds) // 2
        first_half = self._wounds[:mid]
        second_half = self._wounds[mid:]

        # Use wound severity (drop magnitude)
        first_avg = sum(w.drop_magnitude for w in first_half) / len(first_half)
        second_avg = sum(w.drop_magnitude for w in second_half) / len(second_half)

        diff = second_avg - first_avg

        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"

    def get_stats(self) -> dict:
        """Get wound tracking statistics.

        Returns:
            Stats dict
        """
        if not self._wounds:
            avg_drop = 0.0
            max_drop = 0.0
        else:
            avg_drop = sum(w.drop_magnitude for w in self._wounds) / len(self._wounds)
            max_drop = max(w.drop_magnitude for w in self._wounds)

        return {
            "total_wounds": self._total_wounds,
            "recent_wounds": len(self._wounds),
            "spawn_triggers": self._spawn_triggered_count,
            "average_drop": avg_drop,
            "max_drop": max_drop,
            "trend": self.get_wound_trend(),
            "should_spawn": self.should_spawn_helpers(),
            "helpers_to_spawn": self.get_helper_count_to_spawn()
        }

    def reset(self):
        """Reset wound tracker."""
        self._wounds = []
        self._recent_confidences = []
        self._total_wounds = 0
        self._spawn_triggered_count = 0


# Global wound tracker instance
_wound_tracker: Optional[WoundTracker] = None


def get_wound_tracker() -> WoundTracker:
    """Get the global wound tracker.

    Returns:
        Global WoundTracker instance
    """
    global _wound_tracker
    if _wound_tracker is None:
        _wound_tracker = WoundTracker()
    return _wound_tracker


def reset_wound_tracker():
    """Reset the global wound tracker."""
    global _wound_tracker
    _wound_tracker = WoundTracker()
