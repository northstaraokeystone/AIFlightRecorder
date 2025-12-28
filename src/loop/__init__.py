"""Loop Module - v2.0

Wound tracking and spawn trigger logic.
Wounds are confidence drops that trigger helper spawning.
"""

from .wounds import (
    detect_wound,
    track_wounds,
    spawn_threshold_check,
    emit_wound_receipt,
    WoundTracker,
    get_wound_tracker
)

__all__ = [
    "detect_wound",
    "track_wounds",
    "spawn_threshold_check",
    "emit_wound_receipt",
    "WoundTracker",
    "get_wound_tracker"
]
