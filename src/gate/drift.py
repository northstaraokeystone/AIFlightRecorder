"""Context Drift Measurement

Measure how much context has changed since a decision started.
"""

import json
import gzip
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class DriftScore:
    """Result of drift measurement."""
    score: float              # 0-1, higher = more drift
    initial_hash: str
    current_hash: str
    changed_fields: list[str]
    is_significant: bool      # True if drift > 0.3


def measure_drift(
    initial_context: dict,
    current_context: dict,
    significant_threshold: float = 0.3
) -> DriftScore:
    """Measure context drift since decision started.

    Uses compression-based distance for context comparison.

    Args:
        initial_context: Context when decision was made
        current_context: Current context
        significant_threshold: Threshold for significant drift

    Returns:
        DriftScore with drift measurement
    """
    # Serialize contexts
    initial_bytes = json.dumps(initial_context, sort_keys=True).encode()
    current_bytes = json.dumps(current_context, sort_keys=True).encode()

    # Simple hash comparison for exact match
    import hashlib
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()[:16]
    current_hash = hashlib.sha256(current_bytes).hexdigest()[:16]

    if initial_hash == current_hash:
        return DriftScore(
            score=0.0,
            initial_hash=initial_hash,
            current_hash=current_hash,
            changed_fields=[],
            is_significant=False
        )

    # Calculate NCD-based drift
    combined = initial_bytes + current_bytes

    c_initial = len(gzip.compress(initial_bytes, compresslevel=6))
    c_current = len(gzip.compress(current_bytes, compresslevel=6))
    c_combined = len(gzip.compress(combined, compresslevel=6))

    min_c = min(c_initial, c_current)
    max_c = max(c_initial, c_current)

    if max_c == 0:
        ncd = 0.0
    else:
        ncd = (c_combined - min_c) / max_c

    # Normalize to 0-1 range
    # NCD is typically between 0 and ~1.1
    drift_score = min(1.0, max(0.0, ncd))

    # Find changed fields
    changed_fields = find_changed_fields(initial_context, current_context)

    return DriftScore(
        score=drift_score,
        initial_hash=initial_hash,
        current_hash=current_hash,
        changed_fields=changed_fields,
        is_significant=drift_score > significant_threshold
    )


def find_changed_fields(
    initial: dict,
    current: dict,
    prefix: str = ""
) -> list[str]:
    """Find which fields changed between contexts.

    Args:
        initial: Initial context
        current: Current context
        prefix: Prefix for nested fields

    Returns:
        List of changed field paths
    """
    changed = []

    # All keys in either dict
    all_keys = set(initial.keys()) | set(current.keys())

    for key in all_keys:
        field_path = f"{prefix}.{key}" if prefix else key

        if key not in initial:
            changed.append(f"+{field_path}")  # Added
        elif key not in current:
            changed.append(f"-{field_path}")  # Removed
        elif initial[key] != current[key]:
            if isinstance(initial[key], dict) and isinstance(current[key], dict):
                # Recurse for nested dicts
                changed.extend(find_changed_fields(initial[key], current[key], field_path))
            else:
                changed.append(f"~{field_path}")  # Modified

    return changed


def context_from_decision(decision: dict) -> dict:
    """Extract context from a decision for drift comparison.

    Args:
        decision: Decision dict

    Returns:
        Context dict suitable for drift measurement
    """
    if "full_decision" in decision:
        d = decision["full_decision"]
    else:
        d = decision

    context = {
        "perception": d.get("perception", {}),
        "telemetry": d.get("telemetry_snapshot", {}),
        "action_type": d.get("action", {}).get("type", "UNKNOWN"),
        "alternatives_count": len(d.get("alternative_actions_considered", []))
    }

    return context


def drift_trend(drift_history: list[float], window: int = 5) -> str:
    """Analyze drift trend over recent measurements.

    Args:
        drift_history: Recent drift scores
        window: Analysis window

    Returns:
        "increasing", "stable", or "decreasing"
    """
    if len(drift_history) < window:
        return "stable"

    recent = drift_history[-window:]
    mid = window // 2

    first_avg = sum(recent[:mid]) / mid if mid > 0 else 0
    second_avg = sum(recent[mid:]) / (window - mid) if (window - mid) > 0 else 0

    diff = second_avg - first_avg

    if diff > 0.1:
        return "increasing"
    elif diff < -0.1:
        return "decreasing"
    else:
        return "stable"
