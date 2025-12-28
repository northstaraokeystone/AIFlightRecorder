"""Provenance Tracking - Model and Policy Versioning (v2.1)

Captures and tracks provenance of decisions including:
- Model version and hash
- Policy version and hash
- Training data version
- Configuration version
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..core import dual_hash, emit_receipt


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a decision."""
    decision_id: str
    model_version: str
    model_hash: str
    policy_version: str
    policy_hash: str
    config_version: str
    config_hash: str
    training_data_version: Optional[str]
    training_data_hash: Optional[str]
    captured_at: str


# Version tracking (would be loaded from environment/config in production)
_versions = {
    "model": {
        "version": os.environ.get("MODEL_VERSION", "v1.0.0"),
        "hash": os.environ.get("MODEL_HASH", "unknown")
    },
    "policy": {
        "version": os.environ.get("POLICY_VERSION", "v1.0.0"),
        "hash": os.environ.get("POLICY_HASH", "unknown")
    },
    "config": {
        "version": os.environ.get("CONFIG_VERSION", "v1.0.0"),
        "hash": os.environ.get("CONFIG_HASH", "unknown")
    },
    "training_data": {
        "version": os.environ.get("TRAINING_DATA_VERSION", None),
        "hash": os.environ.get("TRAINING_DATA_HASH", None)
    }
}


def set_version(component: str, version: str, hash_value: str):
    """Set version information for a component.

    Args:
        component: Component name (model, policy, config, training_data)
        version: Version string
        hash_value: Hash of the component
    """
    if component in _versions:
        _versions[component] = {
            "version": version,
            "hash": hash_value
        }


def get_model_version() -> dict:
    """Get current model version.

    Returns:
        Dict with version and hash
    """
    return _versions["model"].copy()


def get_policy_version() -> dict:
    """Get current policy version.

    Returns:
        Dict with version and hash
    """
    return _versions["policy"].copy()


def get_config_version() -> dict:
    """Get current config version.

    Returns:
        Dict with version and hash
    """
    return _versions["config"].copy()


def get_training_data_version() -> dict:
    """Get current training data version.

    Returns:
        Dict with version and hash
    """
    return _versions["training_data"].copy()


def capture_provenance(decision_id: str,
                       additional_context: Optional[dict] = None) -> ProvenanceRecord:
    """Capture complete provenance for a decision.

    Args:
        decision_id: Decision identifier
        additional_context: Additional provenance context

    Returns:
        ProvenanceRecord
    """
    model = get_model_version()
    policy = get_policy_version()
    config = get_config_version()
    training = get_training_data_version()

    record = ProvenanceRecord(
        decision_id=decision_id,
        model_version=model["version"],
        model_hash=model["hash"],
        policy_version=policy["version"],
        policy_hash=policy["hash"],
        config_version=config["version"],
        config_hash=config["hash"],
        training_data_version=training["version"],
        training_data_hash=training["hash"],
        captured_at=datetime.now(timezone.utc).isoformat()
    )

    # Emit receipt
    emit_provenance_receipt(record, additional_context)

    return record


def validate_provenance(record: ProvenanceRecord) -> tuple[bool, list]:
    """Validate that provenance record is complete.

    Args:
        record: Provenance record to validate

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = []

    if not record.model_version:
        missing.append("model_version")
    if not record.model_hash or record.model_hash == "unknown":
        missing.append("model_hash")
    if not record.policy_version:
        missing.append("policy_version")
    if not record.policy_hash or record.policy_hash == "unknown":
        missing.append("policy_hash")

    return len(missing) == 0, missing


def compute_provenance_hash(record: ProvenanceRecord) -> str:
    """Compute hash of provenance record for verification.

    Args:
        record: Provenance record

    Returns:
        Hash of record
    """
    data = {
        "model": f"{record.model_version}:{record.model_hash}",
        "policy": f"{record.policy_version}:{record.policy_hash}",
        "config": f"{record.config_version}:{record.config_hash}",
        "training": f"{record.training_data_version}:{record.training_data_hash}"
    }
    return dual_hash(json.dumps(data, sort_keys=True))


def emit_provenance_receipt(record: ProvenanceRecord,
                            context: Optional[dict] = None) -> dict:
    """Emit provenance receipt.

    Args:
        record: Provenance record
        context: Additional context

    Returns:
        Receipt dict
    """
    data = {
        "decision_id": record.decision_id,
        "model_version": record.model_version,
        "model_hash": record.model_hash,
        "policy_version": record.policy_version,
        "policy_hash": record.policy_hash,
        "config_version": record.config_version,
        "config_hash": record.config_hash,
        "training_data_version": record.training_data_version,
        "training_data_hash": record.training_data_hash,
        "provenance_hash": compute_provenance_hash(record)
    }

    if context:
        data["context"] = context

    return emit_receipt("provenance", data, silent=True)


def compare_provenance(record1: ProvenanceRecord,
                       record2: ProvenanceRecord) -> dict:
    """Compare two provenance records.

    Args:
        record1: First record
        record2: Second record

    Returns:
        Dict of differences
    """
    differences = {}

    if record1.model_version != record2.model_version:
        differences["model_version"] = {
            "from": record1.model_version,
            "to": record2.model_version
        }

    if record1.model_hash != record2.model_hash:
        differences["model_hash"] = {
            "from": record1.model_hash[:16] + "...",
            "to": record2.model_hash[:16] + "..."
        }

    if record1.policy_version != record2.policy_version:
        differences["policy_version"] = {
            "from": record1.policy_version,
            "to": record2.policy_version
        }

    if record1.policy_hash != record2.policy_hash:
        differences["policy_hash"] = {
            "from": record1.policy_hash[:16] + "...",
            "to": record2.policy_hash[:16] + "..."
        }

    return differences


class ProvenanceTracker:
    """Stateful provenance tracker for decision history."""

    def __init__(self):
        self._history: list[ProvenanceRecord] = []
        self._current_session_start = datetime.now(timezone.utc).isoformat()

    def capture(self, decision_id: str,
                context: Optional[dict] = None) -> ProvenanceRecord:
        """Capture and track provenance.

        Args:
            decision_id: Decision ID
            context: Additional context

        Returns:
            ProvenanceRecord
        """
        record = capture_provenance(decision_id, context)
        self._history.append(record)
        return record

    def get_history(self) -> list[ProvenanceRecord]:
        """Get provenance history.

        Returns:
            List of provenance records
        """
        return self._history.copy()

    def detect_drift(self) -> list[dict]:
        """Detect provenance drift in history.

        Returns:
            List of drift events
        """
        if len(self._history) < 2:
            return []

        drifts = []
        for i in range(1, len(self._history)):
            diff = compare_provenance(self._history[i-1], self._history[i])
            if diff:
                drifts.append({
                    "from_decision": self._history[i-1].decision_id,
                    "to_decision": self._history[i].decision_id,
                    "changes": diff
                })

        return drifts
