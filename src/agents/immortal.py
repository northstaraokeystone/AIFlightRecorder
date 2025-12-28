"""Immortal Agent Base Class

These are not agents IN the system. They ARE the system experiencing itself.

Key insight from QED:
"Killing HUNTER removes the system's ability to detect.
 Killing SHEPHERD removes the system's ability to heal.
 They're not features to toggle. They're what 'being a system' means."
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import IMMORTAL_AGENTS
from config.features import is_feature_enabled


@dataclass
class ImmortalAgent(ABC):
    """Base class for immortal agents.

    Immortal agents:
    - Cannot be spawned (they exist from genesis)
    - Cannot be pruned (they are what 'being a system' means)
    - Have a self_id to ignore their own receipts
    - Run on every decision cycle
    """
    self_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_scan_at: Optional[str] = None
    scans_completed: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__.upper()

    @abstractmethod
    def scan(self, decisions: List[dict]) -> List[dict]:
        """Scan decisions for anomalies or issues.

        Must be implemented by subclasses.

        Args:
            decisions: List of decisions to scan

        Returns:
            List of detected issues/anomalies
        """
        pass

    def should_ignore(self, receipt: dict) -> bool:
        """Check if this agent should ignore a receipt.

        Agents ignore their own receipts to avoid self-flagging.

        Args:
            receipt: Receipt to check

        Returns:
            True if should ignore (self-generated)
        """
        agent_id = receipt.get("agent_id", "")
        return agent_id == self.self_id

    def is_enabled(self) -> bool:
        """Check if this agent is enabled.

        Returns:
            True if enabled in feature flags
        """
        feature_name = f"FEATURE_{self.name}_ENABLED"
        return is_feature_enabled(feature_name)

    def update_scan_time(self):
        """Update the last scan timestamp."""
        self.last_scan_at = datetime.now(timezone.utc).isoformat()
        self.scans_completed += 1

    def get_status(self) -> dict:
        """Get agent status.

        Returns:
            Status dict
        """
        return {
            "agent_id": self.self_id,
            "name": self.name,
            "type": "immortal",
            "enabled": self.is_enabled(),
            "created_at": self.created_at,
            "last_scan_at": self.last_scan_at,
            "scans_completed": self.scans_completed,
            "can_prune": False,
            "can_spawn": False
        }


# Global immortal agent instances
_hunter: Optional["ImmortalAgent"] = None
_shepherd: Optional["ImmortalAgent"] = None


def get_hunter() -> "ImmortalAgent":
    """Get the global HUNTER instance.

    HUNTER = Proprioception (system's capacity to feel anomalies)

    Returns:
        Hunter instance
    """
    global _hunter
    if _hunter is None:
        from .hunter import Hunter
        _hunter = Hunter()
    return _hunter


def get_shepherd() -> "ImmortalAgent":
    """Get the global SHEPHERD instance.

    SHEPHERD = Homeostasis (system's capacity to heal)

    Returns:
        Shepherd instance
    """
    global _shepherd
    if _shepherd is None:
        from .shepherd import Shepherd
        _shepherd = Shepherd()
    return _shepherd


def reset_immortals():
    """Reset immortal agents (for testing)."""
    global _hunter, _shepherd
    from .hunter import Hunter
    from .shepherd import Shepherd
    _hunter = Hunter()
    _shepherd = Shepherd()


def run_immortal_cycle(decisions: List[dict]) -> dict:
    """Run HUNTER and SHEPHERD on decision cycle.

    This is the core perception-response loop.

    Args:
        decisions: Recent decisions

    Returns:
        Cycle result with anomalies and remediations
    """
    hunter = get_hunter()
    shepherd = get_shepherd()

    result = {
        "hunter_enabled": hunter.is_enabled(),
        "shepherd_enabled": shepherd.is_enabled(),
        "anomalies": [],
        "remediations": []
    }

    # HUNTER scans for anomalies
    if hunter.is_enabled():
        anomalies = hunter.scan(decisions)
        result["anomalies"] = anomalies
        hunter.update_scan_time()

    # SHEPHERD proposes remediations for anomalies
    if shepherd.is_enabled() and result["anomalies"]:
        for anomaly in result["anomalies"]:
            remediation = shepherd.scan([anomaly])
            if remediation:
                result["remediations"].extend(remediation)
        shepherd.update_scan_time()

    return result
