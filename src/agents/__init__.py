"""Immortal Agents Module - v2.0

HUNTER and SHEPHERD are not agents IN the system.
They ARE the system experiencing itself.

HUNTER = Proprioception (system's capacity to feel anomalies)
SHEPHERD = Homeostasis (system's capacity to heal)

They cannot be spawned or pruned - they are immortal (germline).
"""

from .immortal import ImmortalAgent, get_hunter, get_shepherd
from .hunter import Hunter, scan_for_anomalies
from .shepherd import Shepherd, propose_remediation

__all__ = [
    "ImmortalAgent",
    "get_hunter",
    "get_shepherd",
    "Hunter",
    "scan_for_anomalies",
    "Shepherd",
    "propose_remediation"
]
