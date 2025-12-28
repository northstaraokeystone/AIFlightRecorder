"""Pattern Store

Store and retrieve graduated patterns.
Patterns are solutions that have been proven effective.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
from threading import Lock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class StoredPattern:
    """A stored pattern."""
    pattern_id: str
    pattern_data: dict
    effectiveness: float
    source_agent_id: str
    created_at: str
    usage_count: int = 0
    last_used: Optional[str] = None
    domain: str = "default"


class PatternStore:
    """Thread-safe pattern store."""

    def __init__(self, storage_path: Optional[Path] = None):
        self._patterns: Dict[str, StoredPattern] = {}
        self._by_domain: Dict[str, List[str]] = {}
        self._lock = Lock()
        self._storage_path = storage_path or Path("patterns.jsonl")
        self._load()

    def _load(self):
        """Load patterns from storage file."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        pattern = StoredPattern(
                            pattern_id=data["pattern_id"],
                            pattern_data=data["pattern_data"],
                            effectiveness=data["effectiveness"],
                            source_agent_id=data["source_agent_id"],
                            created_at=data["created_at"],
                            usage_count=data.get("usage_count", 0),
                            last_used=data.get("last_used"),
                            domain=data.get("domain", "default")
                        )
                        self._patterns[pattern.pattern_id] = pattern

                        if pattern.domain not in self._by_domain:
                            self._by_domain[pattern.domain] = []
                        self._by_domain[pattern.domain].append(pattern.pattern_id)
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupt file, start fresh

    def _save(self):
        """Save patterns to storage file."""
        with open(self._storage_path, 'w') as f:
            for pattern in self._patterns.values():
                data = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_data": pattern.pattern_data,
                    "effectiveness": pattern.effectiveness,
                    "source_agent_id": pattern.source_agent_id,
                    "created_at": pattern.created_at,
                    "usage_count": pattern.usage_count,
                    "last_used": pattern.last_used,
                    "domain": pattern.domain
                }
                f.write(json.dumps(data) + "\n")

    def store(
        self,
        pattern: dict,
        effectiveness: float,
        source_agent_id: str,
        domain: str = "default"
    ) -> str:
        """Store a new pattern.

        Args:
            pattern: Pattern data
            effectiveness: Effectiveness score
            source_agent_id: Agent that created this pattern
            domain: Domain classification

        Returns:
            Pattern ID
        """
        with self._lock:
            pattern_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            stored = StoredPattern(
                pattern_id=pattern_id,
                pattern_data=pattern,
                effectiveness=effectiveness,
                source_agent_id=source_agent_id,
                created_at=now,
                domain=domain
            )

            self._patterns[pattern_id] = stored

            if domain not in self._by_domain:
                self._by_domain[domain] = []
            self._by_domain[domain].append(pattern_id)

            self._save()
            return pattern_id

    def get(self, pattern_id: str) -> Optional[StoredPattern]:
        """Get a pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            StoredPattern or None
        """
        with self._lock:
            return self._patterns.get(pattern_id)

    def match(
        self,
        criteria: dict,
        domain: Optional[str] = None,
        min_effectiveness: float = 0.0
    ) -> List[StoredPattern]:
        """Find patterns matching criteria.

        Args:
            criteria: Match criteria (subset check)
            domain: Optional domain filter
            min_effectiveness: Minimum effectiveness threshold

        Returns:
            List of matching patterns
        """
        with self._lock:
            matches = []

            patterns_to_check = self._patterns.values()
            if domain:
                pattern_ids = self._by_domain.get(domain, [])
                patterns_to_check = [
                    self._patterns[pid]
                    for pid in pattern_ids
                    if pid in self._patterns
                ]

            for pattern in patterns_to_check:
                if pattern.effectiveness < min_effectiveness:
                    continue

                # Check if criteria is subset of pattern data
                if self._matches_criteria(pattern.pattern_data, criteria):
                    matches.append(pattern)

            return matches

    def _matches_criteria(self, pattern_data: dict, criteria: dict) -> bool:
        """Check if pattern matches criteria.

        Args:
            pattern_data: Pattern to check
            criteria: Criteria to match

        Returns:
            True if matches
        """
        for key, value in criteria.items():
            if key not in pattern_data:
                return False
            if pattern_data[key] != value:
                return False
        return True

    def record_usage(self, pattern_id: str):
        """Record pattern usage.

        Args:
            pattern_id: Pattern ID
        """
        with self._lock:
            if pattern_id in self._patterns:
                pattern = self._patterns[pattern_id]
                pattern.usage_count += 1
                pattern.last_used = datetime.now(timezone.utc).isoformat()
                self._save()

    def get_by_domain(self, domain: str) -> List[StoredPattern]:
        """Get all patterns for a domain.

        Args:
            domain: Domain name

        Returns:
            List of patterns
        """
        with self._lock:
            pattern_ids = self._by_domain.get(domain, [])
            return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    def get_most_effective(self, n: int = 10) -> List[StoredPattern]:
        """Get the most effective patterns.

        Args:
            n: Number to return

        Returns:
            List of patterns sorted by effectiveness
        """
        with self._lock:
            sorted_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.effectiveness,
                reverse=True
            )
            return sorted_patterns[:n]

    def get_most_used(self, n: int = 10) -> List[StoredPattern]:
        """Get the most used patterns.

        Args:
            n: Number to return

        Returns:
            List of patterns sorted by usage
        """
        with self._lock:
            sorted_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.usage_count,
                reverse=True
            )
            return sorted_patterns[:n]

    def count(self) -> int:
        """Get total pattern count.

        Returns:
            Number of patterns
        """
        with self._lock:
            return len(self._patterns)

    def total_usage(self) -> int:
        """Get total usage across all patterns.

        Returns:
            Total usage count
        """
        with self._lock:
            return sum(p.usage_count for p in self._patterns.values())

    def clear(self):
        """Clear all patterns (for testing)."""
        with self._lock:
            self._patterns.clear()
            self._by_domain.clear()
            if self._storage_path.exists():
                self._storage_path.unlink()


# Global pattern store instance
_global_store = None


def get_pattern_store() -> PatternStore:
    """Get the global pattern store.

    Returns:
        Global PatternStore instance
    """
    global _global_store
    if _global_store is None:
        _global_store = PatternStore()
    return _global_store


def reset_pattern_store():
    """Reset the global pattern store (for testing)."""
    global _global_store
    _global_store = PatternStore()
