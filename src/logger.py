"""Decision Logger - The Flight Recorder

Captures complete decision context at execution time with hash chaining.
Provides indexed storage and retrieval for forensic analysis.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .core import dual_hash, emit_receipt, GENESIS_HASH, StopRule

# Database path
DB_PATH = Path("receipts.db")


@dataclass
class LogEntry:
    """A logged decision with chain linkage."""
    decision_id: str
    decision_hash: str
    prev_hash: str
    merkle_position: int
    timestamp: str
    decision_data: dict


class DecisionLogger:
    """Manages decision chain with SQLite backing and JSONL ledger."""

    def __init__(self, db_path: Optional[Path] = None, ledger_path: Optional[Path] = None):
        """Initialize the decision logger.

        Args:
            db_path: Path to SQLite database
            ledger_path: Path to JSONL ledger file
        """
        self.db_path = db_path or DB_PATH
        self.ledger_path = ledger_path or Path("receipts.jsonl")
        self._chain_tip_hash = GENESIS_HASH
        self._position = 0
        self._init_db()
        self._load_chain_state()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE NOT NULL,
                decision_hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                merkle_position INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                decision_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_decision_hash ON decisions(decision_hash)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_merkle_position ON decisions(merkle_position)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp)
        ''')

        conn.commit()
        conn.close()

    def _load_chain_state(self):
        """Load the current chain tip from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT decision_hash, merkle_position FROM decisions
            ORDER BY merkle_position DESC LIMIT 1
        ''')

        row = cursor.fetchone()
        if row:
            self._chain_tip_hash = row[0]
            self._position = row[1] + 1

        conn.close()

    def log_decision(self, decision: dict, prev_hash: Optional[str] = None) -> dict:
        """Add decision to chain and emit log receipt.

        Args:
            decision: The decision dict to log
            prev_hash: Previous hash in chain (uses current tip if None)

        Returns:
            Log receipt dict

        Raises:
            StopRule: If logging fails critically
        """
        start_time = time.perf_counter()

        try:
            # Get or use provided prev_hash
            prev = prev_hash or self._chain_tip_hash

            # Compute decision hash
            decision_hash = dual_hash(json.dumps(decision, sort_keys=True))

            # Get decision ID
            decision_id = decision.get("decision_id") or decision.get("full_decision", {}).get("decision_id", "unknown")

            # Create timestamp
            timestamp = datetime.now(timezone.utc).isoformat()

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO decisions (decision_id, decision_hash, prev_hash,
                                       merkle_position, timestamp, decision_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (decision_id, decision_hash, prev,
                  self._position, timestamp, json.dumps(decision, sort_keys=True)))

            conn.commit()
            conn.close()

            # Update chain state
            self._chain_tip_hash = decision_hash
            current_position = self._position
            self._position += 1

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Emit log receipt
            log_receipt = emit_receipt("decision_log", {
                "decision_id": decision_id,
                "decision_hash": decision_hash,
                "prev_hash": prev,
                "merkle_position": current_position,
                "local_tree_root": self._chain_tip_hash,  # Simplified for now
                "log_latency_ms": latency_ms
            }, silent=True)

            # Check SLO
            if latency_ms > 100:
                emit_receipt("anomaly", {
                    "metric": "decision_log_latency",
                    "baseline": 100,
                    "actual": latency_ms,
                    "delta": latency_ms - 100,
                    "classification": "degradation",
                    "action": "alert"
                }, silent=True)

            return log_receipt

        except Exception as e:
            emit_receipt("anomaly", {
                "metric": "decision_log",
                "baseline": 0,
                "actual": -1,
                "delta": -1,
                "classification": "violation",
                "action": "halt",
                "error": str(e)
            })
            raise StopRule(f"Decision logging failed: {e}", "decision_log", "halt")

    def get_decision_chain(self, start: int = 0, end: Optional[int] = None) -> list[dict]:
        """Retrieve decision range by merkle position.

        Args:
            start: Starting position (inclusive)
            end: Ending position (exclusive), None for all remaining

        Returns:
            List of decision dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if end is None:
            cursor.execute('''
                SELECT decision_data FROM decisions
                WHERE merkle_position >= ?
                ORDER BY merkle_position ASC
            ''', (start,))
        else:
            cursor.execute('''
                SELECT decision_data FROM decisions
                WHERE merkle_position >= ? AND merkle_position < ?
                ORDER BY merkle_position ASC
            ''', (start, end))

        rows = cursor.fetchall()
        conn.close()

        return [json.loads(row[0]) for row in rows]

    def get_decision_by_id(self, decision_id: str) -> Optional[dict]:
        """Get a specific decision by ID.

        Args:
            decision_id: The decision UUID

        Returns:
            Decision dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT decision_data, decision_hash, prev_hash, merkle_position, timestamp
            FROM decisions WHERE decision_id = ?
        ''', (decision_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "decision": json.loads(row[0]),
                "decision_hash": row[1],
                "prev_hash": row[2],
                "merkle_position": row[3],
                "timestamp": row[4]
            }
        return None

    def get_latest_hash(self) -> str:
        """Get current chain tip hash.

        Returns:
            The hash of the most recent decision
        """
        return self._chain_tip_hash

    def get_chain_length(self) -> int:
        """Get total number of logged decisions.

        Returns:
            Number of decisions in chain
        """
        return self._position

    def export_for_sync(self, since_hash: Optional[str] = None) -> dict:
        """Package decisions for cloud sync.

        Args:
            since_hash: Export decisions after this hash (None for all)

        Returns:
            Sync package dict
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if since_hash and since_hash != GENESIS_HASH:
            # Find position of since_hash
            cursor.execute('''
                SELECT merkle_position FROM decisions WHERE decision_hash = ?
            ''', (since_hash,))
            row = cursor.fetchone()
            start_pos = row[0] + 1 if row else 0
        else:
            start_pos = 0

        cursor.execute('''
            SELECT decision_id, decision_hash, prev_hash, merkle_position,
                   timestamp, decision_data
            FROM decisions WHERE merkle_position >= ?
            ORDER BY merkle_position ASC
        ''', (start_pos,))

        rows = cursor.fetchall()
        conn.close()

        decisions = []
        for row in rows:
            decisions.append({
                "decision_id": row[0],
                "decision_hash": row[1],
                "prev_hash": row[2],
                "merkle_position": row[3],
                "timestamp": row[4],
                "decision_data": json.loads(row[5])
            })

        return {
            "since_hash": since_hash or GENESIS_HASH,
            "current_hash": self._chain_tip_hash,
            "chain_length": self._position,
            "decisions": decisions,
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }

    def verify_chain_integrity(self) -> tuple[bool, list[dict]]:
        """Verify the entire decision chain integrity.

        Returns:
            Tuple of (is_valid, list of violations)
        """
        decisions = self.get_decision_chain()
        violations = []

        if not decisions:
            return True, []

        # Check first decision links to genesis
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prev_hash FROM decisions WHERE merkle_position = 0
        ''')
        row = cursor.fetchone()
        conn.close()

        if row and row[0] != GENESIS_HASH:
            violations.append({
                "position": 0,
                "type": "genesis_link_broken",
                "expected": GENESIS_HASH,
                "actual": row[0]
            })

        # Verify each decision's hash
        prev_hash = GENESIS_HASH
        for i, decision in enumerate(decisions):
            computed_hash = dual_hash(json.dumps(decision, sort_keys=True))

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT decision_hash, prev_hash FROM decisions
                WHERE merkle_position = ?
            ''', (i,))
            row = cursor.fetchone()
            conn.close()

            if row:
                stored_hash, stored_prev = row

                # Check hash matches
                if computed_hash != stored_hash:
                    violations.append({
                        "position": i,
                        "type": "hash_mismatch",
                        "expected": stored_hash,
                        "computed": computed_hash
                    })

                # Check chain linkage
                if stored_prev != prev_hash:
                    violations.append({
                        "position": i,
                        "type": "chain_break",
                        "expected_prev": prev_hash,
                        "stored_prev": stored_prev
                    })

                prev_hash = stored_hash

        return len(violations) == 0, violations

    def clear(self):
        """Clear all decisions (for testing)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM decisions')
        conn.commit()
        conn.close()

        self._chain_tip_hash = GENESIS_HASH
        self._position = 0


# Module-level logger instance
_default_logger: Optional[DecisionLogger] = None


def get_logger() -> DecisionLogger:
    """Get or create the default decision logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = DecisionLogger()
    return _default_logger


def log_decision(decision: dict, prev_hash: Optional[str] = None) -> dict:
    """Log a decision using the default logger."""
    return get_logger().log_decision(decision, prev_hash)


def get_decision_chain(start: int = 0, end: Optional[int] = None) -> list[dict]:
    """Get decisions from the default logger."""
    return get_logger().get_decision_chain(start, end)


def get_latest_hash() -> str:
    """Get latest hash from the default logger."""
    return get_logger().get_latest_hash()


def export_for_sync(since_hash: Optional[str] = None) -> dict:
    """Export for sync from the default logger."""
    return get_logger().export_for_sync(since_hash)
