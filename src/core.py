"""Core Foundation Functions - CLAUDEME §8 Compliance

Every other module imports from here. Foundation for:
- Dual hashing (SHA256 + BLAKE3)
- Receipt emission
- Merkle tree operations
- StopRule exception handling
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

# Try to import blake3, fall back to hashlib if not available
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Constants
GENESIS_HASH = "0" * 64 + ":" + "0" * 64
RECEIPTS_FILE = Path(os.environ.get("RECEIPTS_FILE", "receipts.jsonl"))
TENANT_ID = os.environ.get("TENANT_ID", "edge-device-001")

# Global receipt counter for ordering
_receipt_counter = 0


class StopRule(Exception):
    """Raised when stoprule triggers. Never catch silently.

    StopRules indicate critical failures that require immediate attention.
    They emit an anomaly receipt before raising.
    """
    def __init__(self, message: str, metric: str = "unknown", action: str = "halt"):
        self.message = message
        self.metric = metric
        self.action = action
        super().__init__(message)


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 dual hash. ALWAYS use this, never single hash.

    Args:
        data: Input bytes or string to hash

    Returns:
        String in format "sha256_hex:blake3_hex"

    Example:
        >>> dual_hash(b"test")
        '9f86d08....:d4735e3a...'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')

    sha256_hash = hashlib.sha256(data).hexdigest()

    if HAS_BLAKE3:
        blake3_hash = blake3.blake3(data).hexdigest()
    else:
        # Fallback: use SHA256 as both (for testing without blake3)
        blake3_hash = hashlib.sha256(b"blake3:" + data).hexdigest()

    return f"{sha256_hash}:{blake3_hash}"


def emit_receipt(receipt_type: str, data: dict,
                 tenant_id: Optional[str] = None,
                 to_file: bool = True,
                 silent: bool = False) -> dict:
    """Emit a CLAUDEME-compliant receipt. Every function calls this.

    Args:
        receipt_type: Type of receipt (decision, anchor, anomaly, etc.)
        data: Receipt payload data
        tenant_id: Override default tenant ID
        to_file: Whether to append to receipts.jsonl
        silent: Whether to suppress stdout printing

    Returns:
        Complete receipt dict with ts, tenant_id, payload_hash
    """
    global _receipt_counter
    _receipt_counter += 1

    ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    tid = tenant_id or data.get("tenant_id", TENANT_ID)

    # Build receipt
    receipt = {
        "receipt_type": receipt_type,
        "ts": ts,
        "tenant_id": tid,
        "sequence": _receipt_counter,
        **data
    }

    # Add payload hash (hash of the data without the hash itself)
    data_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
    receipt["payload_hash"] = dual_hash(json.dumps(data_for_hash, sort_keys=True))

    # Output
    receipt_json = json.dumps(receipt, sort_keys=True)

    if not silent:
        print(receipt_json, flush=True)

    if to_file:
        try:
            with open(RECEIPTS_FILE, "a") as f:
                f.write(receipt_json + "\n")
        except IOError:
            pass  # Edge case: file system issues

    return receipt


def merkle_root(items: list[Any]) -> str:
    """Compute Merkle root of items using BLAKE3.

    Args:
        items: List of items (will be JSON-serialized if not bytes)

    Returns:
        Merkle root as dual-hash string

    Complexity: O(n) time, O(n) space for building tree
    """
    if not items:
        return dual_hash(b"empty_tree")

    # Convert items to hashes
    hashes = []
    for item in items:
        if isinstance(item, bytes):
            hashes.append(dual_hash(item))
        elif isinstance(item, str):
            hashes.append(dual_hash(item.encode()))
        else:
            hashes.append(dual_hash(json.dumps(item, sort_keys=True)))

    # Build tree bottom-up
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last for odd count

        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            next_level.append(dual_hash(combined))
        hashes = next_level

    return hashes[0]


def merkle_proof(items: list[Any], index: int) -> list[tuple[str, str]]:
    """Generate O(log N) inclusion proof for item at index.

    Args:
        items: List of all items in the tree
        index: Index of item to prove inclusion for

    Returns:
        List of (hash, direction) tuples for proof verification
        direction is 'L' or 'R' indicating which side the sibling is on

    Raises:
        IndexError: If index is out of range
    """
    if not items:
        return []

    if index < 0 or index >= len(items):
        raise IndexError(f"Index {index} out of range for {len(items)} items")

    # Convert items to hashes
    hashes = []
    for item in items:
        if isinstance(item, bytes):
            hashes.append(dual_hash(item))
        elif isinstance(item, str):
            hashes.append(dual_hash(item.encode()))
        else:
            hashes.append(dual_hash(json.dumps(item, sort_keys=True)))

    proof = []
    current_index = index

    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])

        # Find sibling
        if current_index % 2 == 0:
            # Sibling is on the right
            if current_index + 1 < len(hashes):
                proof.append((hashes[current_index + 1], 'R'))
        else:
            # Sibling is on the left
            proof.append((hashes[current_index - 1], 'L'))

        # Build next level
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            next_level.append(dual_hash(combined))

        hashes = next_level
        current_index //= 2

    return proof


def verify_proof(item: Any, proof: list[tuple[str, str]], expected_root: str) -> bool:
    """Validate an inclusion proof against expected Merkle root.

    Args:
        item: The item whose inclusion is being verified
        proof: List of (hash, direction) tuples from merkle_proof
        expected_root: The expected Merkle root

    Returns:
        True if proof is valid, False otherwise
    """
    # Hash the item
    if isinstance(item, bytes):
        current_hash = dual_hash(item)
    elif isinstance(item, str):
        current_hash = dual_hash(item.encode())
    else:
        current_hash = dual_hash(json.dumps(item, sort_keys=True))

    # Walk up the tree
    for sibling_hash, direction in proof:
        if direction == 'R':
            combined = current_hash + sibling_hash
        else:  # direction == 'L'
            combined = sibling_hash + current_hash
        current_hash = dual_hash(combined)

    return current_hash == expected_root


def emit_stoprule(e: Exception, metric: str, action: str = "halt") -> dict:
    """Emit anomaly receipt for a stoprule violation.

    Args:
        e: The exception that triggered the stoprule
        metric: The metric that violated
        action: Action to take (halt, escalate, alert)

    Returns:
        The anomaly receipt
    """
    return emit_receipt("anomaly", {
        "metric": metric,
        "baseline": 0,
        "actual": -1,
        "delta": -1,
        "classification": "violation",
        "action": action,
        "error": str(e)
    })


def measure_latency(func):
    """Decorator to measure function latency and emit receipt if SLO violated."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Check against SLO thresholds
        slo_limits = {
            "dual_hash": 10,
            "merkle_root": 50,
            "merkle_proof": 50,
            "verify_proof": 50,
        }

        limit = slo_limits.get(func.__name__)
        if limit and elapsed_ms > limit:
            emit_receipt("anomaly", {
                "metric": f"{func.__name__}_latency",
                "baseline": limit,
                "actual": elapsed_ms,
                "delta": elapsed_ms - limit,
                "classification": "degradation",
                "action": "alert"
            })

        return result
    return wrapper


def load_receipts(file_path: Optional[Path] = None) -> list[dict]:
    """Load all receipts from the ledger file.

    Args:
        file_path: Path to receipts file, defaults to RECEIPTS_FILE

    Returns:
        List of receipt dicts
    """
    path = file_path or RECEIPTS_FILE
    receipts = []

    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    receipts.append(json.loads(line))
    except FileNotFoundError:
        pass

    return receipts


def get_receipt_count() -> int:
    """Get the current receipt counter value."""
    return _receipt_counter


def reset_receipt_counter():
    """Reset the receipt counter (for testing)."""
    global _receipt_counter
    _receipt_counter = 0


# Module initialization receipt
if __name__ != "__main__":
    # Don't emit on direct run to avoid test pollution
    pass
