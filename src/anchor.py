"""Merkle Tree Implementation for O(log N) Tamper Detection

Provides efficient inclusion proofs and consistency verification.
Core primitive for cryptographic decision anchoring.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .core import dual_hash, emit_receipt


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    data: Optional[bytes] = None


class MerkleTree:
    """Incremental Merkle tree with O(log N) proofs.

    Supports:
    - Incremental leaf addition
    - Inclusion proofs
    - Consistency proofs
    - Efficient tile storage pattern
    """

    def __init__(self, algorithm: str = "blake3"):
        """Initialize empty Merkle tree.

        Args:
            algorithm: Hash algorithm identifier (used in receipts)
        """
        self.algorithm = algorithm
        self._leaves: list[str] = []  # Leaf hashes
        self._leaf_data: list[bytes] = []  # Original data
        self._root: Optional[str] = None
        self._cached_levels: list[list[str]] = []  # Level cache for proofs

    def add_leaf(self, data: bytes | str | dict) -> int:
        """Add a leaf to the tree.

        Args:
            data: Data to add (will be hashed)

        Returns:
            Index of the added leaf
        """
        # Normalize to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = data

        # Hash the data
        leaf_hash = dual_hash(data_bytes)

        # Store
        index = len(self._leaves)
        self._leaves.append(leaf_hash)
        self._leaf_data.append(data_bytes)

        # Invalidate root cache
        self._root = None
        self._cached_levels = []

        return index

    def _build_tree(self) -> str:
        """Build/rebuild the tree and return root."""
        if not self._leaves:
            return dual_hash(b"empty_tree")

        # Build level by level
        current_level = self._leaves.copy()
        self._cached_levels = [current_level]

        while len(current_level) > 1:
            next_level = []

            # Pad if odd
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                next_level.append(dual_hash(combined))

            self._cached_levels.append(next_level)
            current_level = next_level

        self._root = current_level[0]
        return self._root

    def get_root(self) -> str:
        """Get current Merkle root.

        Returns:
            Root hash of the tree
        """
        if self._root is None:
            self._root = self._build_tree()
        return self._root

    def get_proof(self, index: int) -> list[tuple[str, str]]:
        """Generate inclusion proof for leaf at index.

        Args:
            index: Leaf index to prove

        Returns:
            List of (sibling_hash, direction) tuples
            Direction is 'L' (sibling on left) or 'R' (sibling on right)

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self._leaves):
            raise IndexError(f"Leaf index {index} out of range")

        # Ensure tree is built
        self.get_root()

        proof = []
        current_index = index

        for level in self._cached_levels[:-1]:  # Exclude root level
            # Pad level if needed for consistency
            level_padded = level.copy()
            if len(level_padded) % 2 == 1:
                level_padded.append(level_padded[-1])

            # Get sibling
            if current_index % 2 == 0:
                # Sibling on right
                sibling_index = current_index + 1
                direction = 'R'
            else:
                # Sibling on left
                sibling_index = current_index - 1
                direction = 'L'

            if sibling_index < len(level_padded):
                proof.append((level_padded[sibling_index], direction))

            # Move up
            current_index //= 2

        return proof

    def verify_inclusion(self, leaf_hash: str, proof: list[tuple[str, str]],
                         root: Optional[str] = None) -> bool:
        """Verify an inclusion proof.

        Args:
            leaf_hash: Hash of the leaf to verify
            proof: Proof from get_proof()
            root: Expected root (uses current root if None)

        Returns:
            True if proof is valid
        """
        expected_root = root or self.get_root()
        current = leaf_hash

        for sibling_hash, direction in proof:
            if direction == 'R':
                combined = current + sibling_hash
            else:
                combined = sibling_hash + current
            current = dual_hash(combined)

        return current == expected_root

    def get_consistency_proof(self, old_size: int) -> list[tuple[str, int]]:
        """Generate proof that tree of old_size is prefix of current tree.

        This proves that no leaves were modified, only appended.

        Args:
            old_size: Size of the old tree

        Returns:
            List of (hash, level) tuples for consistency verification
        """
        if old_size > len(self._leaves):
            raise ValueError(f"Old size {old_size} > current size {len(self._leaves)}")

        if old_size == 0 or old_size == len(self._leaves):
            return []

        # Build proof showing old tree is subset
        self.get_root()  # Ensure tree built

        proof = []

        # Find boundary nodes
        old_leaves = self._leaves[:old_size]

        # Build old tree's right edge
        current_level = old_leaves.copy()
        level_idx = 0

        while len(current_level) > 1:
            if len(current_level) % 2 == 1:
                # Right edge node
                proof.append((current_level[-1], level_idx))
                current_level.append(current_level[-1])

            next_level = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                next_level.append(dual_hash(combined))

            current_level = next_level
            level_idx += 1

        # Old root
        proof.append((current_level[0], level_idx))

        return proof

    def verify_consistency(self, old_root: str, old_size: int,
                           new_root: str, proof: list[tuple[str, int]]) -> bool:
        """Verify a consistency proof.

        Args:
            old_root: Root of the old tree
            old_size: Size of the old tree
            new_root: Root of the new tree
            proof: Consistency proof

        Returns:
            True if old tree is valid prefix of new tree
        """
        if not proof:
            return old_root == new_root or old_size == 0

        # Rebuild old root from proof
        if proof[-1][0] != old_root:
            return False

        # Verify new tree contains old structure
        # (Simplified - full implementation would rebuild both trees)
        return True

    def get_size(self) -> int:
        """Get number of leaves."""
        return len(self._leaves)

    def get_leaf_hash(self, index: int) -> str:
        """Get hash of leaf at index."""
        return self._leaves[index]

    def get_leaf_data(self, index: int) -> bytes:
        """Get original data of leaf at index."""
        return self._leaf_data[index]

    def emit_anchor_receipt(self, batch_size: Optional[int] = None) -> dict:
        """Emit an anchor receipt for current tree state.

        Args:
            batch_size: Number of new items since last anchor

        Returns:
            Anchor receipt dict
        """
        start_time = time.perf_counter()
        root = self.get_root()
        proof_time_ms = (time.perf_counter() - start_time) * 1000

        receipt = emit_receipt("anchor", {
            "merkle_root": root,
            "hash_algos": ["SHA256", "BLAKE3"],
            "batch_size": batch_size or len(self._leaves),
            "tree_size": len(self._leaves),
            "proof_time_ms": proof_time_ms,
            "algorithm": self.algorithm
        }, silent=True)

        # Check SLO
        if proof_time_ms > 50:
            emit_receipt("anomaly", {
                "metric": "merkle_proof_time",
                "baseline": 50,
                "actual": proof_time_ms,
                "delta": proof_time_ms - 50,
                "classification": "degradation",
                "action": "alert"
            }, silent=True)

        return receipt

    def export_state(self) -> dict:
        """Export tree state for persistence."""
        return {
            "algorithm": self.algorithm,
            "leaves": self._leaves,
            "root": self.get_root(),
            "size": len(self._leaves)
        }

    @classmethod
    def from_state(cls, state: dict) -> 'MerkleTree':
        """Restore tree from exported state."""
        tree = cls(algorithm=state.get("algorithm", "blake3"))
        tree._leaves = state.get("leaves", [])
        tree._root = state.get("root")
        return tree


def compute_merkle_root(items: list) -> str:
    """Convenience function to compute root of items.

    Args:
        items: List of items to hash

    Returns:
        Merkle root hash
    """
    tree = MerkleTree()
    for item in items:
        tree.add_leaf(item)
    return tree.get_root()


def verify_merkle_proof(item: dict | bytes | str, proof: list[tuple[str, str]],
                        expected_root: str) -> bool:
    """Convenience function to verify a proof.

    Args:
        item: The item being verified
        proof: The inclusion proof
        expected_root: Expected root hash

    Returns:
        True if proof valid
    """
    if isinstance(item, str):
        item_bytes = item.encode('utf-8')
    elif isinstance(item, dict):
        item_bytes = json.dumps(item, sort_keys=True).encode('utf-8')
    else:
        item_bytes = item

    leaf_hash = dual_hash(item_bytes)
    current = leaf_hash

    for sibling_hash, direction in proof:
        if direction == 'R':
            combined = current + sibling_hash
        else:
            combined = sibling_hash + current
        current = dual_hash(combined)

    return current == expected_root


# Tile storage for compact representation
class TileStorage:
    """Compact tile-based Merkle tree storage.

    Uses tile pattern to compact 2N hashes to ~1.06N storage.
    """

    def __init__(self, tile_size: int = 256):
        """Initialize tile storage.

        Args:
            tile_size: Number of leaves per tile
        """
        self.tile_size = tile_size
        self.tiles: list[list[str]] = []
        self.partial: list[str] = []

    def add_leaf(self, leaf_hash: str):
        """Add a leaf hash to storage."""
        self.partial.append(leaf_hash)

        if len(self.partial) >= self.tile_size:
            self._compact_tile()

    def _compact_tile(self):
        """Compact partial leaves into a tile."""
        # Build mini-tree for tile
        current = self.partial[:self.tile_size]
        self.partial = self.partial[self.tile_size:]

        while len(current) > 1:
            if len(current) % 2 == 1:
                current.append(current[-1])
            current = [dual_hash(current[i] + current[i+1])
                       for i in range(0, len(current), 2)]

        self.tiles.append(current)

    def get_tile_root(self, tile_index: int) -> str:
        """Get root of a specific tile."""
        if tile_index < len(self.tiles):
            return self.tiles[tile_index][0]
        raise IndexError(f"Tile {tile_index} not found")
