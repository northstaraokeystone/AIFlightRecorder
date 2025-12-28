"""Tests for Merkle tree implementation."""

import pytest
import time

from src.anchor import MerkleTree, compute_merkle_root, verify_merkle_proof


class TestMerkleTree:
    """Tests for MerkleTree class."""

    def test_empty_tree(self):
        """Empty tree should return valid root."""
        tree = MerkleTree()
        root = tree.get_root()
        assert ":" in root  # Dual hash format

    def test_add_single_leaf(self):
        """Adding single leaf should work."""
        tree = MerkleTree()
        index = tree.add_leaf(b"data")
        assert index == 0
        assert tree.get_size() == 1

    def test_add_multiple_leaves(self):
        """Adding multiple leaves should increment size."""
        tree = MerkleTree()
        for i in range(10):
            idx = tree.add_leaf(f"data_{i}")
            assert idx == i

        assert tree.get_size() == 10

    def test_root_changes_with_new_leaf(self):
        """Root should change when new leaf added."""
        tree = MerkleTree()
        tree.add_leaf(b"first")
        root1 = tree.get_root()

        tree.add_leaf(b"second")
        root2 = tree.get_root()

        assert root1 != root2

    def test_deterministic_root(self):
        """Same leaves should produce same root."""
        tree1 = MerkleTree()
        tree2 = MerkleTree()

        for i in range(5):
            tree1.add_leaf(f"data_{i}")
            tree2.add_leaf(f"data_{i}")

        assert tree1.get_root() == tree2.get_root()

    def test_proof_generation(self):
        """Should generate valid inclusion proofs."""
        tree = MerkleTree()
        for i in range(8):
            tree.add_leaf(f"data_{i}")

        # Get proof for each leaf
        for i in range(8):
            proof = tree.get_proof(i)
            assert isinstance(proof, list)
            # Proof should have direction indicators
            for item in proof:
                assert len(item) == 2  # (hash, direction)
                assert item[1] in ('L', 'R')

    def test_proof_verification(self):
        """Proofs should verify correctly."""
        tree = MerkleTree()
        items = [f"item_{i}" for i in range(16)]

        for item in items:
            tree.add_leaf(item)

        root = tree.get_root()

        # Verify each item
        for i, item in enumerate(items):
            proof = tree.get_proof(i)
            from src.core import dual_hash
            leaf_hash = dual_hash(item.encode())
            assert tree.verify_inclusion(leaf_hash, proof, root)

    def test_invalid_proof_fails(self):
        """Modified proof should fail verification."""
        tree = MerkleTree()
        for i in range(8):
            tree.add_leaf(f"data_{i}")

        proof = tree.get_proof(0)
        root = tree.get_root()

        # Tamper with proof
        if proof:
            tampered = [(proof[0][0][:-1] + "X", proof[0][1])] + proof[1:]
            from src.core import dual_hash
            leaf_hash = dual_hash(b"data_0")
            assert not tree.verify_inclusion(leaf_hash, tampered, root)

    def test_invalid_index_raises(self):
        """Out of range index should raise."""
        tree = MerkleTree()
        tree.add_leaf(b"data")

        with pytest.raises(IndexError):
            tree.get_proof(5)

    def test_accepts_dict(self):
        """Should accept dict input and serialize."""
        tree = MerkleTree()
        idx = tree.add_leaf({"key": "value", "num": 42})
        assert idx == 0
        assert tree.get_size() == 1

    def test_proof_time_slo(self):
        """Proof generation should be under 50ms."""
        tree = MerkleTree()
        for i in range(10000):
            tree.add_leaf(f"data_{i}")

        start = time.perf_counter()
        for _ in range(100):
            tree.get_proof(5000)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 50, f"Proof generation {elapsed_ms}ms exceeds 50ms SLO"


class TestConsistencyProof:
    """Tests for consistency proofs."""

    def test_empty_to_filled(self):
        """Consistency from empty to filled tree."""
        tree = MerkleTree()

        for i in range(10):
            tree.add_leaf(f"data_{i}")

        proof = tree.get_consistency_proof(0)
        assert isinstance(proof, list)

    def test_partial_to_full(self):
        """Consistency from partial to full tree."""
        tree = MerkleTree()

        for i in range(5):
            tree.add_leaf(f"data_{i}")
        old_size = tree.get_size()

        for i in range(5, 10):
            tree.add_leaf(f"data_{i}")

        proof = tree.get_consistency_proof(old_size)
        assert isinstance(proof, list)


class TestAnchorReceipt:
    """Tests for anchor receipt emission."""

    def test_emits_receipt(self):
        """Should emit valid anchor receipt."""
        tree = MerkleTree()
        for i in range(10):
            tree.add_leaf(f"data_{i}")

        receipt = tree.emit_anchor_receipt(batch_size=10)

        assert receipt["receipt_type"] == "anchor"
        assert "merkle_root" in receipt
        assert receipt["batch_size"] == 10
        assert receipt["tree_size"] == 10
        assert "hash_algos" in receipt


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_merkle_root(self):
        """compute_merkle_root should work."""
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        root = compute_merkle_root(items)
        assert ":" in root

    def test_verify_merkle_proof(self):
        """verify_merkle_proof should validate correctly."""
        tree = MerkleTree()
        items = [{"a": 1}, {"b": 2}, {"c": 3}]

        for item in items:
            tree.add_leaf(item)

        proof = tree.get_proof(1)
        root = tree.get_root()

        assert verify_merkle_proof(items[1], proof, root)
        assert not verify_merkle_proof(items[0], proof, root)  # Wrong item
