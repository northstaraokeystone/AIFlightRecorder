"""Pytest configuration and fixtures for AI Flight Recorder tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment
os.environ["RECEIPTS_FILE"] = str(Path(tempfile.gettempdir()) / "test_receipts.jsonl")


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    from src.core import reset_receipt_counter, RECEIPTS_FILE

    reset_receipt_counter()

    # Clear receipts file
    receipts_path = Path(os.environ.get("RECEIPTS_FILE", "receipts.jsonl"))
    if receipts_path.exists():
        receipts_path.unlink()

    yield

    # Cleanup after test
    if receipts_path.exists():
        receipts_path.unlink()


@pytest.fixture
def sample_decisions():
    """Generate sample decisions for testing."""
    from src.drone import run_mission
    decisions, _ = run_mission(20, seed=42)
    return decisions


@pytest.fixture
def merkle_tree():
    """Create a fresh Merkle tree."""
    from src.anchor import MerkleTree
    return MerkleTree()


@pytest.fixture
def decision_logger(tmp_path):
    """Create a decision logger with temporary storage."""
    from src.logger import DecisionLogger
    return DecisionLogger(
        db_path=tmp_path / "test.db",
        ledger_path=tmp_path / "test_receipts.jsonl"
    )


@pytest.fixture
def compression_baseline(sample_decisions):
    """Create compression baseline from sample decisions."""
    from src.compress import build_baseline

    # Get full decision data
    full_decisions = []
    for d in sample_decisions:
        if "full_decision" in d:
            full_decisions.append(d["full_decision"])
        else:
            full_decisions.append(d)

    return build_baseline(full_decisions)
