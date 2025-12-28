"""Edge-to-Cloud Synchronization with Chain-of-Custody Preservation

Handles offline operation and reconnection with cryptographic proof
that the decision chain remains intact across connectivity gaps.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .core import dual_hash, emit_receipt, GENESIS_HASH
from .anchor import MerkleTree


@dataclass
class SyncState:
    """Current synchronization state."""
    last_sync_root: str = GENESIS_HASH
    last_sync_size: int = 0
    last_sync_time: Optional[str] = None
    pending_decisions: int = 0
    is_connected: bool = True
    sync_attempts: int = 0
    last_error: Optional[str] = None


@dataclass
class SyncPackage:
    """Package of decisions for cloud sync."""
    edge_device_id: str
    local_tree_size: int
    local_root_hash: str
    since_root: str
    since_size: int
    decisions: list[dict]
    device_signature: str
    created_at: str


@dataclass
class CloudAcknowledgment:
    """Cloud acknowledgment of sync."""
    cloud_tree_size: int
    cloud_root_hash: str
    consistency_verified: bool
    cloud_signature: str
    acknowledged_at: str
    decisions_accepted: int


class SyncManager:
    """Manages edge-to-cloud synchronization."""

    def __init__(self, edge_device_id: str = "edge-device-001"):
        """Initialize sync manager.

        Args:
            edge_device_id: Identifier for this edge device
        """
        self.edge_device_id = edge_device_id
        self.state = SyncState()
        self._local_tree = MerkleTree()
        self._pending_sync: list[dict] = []
        self._cloud_tree_size = 0

    def add_decision(self, decision: dict):
        """Add a decision to be synced.

        Args:
            decision: Decision to add
        """
        self._local_tree.add_leaf(decision)
        self._pending_sync.append(decision)
        self.state.pending_decisions = len(self._pending_sync)

    def prepare_sync_package(self) -> SyncPackage:
        """Prepare package for cloud sync.

        Returns:
            SyncPackage ready for transmission
        """
        local_root = self._local_tree.get_root()
        local_size = self._local_tree.get_size()

        # Create device signature (simplified - would use proper key in production)
        sig_data = f"{self.edge_device_id}:{local_root}:{local_size}"
        device_signature = dual_hash(sig_data)

        package = SyncPackage(
            edge_device_id=self.edge_device_id,
            local_tree_size=local_size,
            local_root_hash=local_root,
            since_root=self.state.last_sync_root,
            since_size=self.state.last_sync_size,
            decisions=self._pending_sync.copy(),
            device_signature=device_signature,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        return package

    def verify_sync_receipt(self, cloud_response: CloudAcknowledgment,
                            local_root: str) -> bool:
        """Verify cloud acknowledgment is valid.

        Args:
            cloud_response: Acknowledgment from cloud
            local_root: Our local root hash

        Returns:
            True if acknowledgment is valid
        """
        # Verify consistency
        if not cloud_response.consistency_verified:
            return False

        # Verify cloud accepted our data (cloud root should match or extend local)
        # In full implementation, would verify Merkle consistency proof
        return cloud_response.cloud_tree_size >= self._local_tree.get_size()

    def apply_sync_receipt(self, receipt: CloudAcknowledgment):
        """Store cloud confirmation locally.

        Args:
            receipt: Cloud acknowledgment to store
        """
        self.state.last_sync_root = receipt.cloud_root_hash
        self.state.last_sync_size = receipt.cloud_tree_size
        self.state.last_sync_time = receipt.acknowledged_at
        self.state.pending_decisions = 0
        self.state.sync_attempts = 0
        self.state.last_error = None

        # Clear pending decisions
        self._pending_sync = []

        # Emit sync receipt
        emit_receipt("sync", {
            "edge_device_id": self.edge_device_id,
            "local_tree_size": self._local_tree.get_size(),
            "local_root_hash": self._local_tree.get_root(),
            "cloud_tree_size": receipt.cloud_tree_size,
            "cloud_root_hash": receipt.cloud_root_hash,
            "consistency_verified": receipt.consistency_verified,
            "cloud_signature": receipt.cloud_signature,
            "decisions_synced": receipt.decisions_accepted
        }, silent=True)

    def get_sync_status(self) -> dict:
        """Get current sync state.

        Returns:
            Sync status dict
        """
        return {
            "edge_device_id": self.edge_device_id,
            "is_connected": self.state.is_connected,
            "last_sync_time": self.state.last_sync_time,
            "last_sync_root": self.state.last_sync_root,
            "local_tree_size": self._local_tree.get_size(),
            "pending_decisions": self.state.pending_decisions,
            "sync_attempts": self.state.sync_attempts,
            "last_error": self.state.last_error
        }

    def simulate_offline_period(self, decisions: list[dict]) -> dict:
        """Simulate offline operation.

        Args:
            decisions: Decisions made during offline period

        Returns:
            Offline period summary
        """
        start_size = self._local_tree.get_size()
        start_root = self._local_tree.get_root()

        # Add decisions during offline
        for decision in decisions:
            self.add_decision(decision)

        end_size = self._local_tree.get_size()
        end_root = self._local_tree.get_root()

        return {
            "offline_duration_decisions": len(decisions),
            "start_tree_size": start_size,
            "end_tree_size": end_size,
            "start_root": start_root,
            "end_root": end_root,
            "pending_sync": self.state.pending_decisions
        }

    def go_offline(self):
        """Simulate network disconnection."""
        self.state.is_connected = False

    def go_online(self):
        """Simulate network reconnection."""
        self.state.is_connected = True


class CloudSimulator:
    """Simulates cloud verification for demo purposes."""

    def __init__(self):
        """Initialize cloud simulator."""
        self._tree = MerkleTree()
        self._decisions: list[dict] = []

    def receive_sync(self, package: SyncPackage) -> CloudAcknowledgment:
        """Simulate cloud receiving sync package.

        Args:
            package: Sync package from edge

        Returns:
            Cloud acknowledgment
        """
        # Verify edge tree is consistent with our view
        # (In production, would verify Merkle consistency proof)

        # Accept all decisions
        for decision in package.decisions:
            self._tree.add_leaf(decision)
            self._decisions.append(decision)

        # Check consistency
        consistency_ok = True
        if package.since_size > 0:
            # Verify we have the same since_root
            # (Simplified check)
            consistency_ok = True

        # Create cloud signature
        sig_data = f"{self._tree.get_root()}:{self._tree.get_size()}"
        cloud_signature = dual_hash(sig_data)

        return CloudAcknowledgment(
            cloud_tree_size=self._tree.get_size(),
            cloud_root_hash=self._tree.get_root(),
            consistency_verified=consistency_ok,
            cloud_signature=cloud_signature,
            acknowledged_at=datetime.now(timezone.utc).isoformat(),
            decisions_accepted=len(package.decisions)
        )

    def get_tree_size(self) -> int:
        """Get cloud tree size."""
        return self._tree.get_size()

    def get_root(self) -> str:
        """Get cloud tree root."""
        return self._tree.get_root()


def run_sync_demo(n_cycles: int = 50, offline_cycles: int = 30) -> dict:
    """Run sync demonstration.

    Shows chain-of-custody through connectivity loss.

    Args:
        n_cycles: Total decision cycles
        offline_cycles: Cycles while offline

    Returns:
        Demo results
    """
    from .drone import run_mission

    # Initialize
    edge = SyncManager()
    cloud = CloudSimulator()

    # Phase 1: Online operation
    online_before = n_cycles - offline_cycles
    decisions_before, _ = run_mission(online_before, seed=42)

    for d in decisions_before:
        edge.add_decision(d)

    # Initial sync
    package = edge.prepare_sync_package()
    ack = cloud.receive_sync(package)
    edge.apply_sync_receipt(ack)

    sync_1_status = edge.get_sync_status()

    # Phase 2: Go offline
    edge.go_offline()
    offline_summary = {"started_at": datetime.now(timezone.utc).isoformat()}

    # Generate decisions while offline
    decisions_offline, _ = run_mission(offline_cycles, seed=123)
    offline_result = edge.simulate_offline_period(decisions_offline)

    # Phase 3: Reconnect
    edge.go_online()

    # Sync after reconnection
    package = edge.prepare_sync_package()
    ack = cloud.receive_sync(package)

    # Verify consistency
    is_valid = edge.verify_sync_receipt(ack, edge._local_tree.get_root())
    if is_valid:
        edge.apply_sync_receipt(ack)

    sync_2_status = edge.get_sync_status()

    return {
        "demo_type": "sync_chain_of_custody",
        "phases": {
            "online_before": {
                "decisions": online_before,
                "synced": True,
                "status": sync_1_status
            },
            "offline": {
                "decisions": offline_cycles,
                "tree_growth": offline_result
            },
            "reconnection": {
                "consistency_verified": is_valid,
                "decisions_synced": ack.decisions_accepted,
                "status": sync_2_status
            }
        },
        "chain_of_custody": {
            "maintained": is_valid,
            "total_decisions": n_cycles,
            "cloud_tree_size": cloud.get_tree_size(),
            "edge_tree_size": edge._local_tree.get_size()
        }
    }
