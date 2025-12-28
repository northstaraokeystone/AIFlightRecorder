"""Constraint Validators for Monte Carlo Simulations

Provides validation for:
- Hash chain integrity
- Merkle tree consistency
- Compression anomaly calibration
- Receipt CLAUDEME compliance
"""

import json
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import dual_hash
from src.anchor import MerkleTree
from src.compress import build_baseline, detect_anomaly, CompressionBaseline


class IntegrityValidator:
    """Validates hash chain consistency every cycle."""

    def __init__(self):
        self.prev_hash: Optional[str] = None
        self.violations: list[dict] = []
        self.checks_passed: int = 0

    def validate(self, decision: dict, stored_hash: str, stored_prev: str) -> bool:
        """Validate a single decision's integrity.

        Args:
            decision: Decision dict
            stored_hash: Stored hash for this decision
            stored_prev: Stored previous hash

        Returns:
            True if valid
        """
        # Compute hash
        computed_hash = dual_hash(json.dumps(decision, sort_keys=True))

        # Check hash matches
        if computed_hash != stored_hash:
            self.violations.append({
                "type": "hash_mismatch",
                "expected": stored_hash,
                "computed": computed_hash
            })
            return False

        # Check chain linkage
        if self.prev_hash is not None and stored_prev != self.prev_hash:
            self.violations.append({
                "type": "chain_break",
                "expected_prev": self.prev_hash,
                "stored_prev": stored_prev
            })
            return False

        self.prev_hash = stored_hash
        self.checks_passed += 1
        return True

    def reset(self):
        """Reset validator state."""
        self.prev_hash = None
        self.violations = []
        self.checks_passed = 0

    def get_report(self) -> dict:
        """Get validation report."""
        return {
            "checks_passed": self.checks_passed,
            "violations": len(self.violations),
            "is_valid": len(self.violations) == 0,
            "violation_details": self.violations
        }


class MerkleValidator:
    """Validates Merkle tree derivation every cycle."""

    def __init__(self):
        self.tree = MerkleTree()
        self.violations: list[dict] = []
        self.items_added: int = 0

    def add_and_validate(self, item: dict, expected_root: Optional[str] = None) -> bool:
        """Add item and optionally validate against expected root.

        Args:
            item: Item to add to tree
            expected_root: Optional expected root after addition

        Returns:
            True if valid
        """
        self.tree.add_leaf(item)
        self.items_added += 1

        if expected_root is not None:
            actual_root = self.tree.get_root()
            if actual_root != expected_root:
                self.violations.append({
                    "type": "root_mismatch",
                    "position": self.items_added - 1,
                    "expected": expected_root,
                    "actual": actual_root
                })
                return False

        return True

    def validate_proof(self, item: dict, index: int) -> bool:
        """Validate inclusion proof for item.

        Args:
            item: Item to verify
            index: Index in tree

        Returns:
            True if proof valid
        """
        try:
            proof = self.tree.get_proof(index)
            item_hash = dual_hash(json.dumps(item, sort_keys=True))
            is_valid = self.tree.verify_inclusion(item_hash, proof, self.tree.get_root())

            if not is_valid:
                self.violations.append({
                    "type": "proof_invalid",
                    "index": index
                })

            return is_valid
        except Exception as e:
            self.violations.append({
                "type": "proof_error",
                "index": index,
                "error": str(e)
            })
            return False

    def reset(self):
        """Reset validator state."""
        self.tree = MerkleTree()
        self.violations = []
        self.items_added = 0

    def get_report(self) -> dict:
        """Get validation report."""
        return {
            "items_validated": self.items_added,
            "violations": len(self.violations),
            "is_valid": len(self.violations) == 0,
            "current_root": self.tree.get_root(),
            "violation_details": self.violations
        }


class CompressionValidator:
    """Validates anomaly detection calibration."""

    def __init__(self, baseline_size: int = 100):
        self.baseline_size = baseline_size
        self.baseline: Optional[CompressionBaseline] = None
        self.training_data: list[dict] = []
        self.true_positives: int = 0
        self.true_negatives: int = 0
        self.false_positives: int = 0
        self.false_negatives: int = 0

    def add_training(self, decision: dict):
        """Add decision to training data.

        Args:
            decision: Normal decision for baseline
        """
        self.training_data.append(decision)
        if len(self.training_data) >= self.baseline_size:
            self.baseline = build_baseline(self.training_data)

    def validate(self, decision: dict, is_anomaly_ground_truth: bool) -> bool:
        """Validate anomaly detection.

        Args:
            decision: Decision to check
            is_anomaly_ground_truth: Ground truth - is this actually anomalous?

        Returns:
            True if detection matches ground truth
        """
        if self.baseline is None:
            return True  # Not enough training data

        is_detected, score, reason = detect_anomaly(decision, self.baseline)

        if is_detected and is_anomaly_ground_truth:
            self.true_positives += 1
            return True
        elif not is_detected and not is_anomaly_ground_truth:
            self.true_negatives += 1
            return True
        elif is_detected and not is_anomaly_ground_truth:
            self.false_positives += 1
            return False
        else:  # not is_detected and is_anomaly_ground_truth
            self.false_negatives += 1
            return False

    def get_metrics(self) -> dict:
        """Get detection metrics."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives

        if total == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0}

        accuracy = (self.true_positives + self.true_negatives) / total

        precision_denom = self.true_positives + self.false_positives
        precision = self.true_positives / precision_denom if precision_denom > 0 else 0

        recall_denom = self.true_positives + self.false_negatives
        recall = self.true_positives / recall_denom if recall_denom > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives
        }

    def reset(self):
        """Reset validator state."""
        self.baseline = None
        self.training_data = []
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0


class ReceiptValidator:
    """Validates CLAUDEME compliance for receipts."""

    REQUIRED_FIELDS = {
        "decision": ["receipt_type", "ts", "tenant_id", "decision_id", "payload_hash"],
        "anchor": ["receipt_type", "ts", "tenant_id", "merkle_root", "hash_algos", "payload_hash"],
        "anomaly": ["receipt_type", "ts", "tenant_id", "metric", "classification", "action", "payload_hash"],
        "sync": ["receipt_type", "ts", "tenant_id", "edge_device_id", "consistency_verified", "payload_hash"],
        "topology": ["receipt_type", "ts", "tenant_id", "pattern_id", "topology", "effectiveness", "payload_hash"]
    }

    def __init__(self):
        self.violations: list[dict] = []
        self.receipts_validated: int = 0

    def validate(self, receipt: dict) -> bool:
        """Validate receipt CLAUDEME compliance.

        Args:
            receipt: Receipt to validate

        Returns:
            True if compliant
        """
        self.receipts_validated += 1

        receipt_type = receipt.get("receipt_type", "unknown")
        required = self.REQUIRED_FIELDS.get(receipt_type, ["receipt_type", "ts", "tenant_id", "payload_hash"])

        is_valid = True
        for field in required:
            if field not in receipt:
                self.violations.append({
                    "receipt_type": receipt_type,
                    "missing_field": field,
                    "receipt_index": self.receipts_validated - 1
                })
                is_valid = False

        # Validate hash format
        payload_hash = receipt.get("payload_hash", "")
        if payload_hash and ":" not in payload_hash:
            self.violations.append({
                "receipt_type": receipt_type,
                "error": "payload_hash not in dual-hash format",
                "receipt_index": self.receipts_validated - 1
            })
            is_valid = False

        return is_valid

    def get_report(self) -> dict:
        """Get validation report."""
        return {
            "receipts_validated": self.receipts_validated,
            "violations": len(self.violations),
            "is_valid": len(self.violations) == 0,
            "violation_details": self.violations
        }

    def reset(self):
        """Reset validator state."""
        self.violations = []
        self.receipts_validated = 0


class ComprehensiveValidator:
    """Combines all validators for full validation."""

    def __init__(self):
        self.integrity = IntegrityValidator()
        self.merkle = MerkleValidator()
        self.compression = CompressionValidator()
        self.receipt = ReceiptValidator()

    def validate_decision(self, decision: dict, receipt: dict,
                          is_anomaly: bool = False) -> dict:
        """Run all validations on a decision.

        Args:
            decision: The decision
            receipt: The receipt
            is_anomaly: Whether this is known anomalous

        Returns:
            Validation results
        """
        results = {}

        # Receipt compliance
        results["receipt_valid"] = self.receipt.validate(receipt)

        # Add to Merkle tree
        self.merkle.add_and_validate(receipt)

        # Compression check (after baseline established)
        if not self.compression.baseline:
            self.compression.add_training(decision)
            results["compression_checked"] = False
        else:
            results["compression_correct"] = self.compression.validate(decision, is_anomaly)

        return results

    def get_full_report(self) -> dict:
        """Get comprehensive validation report."""
        return {
            "integrity": self.integrity.get_report(),
            "merkle": self.merkle.get_report(),
            "compression": self.compression.get_metrics(),
            "receipt": self.receipt.get_report()
        }

    def reset(self):
        """Reset all validators."""
        self.integrity.reset()
        self.merkle.reset()
        self.compression.reset()
        self.receipt.reset()
