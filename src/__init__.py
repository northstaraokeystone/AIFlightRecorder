"""AI Flight Recorder - Decision Provenance for Autonomous Systems

This package provides cryptographically verifiable records of AI decisions
at execution time. Not post-hoc explanations, but proofs.

Core Components:
- core: Foundation functions (dual_hash, emit_receipt, merkle)
- drone: Simulated AI decision generator
- logger: Decision capture with sensor context
- anchor: Merkle tree implementation
- compress: NCD anomaly detection
- verify: Tamper detection engine
- topology: META-LOOP pattern classification
- sync: Edge-to-cloud chain-of-custody
- dashboard: Streamlit visualization
"""

__version__ = "1.0.0"
__author__ = "AI Flight Recorder Team"

from .core import dual_hash, emit_receipt, merkle_root, merkle_proof, verify_proof, StopRule
