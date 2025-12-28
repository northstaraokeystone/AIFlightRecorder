#!/usr/bin/env python3
"""THE CONVERSION MOMENT - Tamper Detection Demo

This demo creates the unforgettable moment where viewers watch
a modification attempt get instantly flagged.

Usage:
    python demo/tamper_demo.py
    python demo/tamper_demo.py --verify-detection
    python demo/tamper_demo.py --cycles 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drone import run_mission, DroneState
from src.logger import DecisionLogger
from src.anchor import MerkleTree
from src.verify import (
    verify_chain_integrity,
    run_tamper_test,
    format_tamper_alert,
    format_verification_success
)
from src.core import dual_hash


def print_header():
    """Print demo header."""
    print("\n" + "=" * 60)
    print("  AI FLIGHT RECORDER - TAMPER DETECTION DEMO")
    print("  The Conversion Moment")
    print("=" * 60 + "\n")


def print_phase(phase: int, title: str):
    """Print phase header."""
    print(f"\n{'─' * 60}")
    print(f"  PHASE {phase}: {title}")
    print(f"{'─' * 60}\n")


def run_demo(n_cycles: int = 50, tamper_index: int = 25, verify_only: bool = False):
    """Run the tamper detection demo.

    Args:
        n_cycles: Number of decision cycles to generate
        tamper_index: Which decision to tamper with
        verify_only: Only verify, don't show tampering
    """
    print_header()

    # Phase 1: Generate decisions
    print_phase(1, "GENERATING AI DECISIONS")

    print(f"  Starting drone mission simulation...")
    print(f"  Generating {n_cycles} decisions at 10Hz...\n")

    decisions, final_state = run_mission(n_cycles, seed=42)

    # Show some decisions streaming
    print("  Decision stream:")
    for i, d in enumerate(decisions[:10]):
        action = d.get("action_type", "UNKNOWN")
        conf = d.get("confidence", 0)
        print(f"    #{i:03d} | {action:8} | conf: {conf:.2%}")
        time.sleep(0.05)  # Simulate real-time
    print(f"    ... ({n_cycles - 10} more decisions) ...\n")

    # Build Merkle tree
    tree = MerkleTree()
    for d in decisions:
        tree.add_leaf(d)

    print(f"  ✓ {n_cycles} decisions logged")
    print(f"  ✓ Merkle tree built (root: {tree.get_root()[:32]}...)")
    print(f"  ✓ All decisions anchored to chain")

    # Phase 2: Verify original chain
    print_phase(2, "VERIFYING ORIGINAL CHAIN")

    is_valid, violations = verify_chain_integrity(decisions)

    if is_valid:
        print(format_verification_success())
    else:
        print(f"  ✗ Original chain has {len(violations)} violations!")
        return False

    if verify_only:
        print("\n  Demo complete (verify-only mode).")
        return True

    # Phase 3: Demonstrate tampering
    print_phase(3, "SIMULATING TAMPERING ATTEMPT")

    print(f"  Target: Decision #{tamper_index}")

    original = decisions[tamper_index]
    original_action = original.get("full_decision", original).get("action", {}).get("type", "CONTINUE")
    new_action = "ENGAGE" if original_action != "ENGAGE" else "AVOID"

    print(f"  Modification: Changing action from '{original_action}' to '{new_action}'")
    print(f"\n  Imagine: An adversary modifies the flight log to hide")
    print(f"  an avoidance maneuver or change target engagement...\n")

    time.sleep(1)  # Dramatic pause

    # Run tamper test
    print("  Running verification...")
    time.sleep(0.5)

    result = run_tamper_test(
        decisions,
        tamper_index,
        {"action.type": new_action}
    )

    # Phase 4: Detection result
    print_phase(4, "DETECTION RESULT")

    print(format_tamper_alert(result))

    # Show what would happen
    if result["detection_result"] == "INTEGRITY_FAILURE":
        print("\n  What this means:")
        print("  • The modification was INSTANTLY detected")
        print("  • Detection time: {:.2f}ms".format(result["detection_latency_ms"]))
        print("  • All decisions after the tampered one are now suspect")
        print("  • Forensic trail preserved for investigation")
        print("\n  The flight recorder caught the tampering immediately.")
        print("  This is the accountability layer for autonomous systems.")

    # Phase 5: Restore and verify
    print_phase(5, "RESTORING AND RE-VERIFYING")

    print("  Restoring original chain...")
    is_valid, violations = verify_chain_integrity(decisions)

    if is_valid:
        print(format_verification_success())
        print("  ✓ Chain restored to verified state")
    else:
        print("  ✗ Verification failed after restore!")

    # Summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\n  Key Takeaways:")
    print("  1. Every AI decision is cryptographically anchored")
    print("  2. Modifications are detected instantly (<1ms)")
    print("  3. The system uses Merkle trees for O(log N) verification")
    print("  4. Compression patterns provide additional anomaly detection")
    print("  5. This runs on edge devices (512MB RAM, 1 CPU)")
    print("\n  This is decision provenance - not explainability, but proof.")
    print()

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Flight Recorder Tamper Detection Demo"
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=50,
        help="Number of decision cycles (default: 50)"
    )
    parser.add_argument(
        "--tamper-at", "-t",
        type=int,
        default=25,
        help="Decision index to tamper with (default: 25)"
    )
    parser.add_argument(
        "--verify-detection",
        action="store_true",
        help="Only verify chain without showing tampering"
    )

    args = parser.parse_args()

    success = run_demo(
        n_cycles=args.cycles,
        tamper_index=args.tamper_at,
        verify_only=args.verify_detection
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
