#!/usr/bin/env python3
"""AI Flight Recorder CLI

Main entry point for running the flight recorder system.

Usage:
    python cli.py --test          # Run smoke test
    python cli.py --demo          # Run tamper demo
    python cli.py --run 100       # Run 100 decision cycles
    python cli.py --validate      # Run validation scenarios
    python cli.py --dashboard     # Launch Streamlit dashboard
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.core import dual_hash, emit_receipt, reset_receipt_counter
from src.drone import run_mission, DroneState
from src.logger import DecisionLogger
from src.anchor import MerkleTree
from src.verify import verify_chain_integrity, generate_integrity_report
from src.compress import build_baseline, AnomalyDetector


def run_test():
    """Run smoke test - emit receipts to verify system works."""
    reset_receipt_counter()

    # Test dual hash
    h = dual_hash(b"test")
    assert ":" in h, "dual_hash must return SHA256:BLAKE3 format"

    # Test receipt emission
    receipt = emit_receipt("test", {
        "message": "Smoke test receipt",
        "status": "ok"
    })
    assert "receipt_type" in receipt
    assert "payload_hash" in receipt
    assert ":" in receipt["payload_hash"]

    # Test decision generation
    decisions, state = run_mission(10, seed=42)
    assert len(decisions) == 10, "Should generate 10 decisions"

    # Test Merkle tree
    tree = MerkleTree()
    for d in decisions:
        tree.add_leaf(d)
    root = tree.get_root()
    assert root, "Merkle tree should have root"

    # Test verification
    is_valid, violations = verify_chain_integrity(decisions)

    # Emit final test receipt
    emit_receipt("test_complete", {
        "decisions_generated": len(decisions),
        "merkle_root": root[:32],
        "chain_valid": is_valid,
        "violations": len(violations)
    })

    print("\n✓ All smoke tests passed\n", file=sys.stderr)
    return True


def run_cycles(n: int, seed: int = 42):
    """Run n decision cycles and output receipts.

    Args:
        n: Number of cycles
        seed: Random seed
    """
    reset_receipt_counter()

    print(f"Running {n} decision cycles...", file=sys.stderr)

    decisions, final_state = run_mission(n, seed=seed)

    # Build Merkle tree
    tree = MerkleTree()
    for d in decisions:
        tree.add_leaf(d)

    # Verify chain
    is_valid, violations = verify_chain_integrity(decisions)

    # Emit summary
    emit_receipt("run_complete", {
        "cycles": n,
        "merkle_root": tree.get_root(),
        "chain_valid": is_valid,
        "violations": len(violations),
        "final_position": {
            "lat": final_state.lat,
            "lon": final_state.lon,
            "alt": final_state.alt
        },
        "battery_remaining": final_state.battery_pct
    })

    print(f"\n✓ Completed {n} cycles", file=sys.stderr)
    print(f"  Chain valid: {is_valid}", file=sys.stderr)
    print(f"  Merkle root: {tree.get_root()[:32]}...", file=sys.stderr)


def run_validation():
    """Run all validation scenarios."""
    from sim.sim import run_all_scenarios
    from sim.scenarios import QUICK_SCENARIOS
    from sim.reporting import format_all_results

    print("Running validation scenarios...\n", file=sys.stderr)

    results = run_all_scenarios(QUICK_SCENARIOS)

    print(format_all_results(results), file=sys.stderr)

    if results["all_passed"]:
        emit_receipt("validation", {
            "status": "passed",
            "scenarios": len(results["scenarios"])
        })
    else:
        emit_receipt("validation", {
            "status": "failed",
            "scenarios": len(results["scenarios"])
        })

    return results["all_passed"]


def run_demo(demo_type: str = "tamper"):
    """Run a demo script.

    Args:
        demo_type: Which demo to run (tamper, replay, sync)
    """
    if demo_type == "tamper":
        from demo.tamper_demo import run_demo as tamper_demo
        tamper_demo(n_cycles=50)
    elif demo_type == "replay":
        from demo.replay_demo import run_demo as replay_demo
        replay_demo()
    elif demo_type == "sync":
        from demo.sync_demo import run_demo as sync_demo
        sync_demo()
    else:
        print(f"Unknown demo type: {demo_type}", file=sys.stderr)
        sys.exit(1)


def run_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "src" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Flight Recorder - Decision Provenance for Autonomous Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --test                  Run smoke test
  python cli.py --run 100               Generate 100 decisions
  python cli.py --demo tamper           Run tamper detection demo
  python cli.py --validate              Run validation scenarios
  python cli.py --dashboard             Launch Streamlit dashboard

For more information, see spec.md
        """
    )

    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run smoke test"
    )

    parser.add_argument(
        "--run", "-r",
        type=int,
        metavar="N",
        help="Run N decision cycles"
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--demo", "-d",
        type=str,
        choices=["tamper", "replay", "sync"],
        help="Run demo (tamper, replay, or sync)"
    )

    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation scenarios"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate integrity report from receipts.jsonl"
    )

    args = parser.parse_args()

    # Handle commands
    if args.test:
        success = run_test()
        sys.exit(0 if success else 1)

    elif args.run:
        run_cycles(args.run, args.seed)

    elif args.demo:
        run_demo(args.demo)

    elif args.validate:
        success = run_validation()
        sys.exit(0 if success else 1)

    elif args.dashboard:
        run_dashboard()

    elif args.report:
        from src.core import load_receipts
        receipts = load_receipts()
        decisions = [r for r in receipts if r.get("receipt_type") == "decision"]
        report = generate_integrity_report(decisions)
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
