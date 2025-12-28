#!/usr/bin/env python3
"""Watchdog Daemon for AI Flight Recorder

Monitors system health and integrity in production.

Usage:
    python watchdog.py --check     # One-time health check
    python watchdog.py --daemon    # Run as daemon (continuous monitoring)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core import emit_receipt, load_receipts, dual_hash
from src.verify import verify_chain_integrity, generate_integrity_report
from src.compress import AnomalyDetector


def health_check() -> dict:
    """Run comprehensive health check.

    Returns:
        Health status dict
    """
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "healthy": True
    }

    # Check 1: Core functions work
    try:
        h = dual_hash(b"health_check")
        status["checks"]["dual_hash"] = {"status": "ok", "result": h[:32]}
    except Exception as e:
        status["checks"]["dual_hash"] = {"status": "error", "error": str(e)}
        status["healthy"] = False

    # Check 2: Decision generation works
    try:
        from src.drone import run_cycle
        state, receipt = run_cycle({}, seed=42)
        status["checks"]["decision_generation"] = {"status": "ok"}
    except Exception as e:
        status["checks"]["decision_generation"] = {"status": "error", "error": str(e)}
        status["healthy"] = False

    # Check 3: Merkle tree works
    try:
        from src.anchor import MerkleTree
        tree = MerkleTree()
        tree.add_leaf(b"test")
        root = tree.get_root()
        status["checks"]["merkle_tree"] = {"status": "ok", "root": root[:32]}
    except Exception as e:
        status["checks"]["merkle_tree"] = {"status": "error", "error": str(e)}
        status["healthy"] = False

    # Check 4: Receipts file accessible
    try:
        receipts = load_receipts()
        status["checks"]["receipts_file"] = {
            "status": "ok",
            "count": len(receipts)
        }
    except Exception as e:
        status["checks"]["receipts_file"] = {"status": "error", "error": str(e)}
        # Not fatal if no receipts yet

    # Check 5: Chain integrity (if receipts exist)
    try:
        receipts = load_receipts()
        decisions = [r for r in receipts if r.get("receipt_type") == "decision"]
        if decisions:
            is_valid, violations = verify_chain_integrity(decisions)
            status["checks"]["chain_integrity"] = {
                "status": "ok" if is_valid else "warning",
                "valid": is_valid,
                "violations": len(violations)
            }
            if not is_valid:
                status["healthy"] = False
        else:
            status["checks"]["chain_integrity"] = {"status": "ok", "message": "No decisions yet"}
    except Exception as e:
        status["checks"]["chain_integrity"] = {"status": "error", "error": str(e)}

    # Check 6: Compression works
    try:
        from src.compress import compress_decision
        compressed = compress_decision({"test": "data"})
        status["checks"]["compression"] = {"status": "ok", "size": len(compressed)}
    except Exception as e:
        status["checks"]["compression"] = {"status": "error", "error": str(e)}
        status["healthy"] = False

    return status


def run_daemon(interval: int = 60):
    """Run watchdog as daemon.

    Args:
        interval: Check interval in seconds
    """
    print(f"Starting watchdog daemon (interval: {interval}s)")
    print("Press Ctrl+C to stop\n")

    check_count = 0

    try:
        while True:
            check_count += 1
            status = health_check()

            # Print status
            ts = status["timestamp"]
            healthy = "✓ HEALTHY" if status["healthy"] else "✗ UNHEALTHY"
            print(f"[{ts}] Check #{check_count}: {healthy}")

            # Emit receipt
            emit_receipt("watchdog", {
                "check_number": check_count,
                "healthy": status["healthy"],
                "checks_passed": sum(1 for c in status["checks"].values()
                                     if c.get("status") == "ok"),
                "checks_total": len(status["checks"])
            }, silent=True)

            # If unhealthy, print details
            if not status["healthy"]:
                for name, check in status["checks"].items():
                    if check.get("status") != "ok":
                        print(f"  ! {name}: {check}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nWatchdog stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Flight Recorder Watchdog"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run one-time health check"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (daemon mode)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.interval)
    elif args.check:
        status = health_check()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n" + "=" * 50)
            print("  WATCHDOG HEALTH CHECK")
            print("=" * 50)
            print(f"\n  Status: {'✓ HEALTHY' if status['healthy'] else '✗ UNHEALTHY'}")
            print(f"  Time: {status['timestamp']}\n")

            for name, check in status["checks"].items():
                icon = "✓" if check.get("status") == "ok" else "✗"
                print(f"  {icon} {name}: {check.get('status', 'unknown')}")
                if check.get("error"):
                    print(f"      Error: {check['error']}")

            print("\n" + "=" * 50)

        sys.exit(0 if status["healthy"] else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
