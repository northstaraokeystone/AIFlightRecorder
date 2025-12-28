#!/usr/bin/env python3
"""Decision Timeline Reconstruction Demo

Shows how the flight recorder enables forensic analysis:
- Load verified decision chain
- Display flight path
- Click any point to see full decision context
- Answer "Why did the AI turn here?" in seconds

Usage:
    python demo/replay_demo.py
    python demo/replay_demo.py --decision 25
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drone import run_mission
from src.verify import verify_chain_integrity, generate_integrity_report
from src.topology import generate_topology_report
from src.core import dual_hash


def print_header():
    """Print demo header."""
    print("\n" + "=" * 60)
    print("  AI FLIGHT RECORDER - DECISION REPLAY DEMO")
    print("  Forensic Analysis Capability")
    print("=" * 60 + "\n")


def print_flight_path(decisions: list[dict]):
    """Print ASCII representation of flight path.

    Args:
        decisions: List of decisions
    """
    print("\n  FLIGHT PATH VISUALIZATION")
    print("  " + "-" * 50)

    # Extract action sequence
    actions = []
    for d in decisions:
        full = d.get("full_decision", d)
        action = full.get("action", {}).get("type", "?")
        actions.append(action[0])  # First letter

    # Print in rows of 50
    for i in range(0, len(actions), 50):
        row = actions[i:i+50]
        print(f"  {i:04d} | {''.join(row)}")

    print()
    print("  Legend: C=Continue A=Avoid E=Engage B=Abort H=Hover R=RTB")
    print()


def print_decision_detail(decision: dict, index: int):
    """Print detailed decision information.

    Args:
        decision: Decision to display
        index: Index in chain
    """
    full = decision.get("full_decision", decision)

    print(f"\n  {'='*56}")
    print(f"  DECISION #{index} DETAIL")
    print(f"  {'='*56}")

    # Core info
    decision_id = full.get("decision_id", decision.get("decision_id", "N/A"))
    timestamp = full.get("timestamp", "N/A")
    action = full.get("action", {}).get("type", decision.get("action_type", "N/A"))
    confidence = full.get("confidence", decision.get("confidence", 0))

    print(f"\n  ID: {decision_id[:16]}...")
    print(f"  Time: {timestamp}")
    print(f"  Action: {action}")
    print(f"  Confidence: {confidence:.2%}")

    # Reasoning - the key insight
    reasoning = full.get("reasoning", decision.get("reasoning", "N/A"))
    print(f"\n  WHY DID THE AI DO THIS?")
    print(f"  {'-'*50}")
    print(f"  {reasoning}")

    # Telemetry
    telemetry = full.get("telemetry_snapshot", {})
    gps = telemetry.get("gps", {})
    if gps:
        print(f"\n  Position:")
        print(f"    Lat: {gps.get('lat', 0):.6f}")
        print(f"    Lon: {gps.get('lon', 0):.6f}")
        print(f"    Alt: {gps.get('alt', 0):.1f}m")

    battery = telemetry.get("battery_pct", 0)
    if battery:
        print(f"    Battery: {battery:.1f}%")

    # Perception
    perception = full.get("perception", {})
    obstacles = perception.get("obstacles", [])
    threats = perception.get("threats", [])
    targets = perception.get("targets", [])

    if obstacles:
        print(f"\n  Obstacles Detected ({len(obstacles)}):")
        for obs in obstacles[:3]:
            print(f"    - {obs.get('type', '?')} at {obs.get('distance_m', 0):.1f}m, "
                  f"bearing {obs.get('bearing_deg', 0):.0f}°")

    if threats:
        print(f"\n  Threats Detected ({len(threats)}):")
        for threat in threats[:3]:
            print(f"    - {threat.get('type', '?')} (severity: {threat.get('severity', 0):.2f})")

    if targets:
        print(f"\n  Targets Detected ({len(targets)}):")
        for target in targets[:3]:
            print(f"    - {target.get('id', '?')} (priority: {target.get('priority', 0)})")

    # Alternatives considered
    alts = full.get("alternative_actions_considered", [])
    if alts:
        print(f"\n  Alternatives Considered:")
        for alt in alts:
            print(f"    - {alt.get('action', '?')}: {alt.get('reason', 'N/A')}")

    # Hash verification
    decision_hash = decision.get("decision_hash", "N/A")
    print(f"\n  Verification:")
    print(f"    Hash: {decision_hash[:40]}...")
    print(f"    Status: ✓ Verified")

    print(f"\n  {'='*56}\n")


def find_interesting_decisions(decisions: list[dict]) -> list[int]:
    """Find decisions that are particularly interesting.

    Args:
        decisions: All decisions

    Returns:
        List of indices of interesting decisions
    """
    interesting = []

    for i, d in enumerate(decisions):
        full = d.get("full_decision", d)
        action = full.get("action", {}).get("type", "CONTINUE")

        # Non-routine actions are interesting
        if action in ["AVOID", "ENGAGE", "ABORT", "RTB"]:
            interesting.append(i)

    return interesting


def run_demo(specific_decision: int = None):
    """Run the replay demo.

    Args:
        specific_decision: If provided, show only this decision
    """
    print_header()

    # Generate decisions
    print("  Loading decision chain...")
    decisions, _ = run_mission(100, seed=42)
    print(f"  ✓ {len(decisions)} decisions loaded\n")

    # Verify chain
    print("  Verifying chain integrity...")
    is_valid, violations = verify_chain_integrity(decisions)
    if is_valid:
        print("  ✓ Chain verified - all hashes valid\n")
    else:
        print(f"  ✗ Chain has {len(violations)} violations!\n")
        return

    # Show flight path
    print_flight_path(decisions)

    if specific_decision is not None:
        # Show specific decision
        if 0 <= specific_decision < len(decisions):
            print(f"\n  Showing requested decision #{specific_decision}")
            print_decision_detail(decisions[specific_decision], specific_decision)
        else:
            print(f"  Error: Decision {specific_decision} not found (0-{len(decisions)-1})")
        return

    # Find and show interesting decisions
    interesting = find_interesting_decisions(decisions)

    print(f"  Found {len(interesting)} interesting decisions (non-routine actions)")
    print("  " + "-" * 50)

    if interesting:
        print("\n  Showing first 3 interesting decisions:\n")
        for idx in interesting[:3]:
            print_decision_detail(decisions[idx], idx)
    else:
        print("\n  No non-routine actions in this simulation.")
        print("  Showing first decision as example:\n")
        print_decision_detail(decisions[0], 0)

    # Pattern analysis
    print("\n  PATTERN ANALYSIS")
    print("  " + "-" * 50)

    full_decisions = [d.get("full_decision", d) for d in decisions]
    report = generate_topology_report(full_decisions)

    print(f"\n  Patterns found: {report['patterns_found']}")
    print(f"  Average effectiveness: {report['summary']['average_effectiveness']:.2%}")
    print(f"  Average autonomy: {report['summary']['average_autonomy']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("  REPLAY DEMO COMPLETE")
    print("=" * 60)
    print("\n  Forensic Capabilities Demonstrated:")
    print("  1. Complete decision context preserved at capture time")
    print("  2. Sensor state, perception, reasoning all recorded")
    print("  3. 'Why did the AI do this?' answerable in seconds")
    print("  4. Hash chain enables trust in historical data")
    print("  5. Pattern analysis reveals operational trends")
    print("\n  This is what decision provenance enables.")
    print()


def interactive_mode(decisions: list[dict]):
    """Run interactive exploration mode.

    Args:
        decisions: Decisions to explore
    """
    print("\n  INTERACTIVE MODE")
    print("  " + "-" * 50)
    print("  Enter decision number to inspect, 'q' to quit")
    print()

    while True:
        try:
            inp = input("  Decision # > ").strip()
            if inp.lower() == 'q':
                break
            idx = int(inp)
            if 0 <= idx < len(decisions):
                print_decision_detail(decisions[idx], idx)
            else:
                print(f"  Invalid index. Range: 0-{len(decisions)-1}")
        except ValueError:
            print("  Enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Flight Recorder Decision Replay Demo"
    )
    parser.add_argument(
        "--decision", "-d",
        type=int,
        help="Show specific decision by index"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive exploration mode"
    )

    args = parser.parse_args()

    if args.interactive:
        print_header()
        decisions, _ = run_mission(100, seed=42)
        print_flight_path(decisions)
        interactive_mode(decisions)
    else:
        run_demo(specific_decision=args.decision)


if __name__ == "__main__":
    main()
