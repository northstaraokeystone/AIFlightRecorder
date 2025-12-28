#!/usr/bin/env python3
"""AI Flight Recorder CLI - v2.0

Main entry point for running the flight recorder system.

v2.0 New Commands:
    python cli.py spawn status       # Show active agents
    python cli.py spawn history      # Show spawn/prune/graduate events
    python cli.py gate check <id>    # Show gate decision
    python cli.py entropy status     # Current system entropy
    python cli.py monte status       # Monte Carlo stats

Legacy Commands:
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

# Ensure src and config are importable
sys.path.insert(0, str(Path(__file__).parent))

from src.core import dual_hash, emit_receipt, reset_receipt_counter, load_receipts
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


# =============================================================================
# v2.0 CLI COMMANDS
# =============================================================================

def cmd_spawn_status():
    """Show active agents, depth, TTL remaining."""
    from src.spawner.registry import get_registry
    from src.spawner.lifecycle import lifecycle_summary

    registry = get_registry()
    stats = registry.get_stats()
    summary = lifecycle_summary()

    print("\n=== AGENT SPAWN STATUS ===\n")
    print(f"Active Agents: {stats['active_count']}")
    print(f"Max Depth Used: {stats['max_depth_used']}")
    print(f"Population Headroom: {stats['population_headroom']}")

    print("\nBy Type:")
    for agent_type, count in stats["by_type"].items():
        print(f"  {agent_type}: {count}")

    print("\nBy Depth:")
    for depth, count in stats["by_depth"].items():
        print(f"  Level {depth}: {count}")

    # Show active agents with TTL
    active = registry.get_active()
    if active:
        print("\nActive Agents (top 10):")
        import time
        now = time.time()
        for agent in active[:10]:
            ttl_remaining = max(0, agent.expires_at - now)
            print(f"  {agent.agent_id[:8]}... [{agent.agent_type}] "
                  f"depth={agent.depth} ttl={ttl_remaining:.0f}s")


def cmd_spawn_history():
    """Show spawn/prune/graduate events from receipts."""
    receipts = load_receipts()

    spawn_receipts = [r for r in receipts if r.get("receipt_type") == "spawn"]
    prune_receipts = [r for r in receipts if r.get("receipt_type") == "pruning"]
    grad_receipts = [r for r in receipts if r.get("receipt_type") == "graduation"]

    print("\n=== SPAWN HISTORY ===\n")
    print(f"Total Spawns: {len(spawn_receipts)}")
    print(f"Total Prunes: {len(prune_receipts)}")
    print(f"Total Graduations: {len(grad_receipts)}")

    print("\nRecent Spawns (last 5):")
    for r in spawn_receipts[-5:]:
        agents = r.get("child_agents", [])
        print(f"  {r.get('ts', 'N/A')[:19]} - {len(agents)} agents, "
              f"trigger={r.get('trigger', 'N/A')}")


def cmd_spawn_patterns():
    """List graduated patterns."""
    from src.spawner.patterns import get_pattern_store

    store = get_pattern_store()
    patterns = store.get_most_effective(20)

    print("\n=== GRADUATED PATTERNS ===\n")
    print(f"Total Patterns: {store.count()}")
    print(f"Total Usage: {store.total_usage()}")

    if patterns:
        print("\nTop Patterns:")
        for p in patterns:
            print(f"  {p.pattern_id[:8]}... eff={p.effectiveness:.2f} "
                  f"uses={p.usage_count} domain={p.domain}")


def cmd_gate_check(decision_id: str):
    """Show gate decision for a specific decision."""
    receipts = load_receipts()

    gate_receipts = [r for r in receipts
                     if r.get("receipt_type") == "gate_decision"
                     and r.get("decision_id", "").startswith(decision_id)]

    if not gate_receipts:
        print(f"No gate decision found for decision {decision_id}")
        return

    r = gate_receipts[-1]
    print("\n=== GATE DECISION ===\n")
    print(f"Decision ID: {r.get('decision_id', 'N/A')}")
    print(f"Confidence: {r.get('confidence_score', 0):.2f}")
    print(f"Gate Tier: {r.get('gate_tier', 'N/A')}")
    print(f"Context Drift: {r.get('context_drift', 0):.2f}")
    print(f"Agents Spawned: {r.get('agents_spawned', [])}")


def cmd_gate_history():
    """Recent gate decisions."""
    receipts = load_receipts()

    gate_receipts = [r for r in receipts if r.get("receipt_type") == "gate_decision"]

    print("\n=== GATE HISTORY ===\n")
    print(f"Total Gate Decisions: {len(gate_receipts)}")

    # Count by tier
    by_tier = {}
    for r in gate_receipts:
        tier = r.get("gate_tier", "UNKNOWN")
        by_tier[tier] = by_tier.get(tier, 0) + 1

    print("\nBy Tier:")
    for tier, count in sorted(by_tier.items()):
        pct = count / len(gate_receipts) * 100 if gate_receipts else 0
        print(f"  {tier}: {count} ({pct:.1f}%)")


def cmd_gate_thresholds():
    """Current GREEN/YELLOW/RED thresholds."""
    from config.constants import GATE_GREEN_THRESHOLD, GATE_YELLOW_THRESHOLD

    print("\n=== GATE THRESHOLDS ===\n")
    print(f"GREEN: confidence >= {GATE_GREEN_THRESHOLD}")
    print(f"YELLOW: {GATE_YELLOW_THRESHOLD} <= confidence < {GATE_GREEN_THRESHOLD}")
    print(f"RED: confidence < {GATE_YELLOW_THRESHOLD}")


def cmd_entropy_status():
    """Current system entropy."""
    from src.compress import get_entropy_engine, entropy_trend

    engine = get_entropy_engine()
    history = engine.get_entropy_history()

    print("\n=== ENTROPY STATUS ===\n")
    if history:
        print(f"Current Entropy: {history[-1]:.3f} bits")
        print(f"Entropy Budget: {engine.get_budget():.3f} bits")

        trend = entropy_trend(history)
        print(f"Trend: {trend}")

        if len(history) >= 10:
            avg = sum(history[-10:]) / 10
            print(f"10-cycle Average: {avg:.3f} bits")
    else:
        print("No entropy data yet")


def cmd_entropy_conservation():
    """Validate thermodynamic constraints."""
    receipts = load_receipts()
    entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]

    valid_count = sum(1 for r in entropy_receipts if r.get("conservation_valid", False))
    total = len(entropy_receipts)

    print("\n=== ENTROPY CONSERVATION ===\n")
    if total > 0:
        print(f"Conservation Valid: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
        if valid_count < total:
            print("WARNING: Some cycles violated entropy conservation!")
    else:
        print("No entropy cycles recorded yet")


def cmd_monte_status():
    """Monte Carlo simulation stats."""
    receipts = load_receipts()
    monte_receipts = [r for r in receipts if r.get("receipt_type") == "monte_carlo"]

    print("\n=== MONTE CARLO STATUS ===\n")
    print(f"Total Simulations: {len(monte_receipts)}")

    if monte_receipts:
        variances = [r.get("variance", 0) for r in monte_receipts]
        latencies = [r.get("latency_ms", 0) for r in monte_receipts]
        stable_count = sum(1 for r in monte_receipts if r.get("is_stable", False))

        print(f"Average Variance: {sum(variances)/len(variances):.3f}")
        print(f"Average Latency: {sum(latencies)/len(latencies):.1f}ms")
        print(f"Stable Decisions: {stable_count}/{len(monte_receipts)} "
              f"({stable_count/len(monte_receipts)*100:.1f}%)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Flight Recorder v2.0 - Decision Provenance for Autonomous Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v2.0 Commands (use as: python cli.py <command> <subcommand>):
  spawn status              Show active agents
  spawn history             Show spawn/prune/graduate events
  spawn patterns            List graduated patterns
  gate check <id>           Show gate decision for decision ID
  gate history              Recent gate decisions
  gate thresholds           Current GREEN/YELLOW/RED thresholds
  entropy status            Current system entropy
  entropy conservation      Validate thermodynamic constraints
  monte status              Monte Carlo stats

Legacy Commands:
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

    # Add v2.0 subcommand support
    parser.add_argument(
        "command",
        nargs="?",
        help="v2.0 command (spawn, gate, entropy, monte)"
    )

    parser.add_argument(
        "subcommand",
        nargs="?",
        help="v2.0 subcommand (status, history, check, etc.)"
    )

    parser.add_argument(
        "arg",
        nargs="?",
        help="Optional argument for subcommand"
    )

    args = parser.parse_args()

    # Handle v2.0 commands first
    if args.command == "spawn":
        if args.subcommand == "status":
            cmd_spawn_status()
        elif args.subcommand == "history":
            cmd_spawn_history()
        elif args.subcommand == "patterns":
            cmd_spawn_patterns()
        elif args.subcommand == "kill" and args.arg:
            from src.spawner.prune import prune_agents
            result = prune_agents([args.arg], "manual")
            print(f"Terminated: {result.agents_terminated}")
        else:
            print("Usage: cli.py spawn [status|history|patterns|kill <id>]")
        return

    elif args.command == "gate":
        if args.subcommand == "check" and args.arg:
            cmd_gate_check(args.arg)
        elif args.subcommand == "history":
            cmd_gate_history()
        elif args.subcommand == "thresholds":
            cmd_gate_thresholds()
        else:
            print("Usage: cli.py gate [check <id>|history|thresholds]")
        return

    elif args.command == "entropy":
        if args.subcommand == "status":
            cmd_entropy_status()
        elif args.subcommand == "budget":
            from src.compress import entropy_budget
            print(f"Entropy Budget: {entropy_budget():.3f} bits")
        elif args.subcommand == "conservation":
            cmd_entropy_conservation()
        else:
            print("Usage: cli.py entropy [status|budget|conservation]")
        return

    elif args.command == "monte":
        if args.subcommand == "status":
            cmd_monte_status()
        elif args.subcommand == "variance":
            cmd_monte_status()  # Same output for now
        else:
            print("Usage: cli.py monte [status|variance]")
        return

    # Handle legacy commands
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
        receipts = load_receipts()
        decisions = [r for r in receipts if r.get("receipt_type") == "decision"]
        report = generate_integrity_report(decisions)
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
