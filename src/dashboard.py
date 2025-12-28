"""Streamlit Dashboard for Real-Time Decision Visualization - v2.0

Provides:
- Decision timeline with color coding
- 2D flight path visualization
- Tamper status indicator
- Compression health gauge
- Sync status display
- Decision detail inspection

v2.0 Panels:
- Agent Swarm View (active agents, depth, TTL)
- Confidence Gate Timeline (GREEN/YELLOW/RED bands)
- Entropy Health (system entropy gauge)
- Graduated Patterns (effectiveness leaderboard)
- HUNTER/SHEPHERD Status (immortal agent health)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Note: Streamlit imports are conditional for CLI compatibility
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .core import load_receipts
from .verify import verify_chain_integrity, generate_integrity_report
from .compress import build_baseline, detect_anomaly, CompressionBaseline
from .topology import generate_topology_report


def load_decisions(file_path: Optional[str] = None) -> list[dict]:
    """Load decisions from receipts file.

    Args:
        file_path: Optional path to receipts file

    Returns:
        List of decision receipts
    """
    receipts = load_receipts(Path(file_path) if file_path else None)
    return [r for r in receipts if r.get("receipt_type") == "decision"]


def get_verification_status(decisions: list[dict]) -> dict:
    """Get verification status for decisions.

    Args:
        decisions: List of decisions

    Returns:
        Verification status dict
    """
    is_valid, violations = verify_chain_integrity(decisions)
    return {
        "is_valid": is_valid,
        "status": "VERIFIED" if is_valid else "FAILED",
        "violations": len(violations),
        "decisions_checked": len(decisions)
    }


def get_sync_status() -> dict:
    """Get current sync status.

    Returns:
        Sync status dict
    """
    # Would load from sync state file in production
    return {
        "status": "SYNCED",
        "last_sync": datetime.now().isoformat(),
        "pending": 0
    }


def run_dashboard():
    """Run the Streamlit dashboard."""
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(
        page_title="AI Flight Recorder v2.0",
        page_icon="🛩️",
        layout="wide"
    )

    st.title("🛩️ AI Flight Recorder Dashboard v2.0")

    # Load data
    decisions = load_decisions()
    receipts = load_receipts()

    # Top status bar
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        verification = get_verification_status(decisions)
        if verification["is_valid"]:
            st.success(f"✓ INTEGRITY: VERIFIED ({verification['decisions_checked']} decisions)")
        else:
            st.error(f"✗ INTEGRITY: FAILED ({verification['violations']} violations)")

    with col2:
        sync = get_sync_status()
        if sync["status"] == "SYNCED":
            st.success(f"✓ SYNC: ANCHORED")
        else:
            st.warning(f"⏳ SYNC: {sync['pending']} pending")

    with col3:
        st.info(f"📊 Decisions: {len(decisions)}")

    with col4:
        if decisions:
            latest = decisions[-1]
            action = latest.get("action_type", "N/A")
            st.info(f"🎯 Latest: {action}")

    st.divider()

    # Main content - v2.0 tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Timeline", "Flight Path", "Integrity", "Analysis",
        "Agent Swarm", "v2.0 Status"
    ])

    with tab1:
        render_decision_timeline(decisions)

    with tab2:
        render_map_view(decisions)

    with tab3:
        render_tamper_status(decisions)

    with tab4:
        render_analysis(decisions)

    with tab5:
        render_agent_swarm(receipts)

    with tab6:
        render_v2_status(receipts)

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")

        if st.button("🔄 Refresh"):
            st.rerun()

        if st.button("🔍 Verify Chain"):
            with st.spinner("Verifying..."):
                report = generate_integrity_report(decisions)
                st.json(report["summary"])

        st.divider()

        st.header("Filters")
        action_filter = st.multiselect(
            "Action Types",
            ["CONTINUE", "AVOID", "ENGAGE", "ABORT", "HOVER", "RTB"],
            default=["CONTINUE", "AVOID", "ENGAGE"]
        )

        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.5)

        st.divider()

        # v2.0 Quick Stats
        st.header("v2.0 Quick Stats")
        render_sidebar_v2_stats(receipts)


def render_decision_timeline(decisions: list[dict]):
    """Render decision timeline visualization.

    Args:
        decisions: List of decisions
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Decision Timeline")

    if not decisions:
        st.info("No decisions to display")
        return

    # Create timeline data
    timeline_data = []
    for i, d in enumerate(decisions[-100:]):  # Last 100
        action = d.get("action_type", "UNKNOWN")
        confidence = d.get("confidence", 0)
        color = {
            "CONTINUE": "🟢",
            "AVOID": "🟡",
            "ENGAGE": "🔵",
            "ABORT": "🔴",
            "HOVER": "⚪",
            "RTB": "🟠"
        }.get(action, "⚫")

        timeline_data.append({
            "index": i,
            "action": action,
            "icon": color,
            "confidence": confidence,
            "decision_id": d.get("decision_id", "")[:8]
        })

    # Display as horizontal sequence
    cols = st.columns(min(20, len(timeline_data)))
    for i, (col, data) in enumerate(zip(cols, timeline_data[-20:])):
        with col:
            st.write(data["icon"])
            if st.button(f"{i}", key=f"btn_{data['decision_id']}"):
                st.session_state["selected_decision"] = decisions[-(20-i)]

    # Decision detail
    if "selected_decision" in st.session_state:
        st.subheader("Decision Detail")
        d = st.session_state["selected_decision"]
        render_decision_detail(d)


def render_map_view(decisions: list[dict]):
    """Render 2D flight path visualization.

    Args:
        decisions: List of decisions
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Flight Path")

    if not decisions:
        st.info("No flight data to display")
        return

    # Extract GPS coordinates
    coords = []
    for d in decisions:
        full = d.get("full_decision", {})
        telemetry = full.get("telemetry_snapshot", {})
        gps = telemetry.get("gps", {})
        if gps:
            coords.append({
                "lat": gps.get("lat", 0),
                "lon": gps.get("lon", 0),
                "action": d.get("action_type", "CONTINUE")
            })

    if coords:
        # Create a simple plot
        import pandas as pd

        df = pd.DataFrame(coords)

        # Use scatter plot for path
        st.scatter_chart(
            df,
            x="lon",
            y="lat",
            color="action"
        )
    else:
        st.info("No GPS data in decisions")

    # Summary stats
    if decisions:
        col1, col2, col3 = st.columns(3)
        with col1:
            avoid_count = sum(1 for d in decisions if d.get("action_type") == "AVOID")
            st.metric("Avoidance Maneuvers", avoid_count)
        with col2:
            avg_conf = sum(d.get("confidence", 0) for d in decisions) / len(decisions)
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        with col3:
            st.metric("Total Decisions", len(decisions))


def render_tamper_status(decisions: list[dict]):
    """Render integrity verification status.

    Args:
        decisions: List of decisions
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Chain Integrity Verification")

    if not decisions:
        st.info("No decisions to verify")
        return

    col1, col2 = st.columns(2)

    with col1:
        is_valid, violations = verify_chain_integrity(decisions)

        if is_valid:
            st.success("### ✓ CHAIN VERIFIED")
            st.write("All decisions cryptographically verified.")
            st.write("Hash chain intact. Merkle tree consistent.")
        else:
            st.error("### ✗ INTEGRITY FAILURE")
            st.write(f"Found {len(violations)} violations:")
            for v in violations[:5]:  # Show first 5
                st.write(f"- Position {v.position}: {v.violation_type}")

    with col2:
        st.write("### Verification Details")
        st.write(f"- Decisions checked: {len(decisions)}")
        st.write(f"- Chain valid: {'Yes' if is_valid else 'No'}")
        st.write(f"- Violations: {len(violations)}")

        if st.button("Run Full Audit"):
            with st.spinner("Running comprehensive audit..."):
                report = generate_integrity_report(decisions)
                st.json(report)


def render_compression_health(baseline: Optional[CompressionBaseline],
                               current: float):
    """Render compression-based anomaly gauge.

    Args:
        baseline: Compression baseline
        current: Current compression ratio
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Compression Health")

    if baseline is None:
        st.info("Baseline not established")
        return

    delta = abs(current - baseline.mean_ratio) / baseline.mean_ratio
    is_healthy = delta < 0.15

    if is_healthy:
        st.success(f"✓ Normal: {current:.2%} ratio (baseline: {baseline.mean_ratio:.2%})")
    else:
        st.warning(f"⚠ Anomaly: {current:.2%} ratio deviates from baseline {baseline.mean_ratio:.2%}")

    # Gauge visualization
    st.progress(min(1.0, current))


def render_sync_status(sync_state: dict):
    """Render chain-of-custody sync indicator.

    Args:
        sync_state: Sync state dict
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Sync Status")

    is_synced = sync_state.get("pending_decisions", 0) == 0

    if is_synced:
        st.success("✓ SYNCED - Chain of custody verified")
    else:
        pending = sync_state.get("pending_decisions", 0)
        st.warning(f"⏳ {pending} decisions pending sync")

    st.write(f"Last sync: {sync_state.get('last_sync_time', 'Never')}")
    st.write(f"Local tree size: {sync_state.get('local_tree_size', 0)}")


def render_decision_detail(decision: dict):
    """Render full decision context on click.

    Args:
        decision: Decision to display
    """
    if not HAS_STREAMLIT:
        print(json.dumps(decision, indent=2))
        return

    full = decision.get("full_decision", decision)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Decision ID:** {decision.get('decision_id', 'N/A')[:16]}...")
        st.write(f"**Timestamp:** {full.get('timestamp', 'N/A')}")
        st.write(f"**Action:** {decision.get('action_type', 'N/A')}")
        st.write(f"**Confidence:** {decision.get('confidence', 0):.2%}")

    with col2:
        st.write(f"**Model:** {full.get('model_version', 'N/A')}")
        st.write(f"**Cycle:** {full.get('cycle_number', 'N/A')}")

    st.write("**Reasoning:**")
    st.info(decision.get("reasoning", full.get("reasoning", "N/A")))

    # Hash verification
    hash_status = "✓" if decision.get("decision_hash") else "?"
    st.write(f"**Hash:** {hash_status} {decision.get('decision_hash', 'N/A')[:40]}...")

    # Show alternatives
    alts = full.get("alternative_actions_considered", [])
    if alts:
        st.write("**Alternatives Considered:**")
        for alt in alts:
            st.write(f"- {alt.get('action', 'N/A')}: {alt.get('reason', 'N/A')}")


def render_analysis(decisions: list[dict]):
    """Render topology and pattern analysis.

    Args:
        decisions: List of decisions
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Pattern Analysis")

    if len(decisions) < 10:
        st.info("Need at least 10 decisions for analysis")
        return

    # Get full decisions for analysis
    full_decisions = []
    for d in decisions:
        if "full_decision" in d:
            full_decisions.append(d["full_decision"])
        else:
            full_decisions.append(d)

    report = generate_topology_report(full_decisions)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Patterns Found", report["patterns_found"])
        st.metric("Avg Effectiveness", f"{report['summary']['average_effectiveness']:.2%}")

    with col2:
        st.metric("Avg Autonomy", f"{report['summary']['average_autonomy']:.2%}")
        at_velocity = report["summary"]["patterns_at_escape_velocity"]
        st.metric("At Escape Velocity", at_velocity)

    # Pattern details
    st.write("### Patterns")
    for p in report.get("patterns", []):
        with st.expander(f"{p['pattern_type']} - {p['topology']}"):
            st.write(f"- Effectiveness: {p['metrics']['effectiveness']:.2%}")
            st.write(f"- Autonomy: {p['metrics']['autonomy_score']:.2%}")
            st.write(f"- Transfer: {p['metrics']['transfer_score']:.2%}")
            st.write(f"- Can Graduate: {'Yes' if p['can_graduate'] else 'No'}")


# =============================================================================
# v2.0 DASHBOARD PANELS
# =============================================================================

def render_agent_swarm(receipts: list[dict]):
    """Render Agent Swarm View - active agents, depth, TTL.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Agent Swarm View")

    # Get spawn and prune receipts
    spawn_receipts = [r for r in receipts if r.get("receipt_type") == "spawn"]
    prune_receipts = [r for r in receipts if r.get("receipt_type") == "pruning"]
    grad_receipts = [r for r in receipts if r.get("receipt_type") == "graduation"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Spawns", len(spawn_receipts))

    with col2:
        st.metric("Total Prunes", len(prune_receipts))

    with col3:
        st.metric("Graduations", len(grad_receipts))

    with col4:
        # Estimate active agents (spawns - prunes)
        total_spawned = sum(len(r.get("child_agents", [])) for r in spawn_receipts)
        total_pruned = sum(r.get("agents_terminated", 0) for r in prune_receipts)
        active_estimate = max(0, total_spawned - total_pruned)
        st.metric("Active (est.)", active_estimate)

    st.divider()

    # Agent spawn timeline
    st.write("### Spawn Timeline")
    if spawn_receipts:
        spawn_data = []
        for r in spawn_receipts[-20:]:
            spawn_data.append({
                "timestamp": r.get("ts", "")[:19],
                "trigger": r.get("trigger", "unknown"),
                "agents": len(r.get("child_agents", [])),
                "depth": r.get("depth_level", 0),
                "confidence": r.get("confidence_at_spawn", 0)
            })

        import pandas as pd
        df = pd.DataFrame(spawn_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No spawn events recorded yet")

    # Depth distribution
    st.write("### Depth Distribution")
    if spawn_receipts:
        depths = {}
        for r in spawn_receipts:
            depth = r.get("depth_level", 0)
            depths[depth] = depths.get(depth, 0) + len(r.get("child_agents", []))

        import pandas as pd
        depth_df = pd.DataFrame([
            {"Depth": f"Level {k}", "Agents": v}
            for k, v in sorted(depths.items())
        ])
        st.bar_chart(depth_df.set_index("Depth"))
    else:
        st.info("No depth data available")


def render_v2_status(receipts: list[dict]):
    """Render v2.0 status panels.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    # Sub-tabs for v2.0 features
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Confidence Gate", "Entropy Health", "HUNTER/SHEPHERD", "Graduated Patterns"
    ])

    with subtab1:
        render_gate_timeline(receipts)

    with subtab2:
        render_entropy_health(receipts)

    with subtab3:
        render_immortal_status(receipts)

    with subtab4:
        render_graduated_patterns(receipts)


def render_gate_timeline(receipts: list[dict]):
    """Render Confidence Gate Timeline with GREEN/YELLOW/RED bands.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Confidence Gate Timeline")

    gate_receipts = [r for r in receipts if r.get("receipt_type") == "gate_decision"]

    if not gate_receipts:
        st.info("No gate decisions recorded yet")
        return

    # Count by tier
    tier_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for r in gate_receipts:
        tier = r.get("gate_tier", "UNKNOWN")
        if tier in tier_counts:
            tier_counts[tier] += 1

    col1, col2, col3 = st.columns(3)

    with col1:
        pct = tier_counts["GREEN"] / len(gate_receipts) * 100 if gate_receipts else 0
        st.metric("GREEN (≥0.9)", f"{tier_counts['GREEN']} ({pct:.1f}%)")

    with col2:
        pct = tier_counts["YELLOW"] / len(gate_receipts) * 100 if gate_receipts else 0
        st.metric("YELLOW (0.7-0.9)", f"{tier_counts['YELLOW']} ({pct:.1f}%)")

    with col3:
        pct = tier_counts["RED"] / len(gate_receipts) * 100 if gate_receipts else 0
        st.metric("RED (<0.7)", f"{tier_counts['RED']} ({pct:.1f}%)")

    # Timeline visualization
    st.write("### Gate Decisions (last 50)")
    gate_data = []
    for r in gate_receipts[-50:]:
        tier = r.get("gate_tier", "UNKNOWN")
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(tier, "⚫")
        gate_data.append({
            "timestamp": r.get("ts", "")[:19],
            "tier": f"{icon} {tier}",
            "confidence": r.get("confidence_score", 0),
            "drift": r.get("context_drift", 0),
            "spawned": len(r.get("agents_spawned", []))
        })

    import pandas as pd
    df = pd.DataFrame(gate_data)
    st.dataframe(df, use_container_width=True)

    # Confidence trend
    st.write("### Confidence Trend")
    if len(gate_receipts) >= 2:
        confidence_vals = [r.get("confidence_score", 0) for r in gate_receipts[-100:]]
        import pandas as pd
        trend_df = pd.DataFrame({"Confidence": confidence_vals})
        st.line_chart(trend_df)


def render_entropy_health(receipts: list[dict]):
    """Render Entropy Health gauge.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Entropy Health")

    entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]

    if not entropy_receipts:
        st.info("No entropy data recorded yet")
        return

    latest = entropy_receipts[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        entropy = latest.get("system_entropy_bits", 0)
        st.metric("System Entropy", f"{entropy:.3f} bits")

    with col2:
        delta = latest.get("entropy_delta", 0)
        st.metric("Entropy Delta", f"{delta:+.4f}")

    with col3:
        conservation = latest.get("conservation_valid", False)
        if conservation:
            st.success("✓ Conservation Valid")
        else:
            st.error("✗ Conservation Violated!")

    # Conservation check history
    valid_count = sum(1 for r in entropy_receipts if r.get("conservation_valid", False))
    total = len(entropy_receipts)
    pct = valid_count / total * 100 if total else 0

    st.write(f"### Conservation History: {valid_count}/{total} ({pct:.1f}%)")

    if pct < 100:
        st.warning("Some cycles violated entropy conservation!")

    # Entropy trend
    st.write("### Entropy Trend")
    if len(entropy_receipts) >= 2:
        entropy_vals = [r.get("system_entropy_bits", 0) for r in entropy_receipts[-100:]]
        import pandas as pd
        trend_df = pd.DataFrame({"Entropy (bits)": entropy_vals})
        st.line_chart(trend_df)


def render_immortal_status(receipts: list[dict]):
    """Render HUNTER and SHEPHERD status.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Immortal Agents Status")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### HUNTER (Proprioception)")

        anomaly_receipts = [r for r in receipts if r.get("receipt_type") == "anomaly_alert"]

        if anomaly_receipts:
            st.success(f"✓ HUNTER Active - {len(anomaly_receipts)} alerts")

            # Count by type
            by_type = {}
            for r in anomaly_receipts:
                atype = r.get("anomaly_type", "unknown")
                by_type[atype] = by_type.get(atype, 0) + 1

            st.write("**Anomalies by Type:**")
            for atype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                st.write(f"- {atype}: {count}")

            # Recent alerts
            st.write("**Recent Alerts:**")
            for r in anomaly_receipts[-5:]:
                severity = r.get("severity", 0)
                icon = "🔴" if severity > 0.7 else "🟡" if severity > 0.4 else "🟢"
                st.write(f"{icon} {r.get('anomaly_type', 'N/A')}: {r.get('description', 'N/A')[:50]}...")
        else:
            st.info("HUNTER: No anomalies detected yet")

    with col2:
        st.write("### SHEPHERD (Homeostasis)")

        proposal_receipts = [r for r in receipts if r.get("receipt_type") == "remediation_proposal"]
        remediation_receipts = [r for r in receipts if r.get("receipt_type") == "remediation"]
        declined_receipts = [r for r in receipts if r.get("receipt_type") == "remediation_declined"]

        total_proposals = len(proposal_receipts)
        total_executed = len(remediation_receipts)
        total_declined = len(declined_receipts)

        if total_proposals > 0:
            st.success(f"✓ SHEPHERD Active - {total_proposals} proposals")

            st.write(f"**Executed:** {total_executed}")
            st.write(f"**Declined:** {total_declined}")
            st.write(f"**Pending:** {total_proposals - total_executed - total_declined}")

            # Success rate
            if total_executed > 0:
                success_count = sum(1 for r in remediation_receipts if r.get("all_succeeded", False))
                success_rate = success_count / total_executed * 100
                st.write(f"**Success Rate:** {success_rate:.1f}%")

            # Recent remediations
            st.write("**Recent Remediations:**")
            for r in remediation_receipts[-5:]:
                status = "✓" if r.get("all_succeeded", False) else "✗"
                st.write(f"{status} Plan {r.get('plan_id', 'N/A')[:8]}... - {r.get('actions_executed', 0)} actions")
        else:
            st.info("SHEPHERD: No remediations proposed yet")


def render_graduated_patterns(receipts: list[dict]):
    """Render Graduated Patterns leaderboard.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    st.subheader("Graduated Patterns")

    grad_receipts = [r for r in receipts if r.get("receipt_type") == "graduation"]

    if not grad_receipts:
        st.info("No patterns graduated yet")
        return

    st.metric("Total Graduations", len(grad_receipts))

    # Leaderboard
    st.write("### Effectiveness Leaderboard")

    pattern_data = []
    for r in grad_receipts:
        pattern_data.append({
            "pattern_id": r.get("solution_pattern_id", "N/A")[:12] + "...",
            "effectiveness": r.get("effectiveness", 0),
            "autonomy": r.get("autonomy_score", 0),
            "promoted_to": r.get("promoted_to", "N/A"),
            "domain": r.get("domain", "general")
        })

    # Sort by effectiveness
    pattern_data.sort(key=lambda x: -x["effectiveness"])

    import pandas as pd
    df = pd.DataFrame(pattern_data[:20])  # Top 20
    st.dataframe(df, use_container_width=True)

    # Domain distribution
    st.write("### By Domain")
    domains = {}
    for r in grad_receipts:
        domain = r.get("domain", "general")
        domains[domain] = domains.get(domain, 0) + 1

    import pandas as pd
    domain_df = pd.DataFrame([
        {"Domain": k, "Patterns": v}
        for k, v in sorted(domains.items(), key=lambda x: -x[1])
    ])
    st.bar_chart(domain_df.set_index("Domain"))


def render_sidebar_v2_stats(receipts: list[dict]):
    """Render v2.0 quick stats in sidebar.

    Args:
        receipts: All receipts
    """
    if not HAS_STREAMLIT:
        return

    # Gate stats
    gate_receipts = [r for r in receipts if r.get("receipt_type") == "gate_decision"]
    if gate_receipts:
        green_count = sum(1 for r in gate_receipts if r.get("gate_tier") == "GREEN")
        pct = green_count / len(gate_receipts) * 100
        st.write(f"🟢 GREEN Rate: {pct:.1f}%")

    # Entropy
    entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]
    if entropy_receipts:
        latest = entropy_receipts[-1]
        st.write(f"📊 Entropy: {latest.get('system_entropy_bits', 0):.2f} bits")

    # Monte Carlo
    monte_receipts = [r for r in receipts if r.get("receipt_type") == "monte_carlo"]
    if monte_receipts:
        stable_count = sum(1 for r in monte_receipts if r.get("is_stable", False))
        pct = stable_count / len(monte_receipts) * 100
        st.write(f"🎲 MC Stable: {pct:.1f}%")

    # Wounds
    wound_receipts = [r for r in receipts if r.get("receipt_type") == "wound"]
    if wound_receipts:
        st.write(f"🩹 Wounds: {len(wound_receipts)}")

    # Spawns
    spawn_receipts = [r for r in receipts if r.get("receipt_type") == "spawn"]
    if spawn_receipts:
        total = sum(len(r.get("child_agents", [])) for r in spawn_receipts)
        st.write(f"🌱 Spawned: {total}")


# CLI mode for non-Streamlit environments
def print_dashboard_summary(decisions: list[dict]):
    """Print dashboard summary to console.

    Args:
        decisions: List of decisions
    """
    print("\n" + "=" * 60)
    print("  AI FLIGHT RECORDER v2.0 - Dashboard Summary")
    print("=" * 60)

    # Verification
    is_valid, violations = verify_chain_integrity(decisions)
    status = "✓ VERIFIED" if is_valid else f"✗ FAILED ({len(violations)} violations)"
    print(f"\n  Integrity Status: {status}")
    print(f"  Decisions: {len(decisions)}")

    # Action breakdown
    actions = {}
    for d in decisions:
        action = d.get("action_type", "UNKNOWN")
        actions[action] = actions.get(action, 0) + 1

    print("\n  Action Breakdown:")
    for action, count in sorted(actions.items(), key=lambda x: -x[1]):
        pct = count / len(decisions) * 100 if decisions else 0
        print(f"    {action}: {count} ({pct:.1f}%)")

    # Latest decision
    if decisions:
        latest = decisions[-1]
        print(f"\n  Latest Decision:")
        print(f"    Action: {latest.get('action_type', 'N/A')}")
        print(f"    Confidence: {latest.get('confidence', 0):.2%}")
        print(f"    Reasoning: {latest.get('reasoning', 'N/A')[:50]}...")

    # v2.0 stats
    receipts = load_receipts()

    gate_receipts = [r for r in receipts if r.get("receipt_type") == "gate_decision"]
    spawn_receipts = [r for r in receipts if r.get("receipt_type") == "spawn"]
    entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]

    print("\n  v2.0 Stats:")
    print(f"    Gate Decisions: {len(gate_receipts)}")
    print(f"    Agent Spawns: {len(spawn_receipts)}")
    print(f"    Entropy Records: {len(entropy_receipts)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if HAS_STREAMLIT:
        run_dashboard()
    else:
        decisions = load_decisions()
        print_dashboard_summary(decisions)
