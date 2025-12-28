"""Scenario Results Reporting

Generates human-readable and machine-parseable reports
from simulation results.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .sim import SimResult


def format_result_summary(result: SimResult) -> str:
    """Format a single result as text summary.

    Args:
        result: Simulation result

    Returns:
        Formatted string
    """
    status = "✓ PASS" if result.success else "✗ FAIL"

    lines = [
        f"\n{'='*60}",
        f"  {result.config.name}: {status}",
        f"{'='*60}",
        f"  Description: {result.config.description}",
        f"  Cycles: {result.config.n_cycles}",
        f"  Duration: {result.duration_ms:.1f}ms",
        ""
    ]

    # Metrics
    if result.metrics:
        lines.append("  Metrics:")
        for key, value in result.metrics.items():
            if key == "cycle_times":
                continue  # Skip raw times
            if isinstance(value, float):
                lines.append(f"    {key}: {value:.4f}")
            else:
                lines.append(f"    {key}: {value}")

    # Violations
    if result.state.violations:
        lines.append(f"\n  Violations ({len(result.state.violations)}):")
        for v in result.state.violations[:5]:
            lines.append(f"    - {v.get('type', 'unknown')}: {v}")
        if len(result.state.violations) > 5:
            lines.append(f"    ... and {len(result.state.violations) - 5} more")

    # Error
    if result.state.error:
        lines.append(f"\n  Error: {result.state.error}")

    lines.append("")
    return "\n".join(lines)


def format_all_results(results: dict) -> str:
    """Format all scenario results.

    Args:
        results: Results from run_all_scenarios

    Returns:
        Formatted string
    """
    lines = [
        "\n" + "=" * 60,
        "  AI FLIGHT RECORDER - Validation Report",
        "=" * 60,
        f"  Timestamp: {results.get('timestamp', 'N/A')}",
        f"  Overall: {'✓ ALL PASSED' if results.get('all_passed') else '✗ SOME FAILED'}",
        ""
    ]

    # Summary table
    lines.append("  Scenario Results:")
    lines.append("  " + "-" * 50)

    for name, data in results.get("scenarios", {}).items():
        status = "✓" if data["success"] else "✗"
        duration = data.get("duration_ms", 0)
        violations = len(data.get("violations", []))
        lines.append(f"    {status} {name:20} {duration:8.1f}ms  {violations} violations")

    lines.append("  " + "-" * 50)
    lines.append("")

    return "\n".join(lines)


def generate_json_report(results: dict, output_path: Optional[Path] = None) -> dict:
    """Generate JSON report.

    Args:
        results: Results from run_all_scenarios
        output_path: Optional path to write JSON

    Returns:
        Report dict
    """
    report = {
        "report_type": "validation_report",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "all_passed": results.get("all_passed", False),
            "total_scenarios": len(results.get("scenarios", {})),
            "passed": sum(1 for s in results.get("scenarios", {}).values() if s["success"]),
            "failed": sum(1 for s in results.get("scenarios", {}).values() if not s["success"])
        },
        "scenarios": {}
    }

    for name, data in results.get("scenarios", {}).items():
        report["scenarios"][name] = {
            "success": data["success"],
            "duration_ms": data.get("duration_ms", 0),
            "metrics": {k: v for k, v in data.get("metrics", {}).items() if k != "cycle_times"},
            "violations": data.get("violations", [])
        }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


def generate_markdown_report(results: dict) -> str:
    """Generate Markdown report.

    Args:
        results: Results from run_all_scenarios

    Returns:
        Markdown string
    """
    lines = [
        "# AI Flight Recorder Validation Report",
        "",
        f"**Generated:** {results.get('timestamp', 'N/A')}",
        "",
        "## Summary",
        "",
    ]

    if results.get("all_passed"):
        lines.append("**Status:** ✅ ALL SCENARIOS PASSED")
    else:
        lines.append("**Status:** ❌ SOME SCENARIOS FAILED")

    lines.extend([
        "",
        "## Scenario Results",
        "",
        "| Scenario | Status | Duration | Violations |",
        "|----------|--------|----------|------------|"
    ])

    for name, data in results.get("scenarios", {}).items():
        status = "✅ Pass" if data["success"] else "❌ Fail"
        duration = f"{data.get('duration_ms', 0):.1f}ms"
        violations = len(data.get("violations", []))
        lines.append(f"| {name} | {status} | {duration} | {violations} |")

    lines.extend([
        "",
        "## Detailed Metrics",
        ""
    ])

    for name, data in results.get("scenarios", {}).items():
        lines.append(f"### {name}")
        lines.append("")

        metrics = data.get("metrics", {})
        for key, value in metrics.items():
            if key == "cycle_times":
                continue
            if isinstance(value, float):
                lines.append(f"- **{key}:** {value:.4f}")
            else:
                lines.append(f"- **{key}:** {value}")

        if data.get("violations"):
            lines.append("")
            lines.append("**Violations:**")
            for v in data["violations"][:3]:
                lines.append(f"- {v.get('type', 'unknown')}")

        lines.append("")

    return "\n".join(lines)


def print_live_status(scenario_name: str, cycle: int, total: int, metrics: dict):
    """Print live status during simulation.

    Args:
        scenario_name: Name of current scenario
        cycle: Current cycle
        total: Total cycles
        metrics: Current metrics
    """
    pct = cycle / total * 100
    mem = metrics.get("peak_memory_mb", 0)
    avg_ms = metrics.get("avg_cycle_ms", 0)

    print(f"\r  {scenario_name}: {cycle}/{total} ({pct:.1f}%) | "
          f"Mem: {mem:.1f}MB | Avg: {avg_ms:.2f}ms", end="", flush=True)


def save_results(results: dict, output_dir: Path):
    """Save all report formats.

    Args:
        results: Results from run_all_scenarios
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    generate_json_report(results, output_dir / "validation_report.json")

    # Markdown
    md = generate_markdown_report(results)
    with open(output_dir / "validation_report.md", "w") as f:
        f.write(md)

    # Text summary
    txt = format_all_results(results)
    with open(output_dir / "validation_report.txt", "w") as f:
        f.write(txt)
