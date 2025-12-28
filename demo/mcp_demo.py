"""MCP Server Demo (v2.2)

Demonstrates the Model Context Protocol server interface for
external AI orchestrator integration.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp import start_server, stop_server, get_server
from src.mcp.server import MCPRequest, handle_mcp_request
from src.core import emit_receipt


def demo_tools():
    """Demonstrate MCP tool invocations."""
    print("\n=== MCP Tools Demo ===\n")

    # Start server
    server = start_server()
    print(f"MCP Server started with {len(server.get_tools())} tools\n")

    # List available tools
    print("Available tools:")
    for tool in server.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    print()

    # Demo: query_decisions
    print("1. Querying decisions...")
    result = handle_mcp_request(
        request_type="tool",
        name="query_decisions",
        inputs={"limit": 5},
        caller_id="demo_script"
    )
    print(f"   Result: {json.dumps(result['outputs'], indent=2)[:200]}...")
    print()

    # Demo: get_audit_trail
    print("2. Getting audit trail summary...")
    result = handle_mcp_request(
        request_type="tool",
        name="get_audit_trail",
        inputs={"report_type": "summary"},
        caller_id="demo_script"
    )
    print(f"   Result: {json.dumps(result['outputs'], indent=2)[:300]}...")
    print()

    # Demo: verify_chain
    print("3. Verifying chain integrity...")
    result = handle_mcp_request(
        request_type="tool",
        name="verify_chain",
        inputs={},
        caller_id="demo_script"
    )
    print(f"   Result: {json.dumps(result['outputs'], indent=2)}")
    print()

    return server


def demo_resources():
    """Demonstrate MCP resource access."""
    print("\n=== MCP Resources Demo ===\n")

    server = get_server()

    # List available resources
    print("Available resources:")
    for resource in server.get_resources():
        print(f"  - {resource['uri']}: {resource['description']}")
    print()

    # Demo: decision stream
    print("1. Getting decision stream...")
    result = handle_mcp_request(
        request_type="resource",
        name="flight://decisions/stream",
        inputs={"limit": 3},
        caller_id="demo_script"
    )
    print(f"   Type: {result['outputs'].get('type')}")
    print(f"   Count: {result['outputs'].get('count')}")
    print()

    # Demo: entropy metrics
    print("2. Getting entropy metrics...")
    result = handle_mcp_request(
        request_type="resource",
        name="flight://metrics/entropy",
        inputs={},
        caller_id="demo_script"
    )
    print(f"   Available: {result['outputs'].get('available')}")
    if result['outputs'].get('available'):
        print(f"   Current: {result['outputs'].get('current')}")
    print()

    # Demo: active agents
    print("3. Getting active agents...")
    result = handle_mcp_request(
        request_type="resource",
        name="flight://agents/active",
        inputs={},
        caller_id="demo_script"
    )
    print(f"   Count: {result['outputs'].get('count')}")
    print()

    # Demo: graduated patterns
    print("4. Getting graduated patterns...")
    result = handle_mcp_request(
        request_type="resource",
        name="flight://patterns/graduated",
        inputs={},
        caller_id="demo_script"
    )
    print(f"   Count: {result['outputs'].get('count')}")
    print()


def demo_intervention():
    """Demonstrate recording a human intervention via MCP."""
    print("\n=== MCP Intervention Demo ===\n")

    # Simulate an intervention
    print("Recording human intervention...")
    result = handle_mcp_request(
        request_type="tool",
        name="inject_intervention",
        inputs={
            "decision_id": "demo-decision-001",
            "reason_code": "MODEL_ERROR",
            "correction": {
                "action": {"type": "AVOID"},
                "confidence": 0.95
            }
        },
        caller_id="demo_operator"
    )
    print(f"   Intervention ID: {result['outputs'].get('intervention_id')}")
    print(f"   Status: {result['outputs'].get('status')}")
    print()


def demo_spawn_investigator():
    """Demonstrate spawning an investigator agent via MCP."""
    print("\n=== MCP Agent Spawn Demo ===\n")

    print("Spawning investigator for anomaly...")
    result = handle_mcp_request(
        request_type="tool",
        name="spawn_investigator",
        inputs={
            "anomaly_id": "demo-anomaly-001",
            "investigation_type": "targeted"
        },
        caller_id="demo_orchestrator"
    )
    print(f"   Agent IDs: {result['outputs'].get('agent_ids')}")
    print(f"   Status: {result['outputs'].get('status')}")
    print()


def main():
    """Run all MCP demos."""
    print("=" * 60)
    print("AI Flight Recorder v2.2 - MCP Server Demo")
    print("=" * 60)

    try:
        # Run demos
        server = demo_tools()
        demo_resources()
        demo_intervention()
        demo_spawn_investigator()

        # Server stats
        print("\n=== Server Statistics ===\n")
        stats = server.get_stats()
        print(f"Running: {stats['running']}")
        print(f"Request count: {stats['request_count']}")
        print(f"Tools registered: {stats['tools_count']}")
        print(f"Resources registered: {stats['resources_count']}")

    finally:
        # Stop server
        stop_server()
        print("\nMCP Server stopped.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
