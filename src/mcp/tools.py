"""MCP Tools - Flight Recorder Capabilities (v2.2)

Tools exposed via MCP for external AI orchestrators.
"""

from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone

from ..core import emit_receipt, load_receipts
from ..verify import verify_chain_integrity


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

FLIGHT_RECORDER_TOOLS = [
    {
        "name": "verify_chain",
        "description": "Verify decision chain integrity for a time range",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "ISO8601 start time"},
                "end_time": {"type": "string", "description": "ISO8601 end time"}
            }
        }
    },
    {
        "name": "query_decisions",
        "description": "Search decision history with filters",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "description": "Filter by action type"},
                "confidence_min": {"type": "number", "description": "Minimum confidence"},
                "confidence_max": {"type": "number", "description": "Maximum confidence"},
                "limit": {"type": "integer", "description": "Maximum results"}
            }
        }
    },
    {
        "name": "get_audit_trail",
        "description": "Generate compliance audit report",
        "input_schema": {
            "type": "object",
            "properties": {
                "report_type": {"type": "string", "enum": ["summary", "detailed", "compliance"]},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"}
            }
        }
    },
    {
        "name": "inject_intervention",
        "description": "Record human override of decision",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision_id": {"type": "string", "description": "Decision to override"},
                "reason_code": {"type": "string", "description": "Intervention reason code"},
                "correction": {"type": "object", "description": "Corrected decision data"}
            },
            "required": ["decision_id", "reason_code", "correction"]
        }
    },
    {
        "name": "spawn_investigator",
        "description": "Manually trigger helper agent spawn for investigation",
        "input_schema": {
            "type": "object",
            "properties": {
                "anomaly_id": {"type": "string", "description": "Anomaly to investigate"},
                "investigation_type": {"type": "string", "enum": ["deep", "quick", "targeted"]}
            },
            "required": ["anomaly_id"]
        }
    }
]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

def verify_chain_tool(start_time: Optional[str] = None,
                      end_time: Optional[str] = None) -> dict:
    """Verify decision chain integrity.

    Args:
        start_time: Optional start time filter
        end_time: Optional end time filter

    Returns:
        Verification result
    """
    # Load receipts
    all_receipts = load_receipts()

    # Filter by time range
    receipts = []
    for r in all_receipts:
        ts = r.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        receipts.append(r)

    # Get decision receipts
    decisions = [r for r in receipts if r.get("receipt_type") == "decision_log"]

    if not decisions:
        return {
            "verified": True,
            "message": "No decisions in range",
            "decisions_checked": 0,
            "violations": []
        }

    # Verify chain integrity
    try:
        result = verify_chain_integrity(decisions)
        return {
            "verified": result.get("is_valid", False),
            "message": result.get("message", ""),
            "decisions_checked": len(decisions),
            "violations": result.get("violations", [])
        }
    except Exception as e:
        return {
            "verified": False,
            "message": str(e),
            "decisions_checked": len(decisions),
            "violations": [{"error": str(e)}]
        }


def query_decisions_tool(action_type: Optional[str] = None,
                         confidence_min: Optional[float] = None,
                         confidence_max: Optional[float] = None,
                         limit: int = 100) -> dict:
    """Query decision history.

    Args:
        action_type: Filter by action type
        confidence_min: Minimum confidence
        confidence_max: Maximum confidence
        limit: Maximum results

    Returns:
        Query results
    """
    all_receipts = load_receipts()

    # Filter decisions
    decisions = []
    for r in all_receipts:
        if r.get("receipt_type") != "decision":
            continue

        # Apply filters
        if action_type and r.get("action_type") != action_type:
            continue

        confidence = r.get("confidence", 0.5)
        if confidence_min is not None and confidence < confidence_min:
            continue
        if confidence_max is not None and confidence > confidence_max:
            continue

        decisions.append({
            "decision_id": r.get("decision_id"),
            "action_type": r.get("action_type"),
            "confidence": confidence,
            "timestamp": r.get("ts")
        })

        if len(decisions) >= limit:
            break

    return {
        "count": len(decisions),
        "decisions": decisions
    }


def get_audit_trail_tool(report_type: str = "summary",
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None) -> dict:
    """Generate audit trail report.

    Args:
        report_type: Type of report
        start_time: Start time filter
        end_time: End time filter

    Returns:
        Audit report
    """
    all_receipts = load_receipts()

    # Filter by time
    receipts = []
    for r in all_receipts:
        ts = r.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue
        receipts.append(r)

    # Count by type
    type_counts = {}
    for r in receipts:
        rt = r.get("receipt_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    # Basic stats
    decision_count = type_counts.get("decision", 0) + type_counts.get("decision_log", 0)
    anomaly_count = type_counts.get("anomaly", 0) + type_counts.get("anomaly_alert", 0)
    intervention_count = type_counts.get("intervention", 0)

    report = {
        "report_type": report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "time_range": {
            "start": start_time or "beginning",
            "end": end_time or "now"
        },
        "summary": {
            "total_receipts": len(receipts),
            "decisions": decision_count,
            "anomalies": anomaly_count,
            "interventions": intervention_count
        },
        "receipt_types": type_counts
    }

    if report_type == "detailed":
        report["receipts"] = receipts[:1000]  # Limit for large reports

    return report


def inject_intervention_tool(decision_id: str, reason_code: str,
                             correction: dict) -> dict:
    """Record human intervention.

    Args:
        decision_id: Decision being overridden
        reason_code: Reason for intervention
        correction: Corrected decision

    Returns:
        Intervention result
    """
    intervention_id = str(uuid.uuid4())

    # Emit intervention receipt
    receipt = emit_receipt("intervention", {
        "intervention_id": intervention_id,
        "decision_id": decision_id,
        "reason_code": reason_code,
        "correction": correction,
        "source": "mcp",
        "approver": "external_orchestrator"
    }, silent=True)

    return {
        "intervention_id": intervention_id,
        "decision_id": decision_id,
        "status": "recorded",
        "receipt_hash": receipt.get("payload_hash", "")
    }


def spawn_investigator_tool(anomaly_id: str,
                            investigation_type: str = "quick") -> dict:
    """Spawn investigator agents.

    Args:
        anomaly_id: Anomaly to investigate
        investigation_type: Type of investigation

    Returns:
        Spawned agent info
    """
    # This would integrate with the spawner module
    agent_ids = []

    if investigation_type == "deep":
        # Spawn multiple investigators
        agent_ids = [str(uuid.uuid4()) for _ in range(3)]
    elif investigation_type == "targeted":
        agent_ids = [str(uuid.uuid4())]
    else:  # quick
        agent_ids = [str(uuid.uuid4())]

    # Emit spawn receipt
    emit_receipt("spawn", {
        "parent_agent_id": "MCP_ORCHESTRATOR",
        "child_agents": [
            {"agent_id": aid, "agent_type": "investigator", "investigation_type": investigation_type}
            for aid in agent_ids
        ],
        "trigger": f"mcp_request:{anomaly_id}",
        "confidence_at_spawn": 0.5,
        "depth_level": 1
    }, silent=True)

    return {
        "anomaly_id": anomaly_id,
        "investigation_type": investigation_type,
        "agent_ids": agent_ids,
        "status": "spawned"
    }


# =============================================================================
# HANDLER REGISTRATION
# =============================================================================

def register_default_tools(server):
    """Register default tools with server.

    Args:
        server: MCPServer instance
    """
    server.register_tool(
        "verify_chain",
        verify_chain_tool,
        "Verify decision chain integrity",
        FLIGHT_RECORDER_TOOLS[0]["input_schema"]
    )

    server.register_tool(
        "query_decisions",
        query_decisions_tool,
        "Query decision history",
        FLIGHT_RECORDER_TOOLS[1]["input_schema"]
    )

    server.register_tool(
        "get_audit_trail",
        get_audit_trail_tool,
        "Generate audit report",
        FLIGHT_RECORDER_TOOLS[2]["input_schema"]
    )

    server.register_tool(
        "inject_intervention",
        inject_intervention_tool,
        "Record human intervention",
        FLIGHT_RECORDER_TOOLS[3]["input_schema"]
    )

    server.register_tool(
        "spawn_investigator",
        spawn_investigator_tool,
        "Spawn investigator agents",
        FLIGHT_RECORDER_TOOLS[4]["input_schema"]
    )


def handle_tool_call(tool_name: str, inputs: dict) -> dict:
    """Handle a tool call directly.

    Args:
        tool_name: Name of tool to call
        inputs: Tool inputs

    Returns:
        Tool result
    """
    handlers = {
        "verify_chain": verify_chain_tool,
        "query_decisions": query_decisions_tool,
        "get_audit_trail": get_audit_trail_tool,
        "inject_intervention": inject_intervention_tool,
        "spawn_investigator": spawn_investigator_tool
    }

    if tool_name not in handlers:
        return {"error": f"Unknown tool: {tool_name}"}

    return handlers[tool_name](**inputs)
