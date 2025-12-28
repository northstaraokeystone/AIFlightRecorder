"""MCP Resources - Flight Recorder Data (v2.2)

Resources exposed via MCP for external AI orchestrators.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..core import load_receipts


# =============================================================================
# RESOURCE DEFINITIONS
# =============================================================================

FLIGHT_RECORDER_RESOURCES = [
    {
        "uri": "flight://decisions/stream",
        "description": "Live decision feed (recent decisions)"
    },
    {
        "uri": "flight://decisions/*",
        "description": "Individual decision by ID"
    },
    {
        "uri": "flight://agents/active",
        "description": "Currently active spawned agents"
    },
    {
        "uri": "flight://agents/*",
        "description": "Individual agent by ID"
    },
    {
        "uri": "flight://metrics/entropy",
        "description": "System entropy and health metrics"
    },
    {
        "uri": "flight://metrics/slo",
        "description": "SLO compliance metrics"
    },
    {
        "uri": "flight://patterns/graduated",
        "description": "Graduated permanent patterns"
    },
    {
        "uri": "flight://receipts/recent",
        "description": "Recent receipts (last 100)"
    }
]


# =============================================================================
# RESOURCE HANDLERS
# =============================================================================

def get_decision_stream(uri: str, limit: int = 50, **kwargs) -> dict:
    """Get recent decisions.

    Args:
        uri: Resource URI
        limit: Maximum decisions to return

    Returns:
        Decision stream data
    """
    all_receipts = load_receipts()

    # Filter for decision receipts
    decisions = [
        r for r in all_receipts
        if r.get("receipt_type") in ("decision", "decision_log")
    ]

    # Get most recent
    recent = decisions[-limit:] if len(decisions) > limit else decisions

    return {
        "uri": uri,
        "type": "decision_stream",
        "count": len(recent),
        "decisions": [
            {
                "decision_id": d.get("decision_id"),
                "action_type": d.get("action_type"),
                "confidence": d.get("confidence"),
                "timestamp": d.get("ts")
            }
            for d in recent
        ],
        "as_of": datetime.now(timezone.utc).isoformat()
    }


def get_decision_by_id(uri: str, **kwargs) -> dict:
    """Get a specific decision by ID.

    Args:
        uri: Resource URI containing decision ID

    Returns:
        Decision data or not found
    """
    # Extract ID from URI
    parts = uri.split("/")
    if len(parts) < 3:
        return {"error": "Invalid decision URI", "uri": uri}

    decision_id = parts[-1]

    all_receipts = load_receipts()

    for r in all_receipts:
        if r.get("decision_id") == decision_id:
            return {
                "uri": uri,
                "type": "decision",
                "found": True,
                "decision": r
            }

    return {
        "uri": uri,
        "type": "decision",
        "found": False,
        "decision_id": decision_id
    }


def get_active_agents(uri: str, **kwargs) -> dict:
    """Get currently active agents.

    Args:
        uri: Resource URI

    Returns:
        Active agents data
    """
    # This would integrate with the spawner registry
    # For now, return structure based on spawn receipts
    all_receipts = load_receipts()

    spawns = [r for r in all_receipts if r.get("receipt_type") == "spawn"]
    prunings = [r for r in all_receipts if r.get("receipt_type") == "pruning"]

    # Track which agents are still active
    spawned_agents = {}
    for s in spawns:
        for child in s.get("child_agents", []):
            if isinstance(child, dict):
                agent_id = child.get("agent_id", "")
            else:
                agent_id = str(child)
            if agent_id:
                spawned_agents[agent_id] = {
                    "agent_id": agent_id,
                    "spawned_at": s.get("ts"),
                    "parent": s.get("parent_agent_id"),
                    "trigger": s.get("trigger")
                }

    # Remove pruned agents
    for p in prunings:
        for agent_id in p.get("agents_terminated", []):
            spawned_agents.pop(agent_id, None)

    return {
        "uri": uri,
        "type": "agent_registry",
        "count": len(spawned_agents),
        "agents": list(spawned_agents.values()),
        "as_of": datetime.now(timezone.utc).isoformat()
    }


def get_agent_by_id(uri: str, **kwargs) -> dict:
    """Get a specific agent by ID.

    Args:
        uri: Resource URI containing agent ID

    Returns:
        Agent data or not found
    """
    parts = uri.split("/")
    if len(parts) < 3:
        return {"error": "Invalid agent URI", "uri": uri}

    agent_id = parts[-1]

    all_receipts = load_receipts()

    # Find spawn receipt for this agent
    for r in all_receipts:
        if r.get("receipt_type") == "spawn":
            for child in r.get("child_agents", []):
                child_id = child.get("agent_id") if isinstance(child, dict) else str(child)
                if child_id == agent_id:
                    return {
                        "uri": uri,
                        "type": "agent",
                        "found": True,
                        "agent": {
                            "agent_id": agent_id,
                            "spawned_at": r.get("ts"),
                            "parent": r.get("parent_agent_id"),
                            "trigger": r.get("trigger"),
                            "depth_level": r.get("depth_level")
                        }
                    }

    return {
        "uri": uri,
        "type": "agent",
        "found": False,
        "agent_id": agent_id
    }


def get_entropy_metrics(uri: str, **kwargs) -> dict:
    """Get system entropy metrics.

    Args:
        uri: Resource URI

    Returns:
        Entropy metrics
    """
    all_receipts = load_receipts()

    # Find entropy receipts
    entropy_receipts = [r for r in all_receipts if r.get("receipt_type") == "entropy"]

    if not entropy_receipts:
        return {
            "uri": uri,
            "type": "entropy_metrics",
            "available": False,
            "message": "No entropy data available"
        }

    # Get most recent
    latest = entropy_receipts[-1]

    # Calculate trend from recent history
    recent = entropy_receipts[-10:]
    if len(recent) >= 2:
        first_entropy = recent[0].get("system_entropy_bits", 0)
        last_entropy = recent[-1].get("system_entropy_bits", 0)
        trend = "decreasing" if last_entropy < first_entropy else "increasing"
    else:
        trend = "stable"

    return {
        "uri": uri,
        "type": "entropy_metrics",
        "available": True,
        "current": {
            "system_entropy_bits": latest.get("system_entropy_bits"),
            "entropy_delta": latest.get("entropy_delta"),
            "conservation_valid": latest.get("conservation_valid"),
            "superposition_count": latest.get("superposition_count", 0)
        },
        "trend": trend,
        "history_count": len(entropy_receipts),
        "as_of": latest.get("ts")
    }


def get_slo_metrics(uri: str, **kwargs) -> dict:
    """Get SLO compliance metrics.

    Args:
        uri: Resource URI

    Returns:
        SLO metrics
    """
    all_receipts = load_receipts()

    # Count anomalies by type
    anomalies = [r for r in all_receipts if r.get("receipt_type") == "anomaly"]

    slo_violations = {}
    for a in anomalies:
        metric = a.get("metric", "unknown")
        classification = a.get("classification", "unknown")
        key = f"{metric}:{classification}"
        slo_violations[key] = slo_violations.get(key, 0) + 1

    return {
        "uri": uri,
        "type": "slo_metrics",
        "total_violations": len(anomalies),
        "violations_by_type": slo_violations,
        "compliance_rate": 1.0 - (len(anomalies) / max(1, len(all_receipts))),
        "as_of": datetime.now(timezone.utc).isoformat()
    }


def get_graduated_patterns(uri: str, **kwargs) -> dict:
    """Get graduated permanent patterns.

    Args:
        uri: Resource URI

    Returns:
        Graduated patterns
    """
    all_receipts = load_receipts()

    # Find graduation receipts
    graduations = [r for r in all_receipts if r.get("receipt_type") == "graduation"]

    patterns = []
    for g in graduations:
        patterns.append({
            "pattern_id": g.get("solution_pattern_id"),
            "agent_id": g.get("agent_id"),
            "effectiveness": g.get("effectiveness"),
            "autonomy_score": g.get("autonomy_score"),
            "promoted_to": g.get("promoted_to"),
            "graduated_at": g.get("ts")
        })

    return {
        "uri": uri,
        "type": "graduated_patterns",
        "count": len(patterns),
        "patterns": patterns,
        "as_of": datetime.now(timezone.utc).isoformat()
    }


def get_recent_receipts(uri: str, limit: int = 100, **kwargs) -> dict:
    """Get recent receipts.

    Args:
        uri: Resource URI
        limit: Maximum receipts

    Returns:
        Recent receipts
    """
    all_receipts = load_receipts()
    recent = all_receipts[-limit:] if len(all_receipts) > limit else all_receipts

    return {
        "uri": uri,
        "type": "receipts",
        "count": len(recent),
        "receipts": recent,
        "total_available": len(all_receipts),
        "as_of": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# HANDLER REGISTRATION
# =============================================================================

def register_default_resources(server):
    """Register default resources with server.

    Args:
        server: MCPServer instance
    """
    server.register_resource(
        "flight://decisions/stream",
        get_decision_stream,
        "Live decision feed"
    )

    server.register_resource(
        "flight://decisions/*",
        get_decision_by_id,
        "Individual decision by ID"
    )

    server.register_resource(
        "flight://agents/active",
        get_active_agents,
        "Active spawned agents"
    )

    server.register_resource(
        "flight://agents/*",
        get_agent_by_id,
        "Individual agent by ID"
    )

    server.register_resource(
        "flight://metrics/entropy",
        get_entropy_metrics,
        "System entropy metrics"
    )

    server.register_resource(
        "flight://metrics/slo",
        get_slo_metrics,
        "SLO compliance metrics"
    )

    server.register_resource(
        "flight://patterns/graduated",
        get_graduated_patterns,
        "Graduated patterns"
    )

    server.register_resource(
        "flight://receipts/recent",
        get_recent_receipts,
        "Recent receipts"
    )


def handle_resource_request(uri: str, **kwargs) -> dict:
    """Handle a resource request directly.

    Args:
        uri: Resource URI
        **kwargs: Additional parameters

    Returns:
        Resource data
    """
    # Route to appropriate handler
    if uri == "flight://decisions/stream":
        return get_decision_stream(uri, **kwargs)
    elif uri.startswith("flight://decisions/"):
        return get_decision_by_id(uri, **kwargs)
    elif uri == "flight://agents/active":
        return get_active_agents(uri, **kwargs)
    elif uri.startswith("flight://agents/"):
        return get_agent_by_id(uri, **kwargs)
    elif uri == "flight://metrics/entropy":
        return get_entropy_metrics(uri, **kwargs)
    elif uri == "flight://metrics/slo":
        return get_slo_metrics(uri, **kwargs)
    elif uri == "flight://patterns/graduated":
        return get_graduated_patterns(uri, **kwargs)
    elif uri == "flight://receipts/recent":
        return get_recent_receipts(uri, **kwargs)
    else:
        return {"error": f"Unknown resource: {uri}"}
