"""MCP Server Implementation (v2.2)

Model Context Protocol server for external AI orchestrator integration.
Request/response only (no real-time streaming in v2.2).
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core import emit_receipt


@dataclass
class MCPRequest:
    """Incoming MCP request."""
    request_id: str
    request_type: str  # "tool" or "resource"
    name: str
    inputs: dict
    caller_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MCPResponse:
    """Outgoing MCP response."""
    request_id: str
    success: bool
    outputs: dict
    error: Optional[str] = None
    latency_ms: float = 0.0


class MCPServer:
    """MCP Server for flight recorder integration.

    Provides:
    - Tool invocation for external orchestrators
    - Resource access for flight recorder data
    - Request/response pattern (no streaming)
    """

    def __init__(self):
        self._running = False
        self._tools: Dict[str, Callable] = {}
        self._resources: Dict[str, Callable] = {}
        self._request_count = 0
        self._start_time: Optional[str] = None

    def start(self):
        """Start the MCP server."""
        self._running = True
        self._start_time = datetime.now(timezone.utc).isoformat()

        # Register default tools
        from .tools import register_default_tools
        register_default_tools(self)

        # Register default resources
        from .resources import register_default_resources
        register_default_resources(self)

        emit_receipt("system_event", {
            "event_type": "mcp_server_start",
            "tools_registered": list(self._tools.keys()),
            "resources_registered": list(self._resources.keys())
        }, silent=True)

    def stop(self):
        """Stop the MCP server."""
        self._running = False

        emit_receipt("system_event", {
            "event_type": "mcp_server_stop",
            "requests_handled": self._request_count,
            "uptime_since": self._start_time
        }, silent=True)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    def register_tool(self, name: str, handler: Callable,
                      description: str = "",
                      input_schema: Optional[dict] = None):
        """Register a tool handler.

        Args:
            name: Tool name
            handler: Callable that handles tool calls
            description: Tool description
            input_schema: JSON schema for inputs
        """
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema or {}
        }

    def register_resource(self, uri_pattern: str, handler: Callable,
                          description: str = ""):
        """Register a resource handler.

        Args:
            uri_pattern: URI pattern (e.g., "flight://decisions/stream")
            handler: Callable that returns resource data
            description: Resource description
        """
        self._resources[uri_pattern] = {
            "handler": handler,
            "description": description
        }

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request.

        Args:
            request: The MCP request

        Returns:
            MCPResponse with result
        """
        start_time = time.perf_counter()
        self._request_count += 1

        try:
            if request.request_type == "tool":
                response = self._handle_tool(request)
            elif request.request_type == "resource":
                response = self._handle_resource(request)
            else:
                response = MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    outputs={},
                    error=f"Unknown request type: {request.request_type}"
                )

        except Exception as e:
            response = MCPResponse(
                request_id=request.request_id,
                success=False,
                outputs={},
                error=str(e)
            )

        response.latency_ms = (time.perf_counter() - start_time) * 1000

        # Emit receipt
        self._emit_mcp_receipt(request, response)

        return response

    def _handle_tool(self, request: MCPRequest) -> MCPResponse:
        """Handle a tool invocation request.

        Args:
            request: Tool request

        Returns:
            MCPResponse
        """
        if request.name not in self._tools:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                outputs={},
                error=f"Unknown tool: {request.name}"
            )

        tool = self._tools[request.name]
        handler = tool["handler"]

        # Call the handler
        result = handler(**request.inputs)

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            outputs=result if isinstance(result, dict) else {"result": result}
        )

    def _handle_resource(self, request: MCPRequest) -> MCPResponse:
        """Handle a resource request.

        Args:
            request: Resource request

        Returns:
            MCPResponse
        """
        # Find matching resource
        handler = None
        for pattern, resource in self._resources.items():
            if self._matches_pattern(request.name, pattern):
                handler = resource["handler"]
                break

        if handler is None:
            return MCPResponse(
                request_id=request.request_id,
                success=False,
                outputs={},
                error=f"Unknown resource: {request.name}"
            )

        # Call the handler
        result = handler(request.name, **request.inputs)

        return MCPResponse(
            request_id=request.request_id,
            success=True,
            outputs=result if isinstance(result, dict) else {"data": result}
        )

    def _matches_pattern(self, uri: str, pattern: str) -> bool:
        """Check if URI matches pattern.

        Args:
            uri: Requested URI
            pattern: Pattern to match

        Returns:
            True if matches
        """
        # Simple prefix matching for now
        if pattern.endswith("*"):
            return uri.startswith(pattern[:-1])
        return uri == pattern

    def _emit_mcp_receipt(self, request: MCPRequest,
                          response: MCPResponse):
        """Emit MCP request receipt.

        Args:
            request: The request
            response: The response
        """
        emit_receipt("mcp", {
            "request_id": request.request_id,
            "tool_or_resource": request.name,
            "caller_id": request.caller_id,
            "inputs": request.inputs,
            "outputs": response.outputs if response.success else {"error": response.error},
            "success": response.success,
            "latency_ms": response.latency_ms
        }, silent=True)

    def get_tools(self) -> List[dict]:
        """Get list of available tools.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": name,
                "description": info["description"],
                "input_schema": info["input_schema"]
            }
            for name, info in self._tools.items()
        ]

    def get_resources(self) -> List[dict]:
        """Get list of available resources.

        Returns:
            List of resource definitions
        """
        return [
            {
                "uri": pattern,
                "description": info["description"]
            }
            for pattern, info in self._resources.items()
        ]

    def get_stats(self) -> dict:
        """Get server statistics.

        Returns:
            Stats dict
        """
        return {
            "running": self._running,
            "start_time": self._start_time,
            "request_count": self._request_count,
            "tools_count": len(self._tools),
            "resources_count": len(self._resources)
        }


# =============================================================================
# MODULE-LEVEL INTERFACE
# =============================================================================

_mcp_server: Optional[MCPServer] = None


def get_server() -> MCPServer:
    """Get the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server


def start_server() -> MCPServer:
    """Start the global MCP server.

    Returns:
        The running server
    """
    server = get_server()
    if not server.is_running():
        server.start()
    return server


def stop_server():
    """Stop the global MCP server."""
    global _mcp_server
    if _mcp_server is not None and _mcp_server.is_running():
        _mcp_server.stop()


def handle_mcp_request(request_type: str, name: str,
                       inputs: dict, caller_id: str = "unknown") -> dict:
    """Handle an MCP request via the global server.

    Args:
        request_type: "tool" or "resource"
        name: Tool or resource name
        inputs: Request inputs
        caller_id: Caller identifier

    Returns:
        Response dict
    """
    server = get_server()
    if not server.is_running():
        server.start()

    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        request_type=request_type,
        name=name,
        inputs=inputs,
        caller_id=caller_id
    )

    response = server.handle_request(request)

    return {
        "success": response.success,
        "outputs": response.outputs,
        "error": response.error,
        "latency_ms": response.latency_ms
    }
