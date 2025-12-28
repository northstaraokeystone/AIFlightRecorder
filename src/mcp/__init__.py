"""MCP Server Interface Module (v2.2)

Model Context Protocol server for external AI orchestrators.
Exposes flight recorder capabilities as MCP tools and resources.
"""

from .server import MCPServer, start_server, stop_server, get_server
from .tools import FLIGHT_RECORDER_TOOLS, handle_tool_call
from .resources import FLIGHT_RECORDER_RESOURCES, handle_resource_request

__all__ = [
    "MCPServer",
    "start_server",
    "stop_server",
    "get_server",
    "FLIGHT_RECORDER_TOOLS",
    "handle_tool_call",
    "FLIGHT_RECORDER_RESOURCES",
    "handle_resource_request"
]
