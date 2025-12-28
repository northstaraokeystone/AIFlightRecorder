"""Tests for MCP server interface (v2.2)."""

import pytest
import time

from src.mcp import start_server, stop_server, get_server
from src.mcp.server import MCPRequest, MCPResponse, MCPServer, handle_mcp_request
from src.mcp.tools import FLIGHT_RECORDER_TOOLS, handle_tool_call
from src.mcp.resources import FLIGHT_RECORDER_RESOURCES, handle_resource_request


@pytest.fixture
def mcp_server():
    """Create and return MCP server, stop after test."""
    server = start_server()
    yield server
    stop_server()


class TestMCPServer:
    """Tests for MCP server."""

    def test_server_start(self, mcp_server):
        """Server should start successfully."""
        assert mcp_server is not None
        stats = mcp_server.get_stats()
        assert stats["running"] is True

    def test_server_singleton(self, mcp_server):
        """get_server should return same instance."""
        server2 = get_server()
        assert server2 is mcp_server

    def test_server_stop(self):
        """Server should stop cleanly."""
        server = start_server()
        stop_server()

        # Getting server after stop should return new one or None
        # Implementation-dependent

    def test_server_tools_registered(self, mcp_server):
        """Server should have tools registered."""
        tools = mcp_server.get_tools()
        assert len(tools) > 0
        tool_names = [t["name"] for t in tools]
        assert "verify_chain" in tool_names
        assert "query_decisions" in tool_names

    def test_server_resources_registered(self, mcp_server):
        """Server should have resources registered."""
        resources = mcp_server.get_resources()
        assert len(resources) > 0
        resource_uris = [r["uri"] for r in resources]
        assert "flight://decisions/stream" in resource_uris


class TestMCPRequest:
    """Tests for MCP request handling."""

    def test_request_dataclass(self):
        """MCPRequest should store all fields."""
        request = MCPRequest(
            request_type="tool",
            name="verify_chain",
            inputs={"receipts": []},
            caller_id="test"
        )
        assert request.request_type == "tool"
        assert request.name == "verify_chain"

    def test_handle_tool_request(self, mcp_server):
        """Tool requests should be handled."""
        result = handle_mcp_request(
            request_type="tool",
            name="verify_chain",
            inputs={},
            caller_id="test"
        )

        assert result["success"] is True
        assert "outputs" in result

    def test_handle_resource_request(self, mcp_server):
        """Resource requests should be handled."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://decisions/stream",
            inputs={"limit": 5},
            caller_id="test"
        )

        assert result["success"] is True
        assert "outputs" in result

    def test_handle_unknown_tool(self, mcp_server):
        """Unknown tool should fail gracefully."""
        result = handle_mcp_request(
            request_type="tool",
            name="unknown_tool",
            inputs={},
            caller_id="test"
        )

        assert result["success"] is False
        assert "error" in result

    def test_handle_unknown_resource(self, mcp_server):
        """Unknown resource should fail gracefully."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://unknown",
            inputs={},
            caller_id="test"
        )

        assert result["success"] is False


class TestMCPTools:
    """Tests for individual MCP tools."""

    def test_verify_chain_tool(self, mcp_server):
        """verify_chain tool should work."""
        result = handle_mcp_request(
            request_type="tool",
            name="verify_chain",
            inputs={},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "is_valid" in outputs

    def test_query_decisions_tool(self, mcp_server):
        """query_decisions tool should work."""
        result = handle_mcp_request(
            request_type="tool",
            name="query_decisions",
            inputs={"limit": 10},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "decisions" in outputs or "count" in outputs

    def test_get_audit_trail_tool(self, mcp_server):
        """get_audit_trail tool should work."""
        result = handle_mcp_request(
            request_type="tool",
            name="get_audit_trail",
            inputs={"report_type": "summary"},
            caller_id="test"
        )
        assert result["success"] is True

    def test_inject_intervention_tool(self, mcp_server):
        """inject_intervention tool should work."""
        result = handle_mcp_request(
            request_type="tool",
            name="inject_intervention",
            inputs={
                "decision_id": "d1",
                "reason_code": "TESTING",
                "correction": {"action": {"type": "CONTINUE"}}
            },
            caller_id="test"
        )
        assert result["success"] is True
        assert "intervention_id" in result["outputs"]

    def test_spawn_investigator_tool(self, mcp_server):
        """spawn_investigator tool should work."""
        result = handle_mcp_request(
            request_type="tool",
            name="spawn_investigator",
            inputs={
                "anomaly_id": "a1",
                "investigation_type": "targeted"
            },
            caller_id="test"
        )
        assert result["success"] is True


class TestMCPResources:
    """Tests for MCP resources."""

    def test_decisions_stream_resource(self, mcp_server):
        """decisions/stream resource should work."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://decisions/stream",
            inputs={"limit": 5},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "type" in outputs
        assert outputs["type"] == "decision_stream"

    def test_entropy_metrics_resource(self, mcp_server):
        """metrics/entropy resource should work."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://metrics/entropy",
            inputs={},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "available" in outputs

    def test_active_agents_resource(self, mcp_server):
        """agents/active resource should work."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://agents/active",
            inputs={},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "count" in outputs

    def test_graduated_patterns_resource(self, mcp_server):
        """patterns/graduated resource should work."""
        result = handle_mcp_request(
            request_type="resource",
            name="flight://patterns/graduated",
            inputs={},
            caller_id="test"
        )
        outputs = result["outputs"]
        assert "count" in outputs


class TestMCPStats:
    """Tests for MCP statistics."""

    def test_request_count(self, mcp_server):
        """Request count should increment."""
        initial_stats = mcp_server.get_stats()
        initial_count = initial_stats["request_count"]

        handle_mcp_request(
            request_type="tool",
            name="verify_chain",
            inputs={},
            caller_id="test"
        )

        new_stats = mcp_server.get_stats()
        assert new_stats["request_count"] > initial_count

    def test_tools_count(self, mcp_server):
        """Tools count should be accurate."""
        stats = mcp_server.get_stats()
        tools = mcp_server.get_tools()
        assert stats["tools_count"] == len(tools)

    def test_resources_count(self, mcp_server):
        """Resources count should be accurate."""
        stats = mcp_server.get_stats()
        resources = mcp_server.get_resources()
        assert stats["resources_count"] == len(resources)


class TestToolsModule:
    """Tests for tools module directly."""

    def test_flight_recorder_tools(self):
        """FLIGHT_RECORDER_TOOLS should have tool definitions."""
        tools = FLIGHT_RECORDER_TOOLS
        assert isinstance(tools, list)
        assert len(tools) == 5  # 5 tools defined

        for tool in tools:
            assert "name" in tool
            assert "description" in tool

    def test_handle_tool_call(self):
        """handle_tool_call should work."""
        result = handle_tool_call("verify_chain", {})
        assert isinstance(result, dict)


class TestResourcesModule:
    """Tests for resources module directly."""

    def test_flight_recorder_resources(self):
        """FLIGHT_RECORDER_RESOURCES should have resource definitions."""
        resources = FLIGHT_RECORDER_RESOURCES
        assert isinstance(resources, list)
        assert len(resources) == 8  # 8 resources defined

        for resource in resources:
            assert "uri" in resource
            assert "description" in resource

    def test_handle_resource_request(self):
        """handle_resource_request should work."""
        result = handle_resource_request("flight://decisions/stream")
        assert isinstance(result, dict)


class TestPerformance:
    """Performance tests for MCP."""

    def test_tool_latency(self, mcp_server):
        """Tool invocation should complete under 50ms."""
        start = time.perf_counter()
        handle_mcp_request(
            request_type="tool",
            name="verify_chain",
            inputs={},
            caller_id="test"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Tool latency {elapsed_ms}ms exceeds 50ms SLO"

    def test_resource_latency(self, mcp_server):
        """Resource access should complete under 50ms."""
        start = time.perf_counter()
        handle_mcp_request(
            request_type="resource",
            name="flight://decisions/stream",
            inputs={"limit": 10},
            caller_id="test"
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Resource latency {elapsed_ms}ms exceeds 50ms SLO"
