"""
Tests for L3 Connectivity Layer.

Tests MCP Host/Client and A2A protocol implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from squadron.connectivity.mcp_host import MCPHost, MCPServer, MCPTool
from squadron.connectivity.a2a import (
    A2AAgent,
    AgentCard,
    AgentCapability,
    A2ATask,
    TaskState,
)


class TestMCPTool:
    """Tests for MCPTool dataclass."""
    
    def test_mcp_tool_dataclass(self):
        tool = MCPTool(
            name="test",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test-server",
        )
        assert tool.name == "test"
        assert tool.server_name == "test-server"
    
    def test_mcp_tool_to_openai_format(self):
        tool = MCPTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="test",
        )
        formatted = tool.to_openai_format()
        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "search"
    
    def test_mcp_tool_to_anthropic_format(self):
        tool = MCPTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object"},
            server_name="test",
        )
        formatted = tool.to_anthropic_format()
        assert formatted["name"] == "search"
        assert "input_schema" in formatted


class TestMCPHost:
    """Tests for MCPHost."""
    
    def test_init(self):
        host = MCPHost()
        assert host._servers == {}
        assert host._tool_index == {}
    
    def test_get_all_tools_empty(self):
        host = MCPHost()
        tools = host.get_all_tools()
        assert tools == []


class TestAgentCard:
    """Tests for AgentCard."""
    
    def test_create(self):
        card = AgentCard(
            id="agent-1",
            name="Test Agent",
            description="A test agent",
            base_url="https://agent.example.com",
        )
        assert card.id == "agent-1"
        assert card.name == "Test Agent"
    
    def test_with_capabilities(self):
        card = AgentCard(
            id="agent-1",
            name="Test Agent",
            description="A test agent",
            capabilities=[
                AgentCapability(name="search", description="Search capability"),
            ],
        )
        assert len(card.capabilities) == 1
        assert card.capabilities[0].name == "search"
    
    def test_to_dict(self):
        card = AgentCard(
            id="agent-1",
            name="Test Agent",
            description="A test agent",
        )
        data = card.to_dict()
        assert data["id"] == "agent-1"
        assert data["name"] == "Test Agent"
    
    def test_from_dict(self):
        data = {
            "id": "agent-2",
            "name": "Another Agent",
            "description": "Description",
        }
        card = AgentCard.from_dict(data)
        assert card.id == "agent-2"
        assert card.name == "Another Agent"


class TestA2ATask:
    """Tests for A2ATask."""
    
    def test_create(self):
        task = A2ATask(
            capability="search",
            input_data={"query": "test"},
        )
        assert task.capability == "search"
        assert task.state == TaskState.REQUESTED
        assert task.input_data == {"query": "test"}
    
    def test_state_transitions(self):
        task = A2ATask(capability="test")
        
        assert task.state == TaskState.REQUESTED
        
        task.state = TaskState.RUNNING
        assert task.state == TaskState.RUNNING
        
        task.state = TaskState.COMPLETED
        assert task.state == TaskState.COMPLETED
    
    def test_to_dict(self):
        task = A2ATask(
            capability="analyze",
            input_data={"text": "hello"},
        )
        data = task.to_dict()
        assert data["capability"] == "analyze"
        assert data["state"] == "requested"
        assert data["input"] == {"text": "hello"}


class TestA2AAgent:
    """Tests for A2AAgent."""
    
    def test_init(self):
        card = AgentCard(
            id="test-agent",
            name="Test",
            description="Test agent",
        )
        agent = A2AAgent(card=card)
        assert agent.card.id == "test-agent"
    
    def test_capability_decorator(self):
        card = AgentCard(id="test", name="Test", description="Test")
        agent = A2AAgent(card=card)
        
        @agent.capability("search")
        async def search_handler(task):
            return {"results": []}
        
        assert "search" in agent._handlers
