"""
L3: Connectivity Layer

Provides standardized protocols for tool and agent integration:
- MCP (Model Context Protocol): Universal tool connectivity
- A2A (Agent-to-Agent Protocol): Multi-agent coordination
"""

from squadron.connectivity.mcp_host import MCPHost, MCPServer, MCPTool
from squadron.connectivity.mcp_client import MCPClient
from squadron.connectivity.a2a import A2AAgent, AgentCard, TaskState

__all__ = [
    # MCP
    "MCPHost",
    "MCPServer",
    "MCPTool",
    "MCPClient",
    # A2A
    "A2AAgent",
    "AgentCard",
    "TaskState",
]
