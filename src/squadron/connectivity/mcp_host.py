"""
MCP Host Implementation

The MCP (Model Context Protocol) Host manages connections to MCP servers,
providing a unified interface for tool discovery and execution.

This is the "USB-C" of AI - any MCP-compliant tool works instantly.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

import structlog

from squadron.core.config import MCPConfig

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MCPTransport(str, Enum):
    """Transport protocol for MCP communication."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPTool:
    """
    Represents a tool exposed by an MCP server.
    
    Tools are executable functions that agents can call.
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    
    # Optional metadata
    tags: list[str] = field(default_factory=list)
    requires_approval: bool = False
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }
    
    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class MCPResource:
    """
    Represents a resource exposed by an MCP server.
    
    Resources are file-like data that can be read by agents.
    """
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    server_name: str = ""


@dataclass
class MCPPrompt:
    """
    Represents a prompt template exposed by an MCP server.
    """
    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server_name: str = ""


class MCPServer:
    """
    Represents a connection to an MCP server.
    
    Handles spawning the server process and communicating via stdio or SSE.
    """
    
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: MCPTransport = MCPTransport.STDIO,
        url: str | None = None,
    ):
        """
        Initialize an MCP server connection.
        
        Args:
            name: Unique identifier for this server
            command: Command to spawn the server (for stdio)
            args: Command arguments
            env: Environment variables
            transport: Communication protocol
            url: Server URL (for SSE/HTTP transport)
        """
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.transport = transport
        self.url = url
        
        self._process: subprocess.Popen | None = None
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._connected = False
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None
    
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            return
        
        if self.transport == MCPTransport.STDIO:
            await self._connect_stdio()
        elif self.transport == MCPTransport.SSE:
            await self._connect_sse()
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")
        
        # Initialize the connection
        await self._initialize()
        
        # Discover capabilities
        await self._discover_tools()
        await self._discover_resources()
        await self._discover_prompts()
        
        self._connected = True
        logger.info(
            "MCP server connected",
            name=self.name,
            tools=len(self._tools),
            resources=len(self._resources),
        )
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        try:
            # Build the full command
            cmd = [self.command] + self.args
            
            # Merge environment
            env = {**dict(subprocess.os.environ), **self.env}
            
            # Spawn the process
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )
            
            # Start reading responses
            self._read_task = asyncio.create_task(self._read_loop())
            
            logger.debug("Spawned MCP server process", command=cmd)
            
        except Exception as e:
            logger.error("Failed to spawn MCP server", error=str(e))
            raise
    
    async def _connect_sse(self) -> None:
        """Connect via SSE transport."""
        # SSE implementation would use aiohttp or httpx
        # For now, we'll stub this out
        logger.warning("SSE transport not yet implemented")
        raise NotImplementedError("SSE transport coming soon")
    
    async def _read_loop(self) -> None:
        """Read responses from the server."""
        if not self._process or not self._process.stdout:
            return
        
        buffer = b""
        while True:
            try:
                # Read in chunks
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.read, 4096
                )
                if not chunk:
                    break
                
                buffer += chunk
                
                # Process complete messages (newline-delimited JSON)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        await self._handle_message(line.decode("utf-8"))
                        
            except Exception as e:
                logger.error("Error reading from MCP server", error=str(e))
                break
    
    async def _handle_message(self, message: str) -> None:
        """Handle an incoming message from the server."""
        try:
            data = json.loads(message)
            
            # Check if this is a response to a pending request
            if "id" in data and data["id"] in self._pending_requests:
                future = self._pending_requests.pop(data["id"])
                if "error" in data:
                    future.set_exception(Exception(data["error"].get("message", "Unknown error")))
                else:
                    future.set_result(data.get("result"))
            
            # Handle notifications
            elif "method" in data and "id" not in data:
                await self._handle_notification(data)
                
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON from MCP server", error=str(e))
    
    async def _handle_notification(self, data: dict) -> None:
        """Handle a notification from the server."""
        method = data.get("method", "")
        params = data.get("params", {})
        
        logger.debug("MCP notification", method=method, params=params)
    
    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a JSON-RPC request to the server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Server not connected")
        
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        
        # Create a future for the response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        # Send the request
        message = json.dumps(request) + "\n"
        await asyncio.get_event_loop().run_in_executor(
            None, self._process.stdin.write, message.encode("utf-8")
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self._process.stdin.flush
        )
        
        # Wait for response with timeout
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out")
    
    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
            },
            "clientInfo": {
                "name": "squadron",
                "version": "0.1.0",
            },
        })
        
        logger.debug("MCP initialized", result=result)
        
        # Send initialized notification
        await self._send_request("notifications/initialized", {})
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        try:
            result = await self._send_request("tools/list", {})
            tools = result.get("tools", [])
            
            for tool_data in tools:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=self.name,
                )
                self._tools[tool.name] = tool
                
        except Exception as e:
            logger.warning("Failed to discover tools", error=str(e))
    
    async def _discover_resources(self) -> None:
        """Discover available resources from the server."""
        try:
            result = await self._send_request("resources/list", {})
            resources = result.get("resources", [])
            
            for res_data in resources:
                resource = MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", res_data["uri"]),
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "text/plain"),
                    server_name=self.name,
                )
                self._resources[resource.uri] = resource
                
        except Exception as e:
            logger.warning("Failed to discover resources", error=str(e))
    
    async def _discover_prompts(self) -> None:
        """Discover available prompts from the server."""
        try:
            result = await self._send_request("prompts/list", {})
            prompts = result.get("prompts", [])
            
            for prompt_data in prompts:
                prompt = MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description", ""),
                    arguments=prompt_data.get("arguments", []),
                    server_name=self.name,
                )
                self._prompts[prompt.name] = prompt
                
        except Exception as e:
            logger.warning("Failed to discover prompts", error=str(e))
    
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call a tool on this server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        
        return result
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource from this server."""
        result = await self._send_request("resources/read", {"uri": uri})
        
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""
    
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get a prompt from this server."""
        result = await self._send_request("prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        
        return result.get("messages", [])
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        self._connected = False
        logger.info("MCP server disconnected", name=self.name)
    
    @property
    def tools(self) -> list[MCPTool]:
        """Get all available tools."""
        return list(self._tools.values())
    
    @property
    def resources(self) -> list[MCPResource]:
        """Get all available resources."""
        return list(self._resources.values())
    
    @property
    def prompts(self) -> list[MCPPrompt]:
        """Get all available prompts."""
        return list(self._prompts.values())


class MCPHost:
    """
    MCP Host - Manages multiple MCP server connections.
    
    The Host provides a unified interface for:
    - Loading server configurations
    - Managing server lifecycles
    - Routing tool calls to the correct server
    - Aggregating tools across all servers
    
    Example:
        ```python
        host = MCPHost(config=MCPConfig())
        await host.load_servers("mcp_servers.json")
        
        # Get all available tools
        tools = host.get_all_tools()
        
        # Call a tool
        result = await host.call_tool("query_database", {"sql": "SELECT * FROM users"})
        ```
    """
    
    def __init__(self, config: MCPConfig | None = None):
        """
        Initialize the MCP Host.
        
        Args:
            config: MCP configuration
        """
        self.config = config or MCPConfig()
        self._servers: dict[str, MCPServer] = {}
        self._tool_index: dict[str, str] = {}  # tool_name -> server_name
        self._initialized = False
    
    async def load_servers(self, config_path: str | Path | None = None) -> None:
        """
        Load MCP servers from a configuration file.
        
        Config format:
        ```json
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
                },
                "database": {
                    "command": "python",
                    "args": ["-m", "mcp_server_postgres"],
                    "env": {"DATABASE_URL": "postgresql://..."}
                }
            }
        }
        ```
        """
        path = Path(config_path or self.config.servers_config_path)
        
        if not path.exists():
            logger.warning("MCP config file not found", path=str(path))
            return
        
        with open(path) as f:
            config_data = json.load(f)
        
        servers_config = config_data.get("mcpServers", {})
        
        for name, server_config in servers_config.items():
            await self.add_server(
                name=name,
                command=server_config.get("command", ""),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
            )
        
        self._initialized = True
        logger.info(
            "MCP servers loaded",
            count=len(self._servers),
            tools=len(self._tool_index),
        )
    
    async def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: MCPTransport = MCPTransport.STDIO,
        url: str | None = None,
    ) -> MCPServer:
        """
        Add and connect to an MCP server.
        
        Args:
            name: Unique server identifier
            command: Command to spawn the server
            args: Command arguments
            env: Environment variables
            transport: Communication protocol
            url: Server URL (for SSE/HTTP)
            
        Returns:
            The connected MCPServer instance
        """
        if name in self._servers:
            logger.warning("Server already exists, replacing", name=name)
            await self._servers[name].disconnect()
        
        server = MCPServer(
            name=name,
            command=command,
            args=args,
            env=env,
            transport=transport,
            url=url,
        )
        
        try:
            await server.connect()
            self._servers[name] = server
            
            # Index tools for quick lookup
            for tool in server.tools:
                if tool.name in self._tool_index:
                    logger.warning(
                        "Tool name collision",
                        tool=tool.name,
                        existing_server=self._tool_index[tool.name],
                        new_server=name,
                    )
                self._tool_index[tool.name] = name
            
            return server
            
        except Exception as e:
            logger.error("Failed to add MCP server", name=name, error=str(e))
            raise
    
    async def remove_server(self, name: str) -> None:
        """Remove and disconnect an MCP server."""
        if name not in self._servers:
            return
        
        server = self._servers.pop(name)
        await server.disconnect()
        
        # Remove tools from index
        self._tool_index = {
            tool: srv for tool, srv in self._tool_index.items()
            if srv != name
        }
    
    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for server in self._servers.values():
            tools.extend(server.tools)
        return tools
    
    def get_tools_openai_format(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self.get_all_tools()]
    
    def get_tools_anthropic_format(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic tool format."""
        return [tool.to_anthropic_format() for tool in self.get_all_tools()]
    
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call a tool by name.
        
        Automatically routes to the correct server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self._tool_index:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tool_index.keys())}")
        
        server_name = self._tool_index[name]
        server = self._servers[server_name]
        
        logger.debug("Calling MCP tool", tool=name, server=server_name)
        
        result = await server.call_tool(name, arguments)
        
        return result
    
    def get_tool(self, name: str) -> MCPTool | None:
        """Get a tool by name."""
        if name not in self._tool_index:
            return None
        
        server_name = self._tool_index[name]
        server = self._servers[server_name]
        return server._tools.get(name)
    
    async def close(self) -> None:
        """Disconnect all servers."""
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()
        self._tool_index.clear()
        logger.info("MCP Host closed")
    
    @property
    def servers(self) -> list[MCPServer]:
        """Get all connected servers."""
        return list(self._servers.values())
    
    @property
    def server_names(self) -> list[str]:
        """Get names of all connected servers."""
        return list(self._servers.keys())


def mcp_tool(
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as an MCP tool.
    
    This is a convenience decorator for creating local tools
    that can be exposed via an MCP server.
    
    Example:
        ```python
        @mcp_tool(description="Query the database")
        def query_database(sql: str) -> str:
            return execute_query(sql)
        ```
    """
    def decorator(func: F) -> F:
        func._mcp_tool = True  # type: ignore
        func._mcp_name = name or func.__name__  # type: ignore
        func._mcp_description = description or func.__doc__ or ""  # type: ignore
        func._mcp_requires_approval = requires_approval  # type: ignore
        return func
    return decorator
