"""
MCP Client Implementation

A lightweight client for connecting to remote MCP servers via SSE or HTTP.
Used when the MCP server is hosted externally (not spawned locally).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urljoin

import structlog

from squadron.connectivity.mcp_host import MCPTool, MCPResource, MCPPrompt

logger = structlog.get_logger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for MCP client connection."""
    base_url: str
    api_key: str | None = None
    timeout: float = 30.0
    headers: dict[str, str] | None = None


class MCPClient:
    """
    MCP Client for remote server connections.
    
    Connects to MCP servers via HTTP/SSE rather than spawning
    a local process. Useful for shared infrastructure.
    
    Example:
        ```python
        client = MCPClient(
            config=MCPClientConfig(
                base_url="https://mcp.example.com",
                api_key="your-api-key",
            )
        )
        await client.connect()
        
        tools = client.tools
        result = await client.call_tool("search", {"query": "hello"})
        ```
    """
    
    def __init__(self, config: MCPClientConfig):
        """
        Initialize the MCP client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self._session: Any = None  # aiohttp.ClientSession
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._connected = False
        self._sse_task: asyncio.Task | None = None
    
    async def connect(self) -> None:
        """Establish connection to the remote MCP server."""
        if self._connected:
            return
        
        try:
            # Import aiohttp lazily
            import aiohttp
            
            # Build headers
            headers = dict(self.config.headers or {})
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            # Create session
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
            
            # Initialize connection
            await self._initialize()
            
            # Discover capabilities
            await self._discover_tools()
            await self._discover_resources()
            await self._discover_prompts()
            
            self._connected = True
            logger.info(
                "MCP client connected",
                url=self.config.base_url,
                tools=len(self._tools),
            )
            
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            raise
        except Exception as e:
            logger.error("Failed to connect to MCP server", error=str(e))
            raise
    
    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        result = await self._post("/initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "squadron",
                "version": "0.1.0",
            },
        })
        logger.debug("MCP client initialized", result=result)
    
    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send a POST request to the server."""
        if not self._session:
            raise RuntimeError("Client not connected")
        
        async with self._session.post(path, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def _get(self, path: str) -> dict[str, Any]:
        """Send a GET request to the server."""
        if not self._session:
            raise RuntimeError("Client not connected")
        
        async with self._session.get(path) as response:
            response.raise_for_status()
            return await response.json()
    
    async def _discover_tools(self) -> None:
        """Discover available tools."""
        try:
            result = await self._get("/tools/list")
            tools = result.get("tools", [])
            
            for tool_data in tools:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name="remote",
                )
                self._tools[tool.name] = tool
                
        except Exception as e:
            logger.warning("Failed to discover tools", error=str(e))
    
    async def _discover_resources(self) -> None:
        """Discover available resources."""
        try:
            result = await self._get("/resources/list")
            resources = result.get("resources", [])
            
            for res_data in resources:
                resource = MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", res_data["uri"]),
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "text/plain"),
                    server_name="remote",
                )
                self._resources[resource.uri] = resource
                
        except Exception as e:
            logger.warning("Failed to discover resources", error=str(e))
    
    async def _discover_prompts(self) -> None:
        """Discover available prompts."""
        try:
            result = await self._get("/prompts/list")
            prompts = result.get("prompts", [])
            
            for prompt_data in prompts:
                prompt = MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description", ""),
                    arguments=prompt_data.get("arguments", []),
                    server_name="remote",
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
        Call a tool on the remote server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        
        result = await self._post("/tools/call", {
            "name": name,
            "arguments": arguments,
        })
        
        return result
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource from the server."""
        result = await self._post("/resources/read", {"uri": uri})
        
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""
    
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get a prompt from the server."""
        result = await self._post("/prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        
        return result.get("messages", [])
    
    async def subscribe_sse(self) -> AsyncIterator[dict[str, Any]]:
        """
        Subscribe to server-sent events.
        
        Yields:
            Event data dictionaries
        """
        if not self._session:
            raise RuntimeError("Client not connected")
        
        async with self._session.get("/sse") as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        self._connected = False
        logger.info("MCP client disconnected")
    
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
