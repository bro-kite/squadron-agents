#!/usr/bin/env python3
"""
MCP Tools Example

Demonstrates how to use the Model Context Protocol (MCP) for tool integration:
- Creating local MCP tools
- Loading tools from MCP servers
- Using tools with agents
"""

import asyncio
import json
from pathlib import Path

from squadron import Agent, MCPHost, SquadronConfig
from squadron.connectivity.mcp_host import mcp_tool, MCPServer
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


# =============================================================================
# Define Local MCP Tools
# =============================================================================

@mcp_tool(description="Read a file from the filesystem")
async def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp_tool(description="Write content to a file")
async def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp_tool(description="List files in a directory", requires_approval=False)
async def list_directory(path: str = ".") -> str:
    """List files and directories in the given path."""
    try:
        items = []
        for item in Path(path).iterdir():
            icon = "📁" if item.is_dir() else "📄"
            items.append(f"{icon} {item.name}")
        return "\n".join(sorted(items)) if items else "Empty directory"
    except Exception as e:
        return f"Error listing directory: {e}"


@mcp_tool(
    description="Execute a shell command (requires approval)",
    requires_approval=True,
)
async def run_command(command: str) -> str:
    """Execute a shell command and return the output."""
    import subprocess
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error executing command: {e}"


@mcp_tool(description="Search for text in files")
async def grep(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    """Search for a pattern in files."""
    import re
    
    results = []
    search_path = Path(path)
    
    try:
        for file_path in search_path.rglob(file_pattern):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                results.append(f"{file_path}:{line_num}: {line.strip()}")
                except:
                    continue
        
        return "\n".join(results[:50]) if results else "No matches found"
    except Exception as e:
        return f"Error searching: {e}"


# =============================================================================
# Example: Using MCPHost with Local Tools
# =============================================================================

async def example_local_tools():
    """Example using locally defined MCP tools."""
    print("\n" + "=" * 50)
    print("🔧 Local MCP Tools Example")
    print("=" * 50)
    
    # Create MCP host
    mcp = MCPHost()
    
    # Register local tools
    mcp.register_local_tool(read_file)
    mcp.register_local_tool(write_file)
    mcp.register_local_tool(list_directory)
    mcp.register_local_tool(grep)
    
    # List available tools
    tools = mcp.get_all_tools()
    print(f"\nRegistered {len(tools)} tools:")
    for tool in tools:
        approval = "⚠️ " if tool.requires_approval else ""
        print(f"  {approval}{tool.name}: {tool.description}")
    
    # Call tools directly
    print("\n📂 Listing current directory:")
    result = await mcp.call_tool("list_directory", {"path": "."})
    print(result[:500])
    
    print("\n🔍 Searching for 'import' in Python files:")
    result = await mcp.call_tool("grep", {
        "pattern": "^import",
        "path": ".",
        "file_pattern": "*.py",
    })
    print(result[:500])


# =============================================================================
# Example: Loading Tools from MCP Server Config
# =============================================================================

async def example_server_config():
    """Example loading MCP servers from configuration."""
    print("\n" + "=" * 50)
    print("📡 MCP Server Configuration Example")
    print("=" * 50)
    
    # Example MCP server configuration
    config = {
        "servers": [
            {
                "name": "filesystem",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
            {
                "name": "github",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_TOKEN": "${GITHUB_TOKEN}",
                },
            },
        ]
    }
    
    print("Example MCP server configuration:")
    print(json.dumps(config, indent=2))
    
    print("\nTo use this configuration:")
    print("1. Save it to 'mcp_servers.json'")
    print("2. Load with: await mcp.load_servers('mcp_servers.json')")
    print("3. Tools from all servers will be available")


# =============================================================================
# Example: Agent with MCP Tools
# =============================================================================

async def example_agent_with_mcp():
    """Example using MCP tools with an agent."""
    print("\n" + "=" * 50)
    print("🤖 Agent with MCP Tools Example")
    print("=" * 50)
    
    # Create MCP host with tools
    mcp = MCPHost()
    mcp.register_local_tool(read_file)
    mcp.register_local_tool(list_directory)
    mcp.register_local_tool(grep)
    
    # Create agent components
    config = SquadronConfig()
    config.governance.max_iterations = 5
    
    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="list_directory",
    )
    
    # Create agent
    agent = Agent(
        name="mcp-agent",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )
    
    # Load MCP tools into agent
    await agent.load_mcp_tools(mcp)
    
    print(f"\n🔧 Agent has {len(agent.tools)} tools loaded")
    
    # Run a task
    print("\n📋 Running task: 'List the Python files in this directory'")
    try:
        final_state = await agent.run("List the Python files in this directory")
        
        print(f"✅ Phase: {final_state.phase.value}")
        if final_state.tool_results:
            for result in final_state.tool_results:
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.tool_name}")
    except Exception as e:
        print(f"❌ Error: {e}")


# =============================================================================
# Example: Custom MCP Server
# =============================================================================

async def example_custom_server():
    """Example creating a custom MCP server."""
    print("\n" + "=" * 50)
    print("🏗️ Custom MCP Server Example")
    print("=" * 50)
    
    print("""
To create a custom MCP server, you can:

1. Use the MCP SDK (Python or TypeScript):

   ```python
   from mcp import Server
   
   server = Server("my-server")
   
   @server.tool("my_tool")
   async def my_tool(arg: str) -> str:
       return f"Result: {arg}"
   
   server.run()
   ```

2. Or expose tools via HTTP with MCPClient:

   ```python
   from squadron import MCPClient
   
   client = MCPClient(base_url="http://my-server:8080")
   tools = await client.discover_tools()
   result = await client.call_tool("my_tool", {"arg": "value"})
   ```

3. Squadron's MCPHost can connect to any MCP-compatible server
   via stdio, SSE, or HTTP.
""")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all MCP examples."""
    print("🚀 Squadron MCP Tools Examples")
    print("=" * 50)
    
    try:
        await example_local_tools()
        await example_server_config()
        await example_agent_with_mcp()
        await example_custom_server()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ MCP examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
