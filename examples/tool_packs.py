#!/usr/bin/env python3
"""
Tool Packs Example

Demonstrates how to use pre-built tool packs:
- CodingTools: File operations, code search, git
- ResearchTools: Web search, URL reading, summarization
- OpsTools: Shell commands, Docker, system monitoring
"""

import asyncio
import tempfile
from pathlib import Path

from squadron import Agent, SquadronConfig, CodingTools, ResearchTools, OpsTools
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


# =============================================================================
# Example: CodingTools
# =============================================================================

async def example_coding_tools():
    """Example using CodingTools pack."""
    print("\n" + "=" * 50)
    print("💻 CodingTools Example")
    print("=" * 50)
    
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        Path(tmpdir, "main.py").write_text('''
"""Main module."""

def hello(name: str) -> str:
    """Say hello."""
    # TODO: Add logging
    return f"Hello, {name}!"

def main():
    print(hello("World"))

if __name__ == "__main__":
    main()
''')
        
        Path(tmpdir, "utils.py").write_text('''
"""Utility functions."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    # FIXME: Handle overflow
    return a * b
''')
        
        # Create CodingTools
        tools = CodingTools(workspace_root=tmpdir)
        
        print(f"\n📁 Workspace: {tmpdir}")
        print(f"🔧 Available tools: {[t.name for t in tools.get_tools()]}")
        
        # Read a file
        print("\n📖 Reading main.py:")
        content = await tools.read_file("main.py")
        print(content[:200])
        
        # Search for patterns
        print("\n🔍 Searching for 'def ':")
        matches = await tools.grep("def ")
        for match in matches[:5]:
            print(f"  {match}")
        
        # Find files
        print("\n📂 Finding Python files:")
        files = await tools.find_files("*.py")
        for f in files:
            print(f"  {f}")
        
        # Search for TODOs
        print("\n📝 Finding TODOs and FIXMEs:")
        todos = await tools.grep("TODO|FIXME")
        for todo in todos:
            print(f"  {todo}")
        
        # Write a new file
        print("\n✍️ Writing new file:")
        result = await tools.write_file("config.py", "DEBUG = True\nVERSION = '1.0.0'")
        print(f"  {result}")


# =============================================================================
# Example: ResearchTools
# =============================================================================

async def example_research_tools():
    """Example using ResearchTools pack."""
    print("\n" + "=" * 50)
    print("🔬 ResearchTools Example")
    print("=" * 50)
    
    # Create ResearchTools
    # Note: Web search requires API keys (SERPER_API_KEY, TAVILY_API_KEY, etc.)
    tools = ResearchTools()
    
    print(f"🔧 Available tools: {[t.name for t in tools.get_tools()]}")
    
    # Summarize text
    print("\n📝 Summarizing text:")
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" 
    cognitive skills that are associated with the human mind, such as "learning" and 
    "problem-solving". This definition has since been rejected by major AI researchers 
    who now describe AI in terms of rationality and acting rationally, which does not 
    limit how intelligence can be articulated.
    """ * 3
    
    summary = await tools.summarize(long_text, max_length=150)
    print(f"  Original: {len(long_text)} chars")
    print(f"  Summary: {summary[:150]}...")
    
    # Extract information
    print("\n🔎 Extracting information:")
    text = "Contact John Smith at john@example.com or call 555-1234. Visit https://example.com for more info."
    info = await tools.extract_info(text)
    print(f"  Extracted: {info}")
    
    # Web search (requires API key)
    print("\n🌐 Web search:")
    print("  Note: Requires SERPER_API_KEY or similar")
    # results = await tools.web_search("Python async programming")
    # for r in results[:3]:
    #     print(f"  - {r.title}: {r.url}")


# =============================================================================
# Example: OpsTools
# =============================================================================

async def example_ops_tools():
    """Example using OpsTools pack."""
    print("\n" + "=" * 50)
    print("⚙️ OpsTools Example")
    print("=" * 50)
    
    # Create OpsTools with safety restrictions
    tools = OpsTools(
        allowed_commands=["echo", "ls", "cat", "pwd", "date", "whoami", "uname"],
        blocked_commands=["rm", "sudo", "chmod", "chown"],
    )
    
    print(f"🔧 Available tools: {[t.name for t in tools.get_tools()]}")
    print(f"✅ Allowed commands: {tools.allowed_commands}")
    print(f"❌ Blocked commands: {tools.blocked_commands}")
    
    # Run safe commands
    print("\n🖥️ Running commands:")
    
    result = await tools.run_command("echo 'Hello from OpsTools!'")
    print(f"  echo: {result.stdout.strip()}")
    
    result = await tools.run_command("pwd")
    print(f"  pwd: {result.stdout.strip()}")
    
    result = await tools.run_command("date")
    print(f"  date: {result.stdout.strip()}")
    
    # Get system info
    print("\n💻 System information:")
    info = await tools.get_system_info()
    print(f"  {info[:200]}...")
    
    # List processes
    print("\n📊 Top processes:")
    processes = await tools.list_processes()
    for proc in processes[:5]:
        print(f"  PID {proc.pid}: {proc.name} (CPU: {proc.cpu_percent}%)")
    
    # Try a blocked command
    print("\n🚫 Trying blocked command:")
    result = await tools.run_command("rm -rf /")
    print(f"  Result: {result.error}")


# =============================================================================
# Example: Agent with Tool Packs
# =============================================================================

async def example_agent_with_tool_packs():
    """Example using tool packs with an agent."""
    print("\n" + "=" * 50)
    print("🤖 Agent with Tool Packs Example")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "app.py").write_text("print('Hello')")
        Path(tmpdir, "README.md").write_text("# My Project\n\nA sample project.")
        
        # Create tool packs
        coding_tools = CodingTools(workspace_root=tmpdir)
        ops_tools = OpsTools(allowed_commands=["echo", "ls", "cat"])
        
        # Create agent
        config = SquadronConfig()
        config.governance.max_iterations = 5
        
        memory = GraphitiMemory()
        reasoner = LATSReasoner(
            config=config.reasoning,
            memory=memory,
            default_tool="read_file",
        )
        
        agent = Agent(
            name="toolpack-agent",
            config=config,
            memory=memory,
            reasoner=reasoner,
        )
        
        # Register tool packs
        agent.register_tool_pack(coding_tools)
        agent.register_tool_pack(ops_tools)
        
        print(f"\n🔧 Agent has {len(agent.tools)} tools registered")
        print(f"   Tools: {list(agent._tool_registry.keys())[:10]}...")
        
        # Run a task
        print("\n📋 Running task: 'Read the README file'")
        try:
            final_state = await agent.run("Read the README.md file and tell me what the project is about")
            
            print(f"✅ Phase: {final_state.phase.value}")
            if final_state.tool_results:
                for result in final_state.tool_results:
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {result.tool_name}")
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# Example: Custom Tool Pack
# =============================================================================

async def example_custom_tool_pack():
    """Example creating a custom tool pack."""
    print("\n" + "=" * 50)
    print("🛠️ Custom Tool Pack Example")
    print("=" * 50)
    
    print("""
To create a custom tool pack:

```python
from squadron.connectivity.mcp_host import mcp_tool, MCPTool

class MyToolPack:
    def __init__(self, config=None):
        self.config = config
    
    def get_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="my_tool",
                description="My custom tool",
                input_schema={...},
                handler=self.my_tool,
            ),
        ]
    
    @mcp_tool(description="My custom tool")
    async def my_tool(self, arg: str) -> str:
        return f"Result: {arg}"

# Use with agent
tools = MyToolPack()
agent.register_tool_pack(tools)
```

Tool packs should:
1. Have a `get_tools()` method returning MCPTool objects
2. Use the `@mcp_tool` decorator for automatic schema generation
3. Handle errors gracefully and return informative messages
""")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all tool pack examples."""
    print("🚀 Squadron Tool Packs Examples")
    print("=" * 50)
    
    try:
        await example_coding_tools()
        await example_research_tools()
        await example_ops_tools()
        await example_agent_with_tool_packs()
        await example_custom_tool_pack()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Tool pack examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
