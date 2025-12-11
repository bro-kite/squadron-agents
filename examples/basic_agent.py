#!/usr/bin/env python3
"""
Basic Squadron Agent Example

Demonstrates the core functionality with a simple coding agent.
"""

import asyncio
import os
from pathlib import Path

from squadron import Agent, SquadronConfig
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


async def main():
    """Run a basic Squadron agent."""
    
    print("🤖 Squadron Agent Framework - Basic Example")
    print("=" * 50)
    
    # Create configuration
    config = SquadronConfig()
    config.governance.max_iterations = 10
    
    # Create components
    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="list_python_files",
        tool_args_fn=lambda state: {"path": "."},
    )
    
    # Create agent
    agent = Agent(
        name="researcher-coder",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )
    
    # Register basic tools
    _register_coding_tools(agent)
    
    # Example tasks
    tasks = [
        "List the Python files in this project",
        "Read the main agent file and explain its structure",
        "Find any TODO comments in the codebase",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task}")
        print("-" * 50)
        
        try:
            final_state = await agent.run(task)
            
            print(f"✅ Phase: {final_state.phase.value}")
            print(f"🔄 Iterations: {final_state.iteration}")
            
            # Show conversation
            if final_state.messages:
                print("\n💬 Conversation:")
                for msg in final_state.messages[-3:]:
                    print(f"  {msg.role.value.upper()}: {msg.content[:80]}...")
            
            # Show tool results
            if final_state.tool_results:
                print("\n🔧 Tool Results:")
                for result in final_state.tool_results:
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {result.tool_name}")
            
            # Show any errors
            if final_state.errors:
                print("\n❌ Errors:")
                for error in final_state.errors:
                    print(f"  {error}")
            
            if final_state.phase.value == "completed":
                print("🎉 Task completed successfully!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Show memory statistics
    print("🧠 Memory Statistics:")
    print(f"  Entities: {memory.stats['total_entities']}")
    print(f"  Edges: {memory.stats['total_edges']}")
    print(f"  Facts: {memory.stats['total_facts']}")


def _register_coding_tools(agent: Agent):
    """Register coding-related tools."""
    
    async def list_python_files(path: str = ".") -> str:
        """List Python files in the project."""
        try:
            python_files = []
            search_path = Path(path)
            
            for py_file in search_path.rglob("*.py"):
                try:
                    rel_path = py_file.relative_to(search_path)
                    python_files.append(str(rel_path))
                except ValueError:
                    continue
            
            return f"Found {len(python_files)} Python files:\n" + "\n".join(python_files)
        except Exception as e:
            return f"Error listing Python files: {e}"
    
    async def read_code_file(filepath: str) -> str:
        """Read and analyze a code file."""
        try:
            if not Path(filepath).exists():
                return f"File {filepath} not found"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            analysis = f"File: {filepath}\n"
            analysis += f"Lines: {len(lines)}\n"
            analysis += f"Size: {len(content)} characters\n\n"
            
            # Show first 20 lines
            analysis += "First 20 lines:\n"
            analysis += "\n".join(f"{i+1:3}: {line}" for i, line in enumerate(lines[:20]))
            
            if len(lines) > 20:
                analysis += f"\n... ({len(lines) - 20} more lines)"
            
            return analysis
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def search_todos(path: str = ".") -> str:
        """Search for TODO comments in the codebase."""
        import re
        try:
            todos = []
            search_path = Path(path)
            
            for py_file in search_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if re.search(r'#\s*TODO|#\s*FIXME|#\s*HACK', line, re.IGNORECASE):
                                rel_path = py_file.relative_to(search_path)
                                todos.append(f"{rel_path}:{line_num} - {line.strip()}")
                except:
                    continue
            
            return f"Found {len(todos)} TODO/FIXME comments:\n" + "\n".join(todos)
        except Exception as e:
            return f"Error searching TODOs: {e}"
    
    async def analyze_structure(path: str = ".") -> str:
        """Analyze project structure."""
        try:
            search_path = Path(path)
            
            # Count files by extension
            ext_counts = {}
            py_files = []
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix or "no_ext"
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    
                    if ext == ".py":
                        py_files.append(file_path)
            
            # Build analysis
            analysis = "Project Structure Analysis:\n"
            analysis += f"Total file types: {len(ext_counts)}\n\n"
            
            analysis += "Files by extension:\n"
            for ext, count in sorted(ext_counts.items()):
                analysis += f"  {ext or 'no_ext'}: {count} files\n"
            
            analysis += f"\nPython files: {len(py_files)}"
            
            # Top-level structure
            analysis += "\n\nTop-level structure:\n"
            for item in sorted(search_path.iterdir()):
                if item.name.startswith('.'):
                    continue
                analysis += f"  {'📁' if item.is_dir() else '📄'} {item.name}\n"
            
            return analysis
        except Exception as e:
            return f"Error analyzing structure: {e}"
    
    # Register tools
    agent.register_tool(list_python_files)
    agent.register_tool(read_code_file)
    agent.register_tool(search_todos)
    agent.register_tool(analyze_structure)
    
    print("🔧 Registered coding tools")


if __name__ == "__main__":
    asyncio.run(main())
