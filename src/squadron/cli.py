"""
Command Line Interface for Squadron Agent Framework

Provides CLI commands for running agents, testing, and development.
"""

import asyncio
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from squadron import Agent, SquadronConfig
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner
from squadron.core.config import LLMConfig, ReasoningConfig, MemoryConfig

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

app = typer.Typer(
    help="Squadron Agent Framework CLI",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    name: str = typer.Option("squadron", "--name", "-n", help="Agent name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    max_iterations: int = typer.Option(10, "--iterations", "-i", help="Max iterations"),
    llm_provider: str = typer.Option("openai", "--llm", help="LLM provider"),
    model: str = typer.Option("gpt-4o", "--model", help="Model name"),
    toolpack: str = typer.Option(
        "general",
        "--toolpack",
        "-t",
        help="Which sample toolpack to load",
        case_sensitive=False,
        show_default=True,
    ),
):
    """
    Run a Squadron agent on a task.
    
    Example:
        squadron run "Analyze this codebase for security issues"
    """
    console.print("[bold blue]Squadron Agent Framework[/bold blue]")
    console.print(f"Task: {task}")
    console.print(f"Agent: {name}")
    console.print()
    
    # Configure the agent
    config = SquadronConfig()
    config.governance.max_iterations = max_iterations
    config.llm.provider = llm_provider  # type: ignore
    config.llm.model = model  # type: ignore
    
    if verbose:
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
        )
    
    # Create components
    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="echo" if toolpack == "general" else "list_directory",
    )
    
    # Create agent
    agent = Agent(
        name=name,
        config=config,
        memory=memory,
        reasoner=reasoner,
    )
    
    # Register basic tools
    _register_toolpack(agent, toolpack)
    
    # Run the agent
    try:
        asyncio.run(_run_agent_async(agent, task, session_id))
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent execution interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def config():
    """Show current configuration."""
    config = SquadronConfig()
    
    console.print("[bold]Squadron Configuration[/bold]\n")
    
    # LLM config
    llm_table = Table(title="LLM Configuration")
    llm_table.add_column("Setting", style="cyan")
    llm_table.add_column("Value", style="green")
    llm_table.add_row("Provider", config.llm.provider)
    llm_table.add_row("Model", config.llm.model)
    llm_table.add_row("Temperature", str(config.llm.temperature))
    llm_table.add_row("Max Tokens", str(config.llm.max_tokens))
    console.print(llm_table)
    
    # Reasoning config
    reasoning_table = Table(title="Reasoning Configuration")
    reasoning_table.add_column("Setting", style="cyan")
    reasoning_table.add_column("Value", style="green")
    reasoning_table.add_row("Candidates", str(config.reasoning.n_candidates))
    reasoning_table.add_row("Max Depth", str(config.reasoning.max_depth))
    reasoning_table.add_row("Verifier Model", config.reasoning.verifier_model)
    console.print(reasoning_table)
    
    # Governance config
    governance_table = Table(title="Governance Configuration")
    governance_table.add_column("Setting", style="cyan")
    governance_table.add_column("Value", style="green")
    governance_table.add_row("Max Iterations", str(config.governance.max_iterations))
    governance_table.add_row("Guardrails Enabled", str(config.governance.enable_guardrails))
    governance_table.add_row("Require Approval", str(config.governance.require_human_approval))
    console.print(governance_table)


@app.command()
def interactive(
    name: str = typer.Option("squadron", "--name", "-n", help="Agent name"),
):
    """
    Start an interactive session with an agent.
    
    Example:
        squadron interactive
    """
    console.print("[bold blue]Squadron Interactive Mode[/bold blue]")
    console.print("Type 'quit' or 'exit' to end the session.\n")
    
    config = SquadronConfig()
    memory = GraphitiMemory()
    reasoner = LATSReasoner(config=config.reasoning, memory=memory)
    
    agent = Agent(
        name=name,
        config=config,
        memory=memory,
        reasoner=reasoner,
    )
    
    _register_basic_tools(agent)
    
    session_id = None
    
    try:
        while True:
            task = typer.prompt("You", default="")
            
            if task.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not task.strip():
                continue
            
            console.print(f"\n[bold]Executing:[/bold] {task}")
            
            try:
                asyncio.run(_run_agent_async(agent, task, session_id))
                session_id = None  # Reset for next task
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            
            console.print()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted[/yellow]")


@app.command()
def demo():
    """Run a demo of the Squadron agent framework."""
    console.print("[bold blue]Squadron Agent Framework Demo[/bold blue]\n")
    
    demo_tasks = [
        "Analyze the project structure",
        "Find Python files in the codebase",
        "Check if there are any TODO comments",
        "List the top-level directories",
    ]
    
    config = SquadronConfig()
    config.governance.max_iterations = 5  # Limit for demo
    
    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="echo",
    )
    
    agent = Agent(
        name="demo-agent",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )
    
    _register_toolpack(agent, "general")
    
    for i, task in enumerate(demo_tasks, 1):
        console.print(f"\n[bold]Demo Task {i}/4:[/bold] {task}")
        console.print("-" * 50)
        
        try:
            asyncio.run(_run_agent_async(agent, task))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        
        if i < len(demo_tasks):
            console.print("\n[dim]Press Enter for next task...[/dim]")
            input()


async def _run_agent_async(agent: Agent, task: str, session_id: Optional[str] = None):
    """Run the agent asynchronously."""
    with console.status("[bold green]Agent thinking...") as status:
        final_state = await agent.run(task, session_id=session_id)
    
    # Display results
    console.print("\n[bold]Execution Results[/bold]")
    console.print("-" * 50)
    
    # Phase and iteration info
    console.print(f"Phase: {final_state.phase}")
    console.print(f"Iterations: {final_state.iteration}")
    
    # Messages
    if final_state.messages:
        console.print("\n[bold]Conversation[/bold]")
        for msg in final_state.messages[-5:]:  # Show last 5 messages
            role = msg.role.value.upper()
            console.print(f"[cyan]{role}[/cyan]: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
    
    # Tool results
    if final_state.tool_results:
        console.print("\n[bold]Tool Results[/bold]")
        for result in final_state.tool_results:
            status = "[green]SUCCESS[/green]" if result.success else "[red]FAILED[/red]"
            console.print(f"  {status} {result.tool_name}: {str(result.result)[:50]}{'...' if len(str(result.result)) > 50 else ''}")
    
    # Errors
    if final_state.errors:
        console.print("\n[bold red]Errors[/bold red]")
        for error in final_state.errors:
            console.print(f"  [red]{error}[/red]")
    
    # Final message
    if final_state.phase.value == "completed":
        console.print("\n[bold green]✓ Task completed successfully![/bold green]")
    elif final_state.phase.value == "error":
        console.print("\n[bold red]✗ Task failed with errors[/bold red]")


def _register_toolpack(agent: Agent, toolpack: str = "general"):
    """Register sample tools for demonstration."""

    import os
    from pathlib import Path
    
    async def list_directory(path: str = ".") -> str:
        """List files in a directory."""
        try:
            if path == ".":
                path = os.getcwd()
            
            items = list(Path(path).iterdir())
            return "\n".join([
                f"{'📁' if item.is_dir() else '📄'} {item.name}"
                for item in sorted(items)
            ])
        except Exception as e:
            return f"Error listing directory: {e}"
    
    async def read_file(filepath: str) -> str:
        """Read content of a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:1000] + ("..." if len(content) > 1000 else "")
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def search_files(pattern: str, path: str = ".") -> str:
        """Search for pattern in files."""
        import re
        try:
            matches = []
            search_path = Path(path)
            
            for file_path in search_path.rglob("*.py"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(pattern, content, re.IGNORECASE):
                            matches.append(str(file_path))
                except:
                    continue
            
            return f"Found pattern '{pattern}' in {len(matches)} files:\n" + "\n".join(matches[:10])
        except Exception as e:
            return f"Error searching files: {e}"
    
    async def run_command(command: str) -> str:
        """Run a shell command."""
        import subprocess
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            return f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout[:500]}\nSTDERR:\n{result.stderr[:500]}"
        except Exception as e:
            return f"Error running command: {e}"
    
    async def echo(text: str) -> str:
        """Echo text back to the user."""
        return text

    async def classify_intent(text: str) -> str:
        """Very naive intent classifier."""
        lowered = text.lower()
        if "todo" in lowered or "task" in lowered:
            return "task_management"
        if "error" in lowered or "bug" in lowered:
            return "debugging"
        if "write" in lowered or "draft" in lowered:
            return "writing"
        return "general"

    packs = {
        "general": [echo, classify_intent],
        "coding": [list_directory, read_file, search_files, run_command],
    }

    chosen = packs.get(toolpack.lower())
    if not chosen:
        chosen = packs["general"]
        console.print(f"[yellow]Unknown toolpack '{toolpack}', defaulting to general[/yellow]")

    for tool in chosen:
        agent.register_tool(tool)

    console.print(f"[dim]Registered toolpack: {toolpack.lower()} ({', '.join(t.__name__ for t in chosen)})[/dim]")


if __name__ == "__main__":
    app()
