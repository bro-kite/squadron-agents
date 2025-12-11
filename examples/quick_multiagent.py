"""
Quick Multi-Agent Example

This is the FASTEST way to get a multi-agent system running with Squadron.
No servers, no complex setup - just agents collaborating locally.

Requirements:
    pip install squadron-agent
    export OPENAI_API_KEY=sk-...  (or any supported LLM)

Run:
    python quick_multiagent.py
"""

import asyncio
from squadron import Agent, SquadronConfig, create_llm
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


async def main():
    """
    Create a simple multi-agent system where:
    - Researcher: Gathers information
    - Analyst: Analyzes the information
    - Writer: Produces final output
    
    They collaborate by passing results to each other.
    """
    
    print("🚀 Squadron Quick Multi-Agent Demo")
    print("=" * 50)
    
    # Shared config and LLM (can be different per agent)
    config = SquadronConfig()
    llm = create_llm()  # Auto-detects from environment
    
    # =========================================
    # STEP 1: Create specialized agents
    # =========================================
    
    researcher = create_agent("researcher", config, llm, tools={
        "search": lambda query: f"[Research results for: {query}] Found 3 relevant sources about AI agents.",
        "fetch": lambda url: f"[Content from {url}] Detailed information about the topic.",
    })
    
    analyst = create_agent("analyst", config, llm, tools={
        "analyze": lambda data: f"[Analysis] Key insights: {data[:100]}... shows promising patterns.",
        "compare": lambda a, b: f"[Comparison] {a} vs {b}: Both have merits, but {a} is more relevant.",
    })
    
    writer = create_agent("writer", config, llm, tools={
        "draft": lambda topic, points: f"[Draft] Article about {topic} covering: {points}",
        "refine": lambda text: f"[Refined] {text} (improved clarity and flow)",
    })
    
    # =========================================
    # STEP 2: Define the collaboration workflow
    # =========================================
    
    task = "Write a brief summary about AI agent frameworks"
    
    print(f"\n📋 Task: {task}")
    print("-" * 50)
    
    # Researcher gathers information
    print("\n🔍 Researcher working...")
    research_result = await researcher.run(
        f"Research the topic: {task}. Use search and fetch tools."
    )
    research_output = get_last_result(research_result)
    print(f"   Result: {research_output[:100]}...")
    
    # Analyst processes the research
    print("\n📊 Analyst working...")
    analysis_result = await analyst.run(
        f"Analyze this research: {research_output}"
    )
    analysis_output = get_last_result(analysis_result)
    print(f"   Result: {analysis_output[:100]}...")
    
    # Writer produces final output
    print("\n✍️ Writer working...")
    writing_result = await writer.run(
        f"Write a summary based on this analysis: {analysis_output}"
    )
    final_output = get_last_result(writing_result)
    print(f"   Result: {final_output[:100]}...")
    
    # =========================================
    # STEP 3: Show final result
    # =========================================
    
    print("\n" + "=" * 50)
    print("📄 FINAL OUTPUT:")
    print("=" * 50)
    print(final_output)
    
    return final_output


def create_agent(name: str, config: SquadronConfig, llm, tools: dict) -> Agent:
    """Helper to quickly create an agent with tools."""
    
    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
    )
    
    agent = Agent(
        name=name,
        config=config,
        llm=llm,
        memory=memory,
        reasoner=reasoner,
    )
    
    # Register tools
    for tool_name, tool_func in tools.items():
        # Wrap sync functions as async
        if not asyncio.iscoroutinefunction(tool_func):
            original = tool_func
            async def async_wrapper(*args, _f=original, **kwargs):
                return _f(*args, **kwargs)
            async_wrapper.__name__ = tool_name
            agent.register_tool(async_wrapper)
        else:
            tool_func.__name__ = tool_name
            agent.register_tool(tool_func)
    
    return agent


def get_last_result(state) -> str:
    """Extract the last meaningful result from agent state."""
    if state.tool_results:
        return str(state.tool_results[-1].result)
    if state.messages:
        return state.messages[-1].content
    return "No result"


# =============================================
# EVEN SIMPLER: One-liner multi-agent
# =============================================

async def pipeline(task: str, agents: list[tuple[str, Agent]]) -> str:
    """
    Run a task through a pipeline of agents.
    
    Usage:
        result = await pipeline("Write about AI", [
            ("research", researcher),
            ("analyze", analyst),
            ("write", writer),
        ])
    """
    context = task
    for step_name, agent in agents:
        print(f"  [{step_name}] Processing...")
        result = await agent.run(context)
        context = get_last_result(result)
    return context


if __name__ == "__main__":
    asyncio.run(main())
