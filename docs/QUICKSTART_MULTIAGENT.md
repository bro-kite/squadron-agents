---
date created: 2025-12-11
date updated: 2025-12-11 17:42 UTC
---

# Quick Start: Multi-Agent Systems

Get a multi-agent system running in **under 5 minutes**.

## Prerequisites

```bash
# 1. Install Squadron
pip install squadron-agent

# 2. Set your LLM API key (pick one)
export OPENAI_API_KEY=sk-...
# OR
export ANTHROPIC_API_KEY=sk-ant-...
# OR run Ollama locally (no key needed)
```

That's it. No Neo4j, no servers, no complex setup.

## Fastest Path: Pipeline Pattern

The simplest multi-agent pattern is a **pipeline** where agents pass results to each other:

```python
import asyncio
from squadron import Agent, SquadronConfig, create_llm
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner

async def main():
    # Create agents with different specializations
    researcher = quick_agent("researcher", {
        "search": lambda q: f"Found info about: {q}",
    })
    
    analyst = quick_agent("analyst", {
        "analyze": lambda data: f"Key insight: {data}",
    })
    
    writer = quick_agent("writer", {
        "write": lambda points: f"Summary: {points}",
    })
    
    # Pipeline: each agent builds on the previous
    task = "Explain quantum computing"
    
    research = await researcher.run(f"Research: {task}")
    analysis = await analyst.run(f"Analyze: {research.messages[-1].content}")
    final = await writer.run(f"Write about: {analysis.messages[-1].content}")
    
    print(final.messages[-1].content)

def quick_agent(name: str, tools: dict) -> Agent:
    """Create an agent in one line."""
    config = SquadronConfig()
    memory = GraphitiMemory()
    agent = Agent(
        name=name,
        config=config,
        memory=memory,
        reasoner=LATSReasoner(config=config.reasoning, memory=memory),
    )
    for tool_name, func in tools.items():
        async def wrapper(*args, f=func, **kwargs): return f(*args, **kwargs)
        wrapper.__name__ = tool_name
        agent.register_tool(wrapper)
    return agent

asyncio.run(main())
```

## Common Multi-Agent Patterns

### 1. Pipeline (Sequential)

Agents work in sequence, each building on the previous:

```
[Researcher] → [Analyst] → [Writer] → Output
```

Best for: Content creation, data processing, report generation

### 2. Parallel (Fan-out/Fan-in)

Multiple agents work simultaneously, results are combined:

```
              ┌─[Agent A]─┐
[Task] ───────┼─[Agent B]─┼───→ [Combiner] → Output
              └─[Agent C]─┘
```

```python
async def parallel_agents(task: str, agents: list[Agent]) -> list:
    """Run multiple agents in parallel."""
    results = await asyncio.gather(*[
        agent.run(task) for agent in agents
    ])
    return results
```

Best for: Research from multiple sources, getting diverse perspectives

### 3. Router (Conditional)

A router agent decides which specialist to use:

```
              ┌─[Code Agent]
[Router] ─────┼─[Research Agent]
              └─[Writing Agent]
```

```python
async def route_task(task: str, router: Agent, specialists: dict[str, Agent]):
    """Route task to appropriate specialist."""
    # Router decides which specialist
    decision = await router.run(f"Classify this task: {task}")
    specialist_name = extract_classification(decision)
    
    # Delegate to specialist
    specialist = specialists[specialist_name]
    return await specialist.run(task)
```

Best for: General assistants, customer service, task delegation

### 4. Debate (Adversarial)

Agents argue different positions, a judge decides:

```
[Advocate A] ──┐
               ├──→ [Judge] → Decision
[Advocate B] ──┘
```

```python
async def debate(topic: str, advocate_a: Agent, advocate_b: Agent, judge: Agent):
    """Two agents debate, judge decides."""
    position_a = await advocate_a.run(f"Argue FOR: {topic}")
    position_b = await advocate_b.run(f"Argue AGAINST: {topic}")
    
    verdict = await judge.run(
        f"Judge this debate:\nFOR: {position_a}\nAGAINST: {position_b}"
    )
    return verdict
```

Best for: Decision making, exploring trade-offs, red-teaming

## Full Example: Research Team

```python
"""
A team of agents that researches a topic and produces a report.
"""

import asyncio
from squadron import Agent, SquadronConfig, create_llm
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner
from squadron.tools import ResearchTools

async def research_team(topic: str) -> str:
    """
    Multi-agent research team:
    1. Planner: Creates research plan
    2. Researchers (x3): Gather information in parallel
    3. Synthesizer: Combines findings
    4. Editor: Polishes final output
    """
    
    config = SquadronConfig()
    
    # Create specialized agents
    planner = create_agent("planner", config)
    researchers = [create_agent(f"researcher_{i}", config) for i in range(3)]
    synthesizer = create_agent("synthesizer", config)
    editor = create_agent("editor", config)
    
    # Step 1: Plan the research
    print("📋 Planning research...")
    plan = await planner.run(
        f"Create a research plan for: {topic}. "
        f"Break it into 3 subtopics for parallel research."
    )
    subtopics = extract_subtopics(plan)
    
    # Step 2: Research in parallel
    print("🔍 Researching in parallel...")
    research_tasks = [
        researcher.run(f"Research: {subtopic}")
        for researcher, subtopic in zip(researchers, subtopics)
    ]
    research_results = await asyncio.gather(*research_tasks)
    
    # Step 3: Synthesize findings
    print("🔗 Synthesizing findings...")
    combined = "\n\n".join([
        f"## {subtopic}\n{get_content(result)}"
        for subtopic, result in zip(subtopics, research_results)
    ])
    synthesis = await synthesizer.run(
        f"Synthesize these research findings into a coherent report:\n{combined}"
    )
    
    # Step 4: Edit and polish
    print("✨ Editing final report...")
    final = await editor.run(
        f"Edit and polish this report for clarity:\n{get_content(synthesis)}"
    )
    
    return get_content(final)

def create_agent(name: str, config: SquadronConfig) -> Agent:
    memory = GraphitiMemory()
    return Agent(
        name=name,
        config=config,
        memory=memory,
        reasoner=LATSReasoner(config=config.reasoning, memory=memory),
    )

def extract_subtopics(plan_result) -> list[str]:
    # Simple extraction - in production, use structured output
    content = get_content(plan_result)
    # Return 3 subtopics (simplified)
    return [f"Subtopic {i+1}" for i in range(3)]

def get_content(state) -> str:
    if state.messages:
        return state.messages[-1].content
    return ""

# Run it
if __name__ == "__main__":
    report = asyncio.run(research_team("The future of AI agents"))
    print("\n" + "="*50)
    print("📄 FINAL REPORT:")
    print("="*50)
    print(report)
```

## What You Need to Get Started

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12+ |
| LLM API Key | Any one provider | OpenAI or Anthropic |
| Memory | None (in-memory) | Neo4j for persistence |
| Hardware | Any | GPU for local models |

## Next Steps

1. **Run the example**: `python examples/quick_multiagent.py`
2. **Try different patterns**: Pipeline, parallel, router, debate
3. **Add real tools**: Use `CodingTools`, `ResearchTools`, `OpsTools`
4. **Scale up**: Add A2A for distributed agents (see `examples/multi_agent_a2a.py`)

## Troubleshooting

**"No module named squadron"**
```bash
pip install squadron-agent
```

**"No API key found"**
```bash
export OPENAI_API_KEY=sk-...
# Or use Ollama for free local models
```

**"Agent not responding"**
- Check your API key is valid
- Try a simpler task first
- Enable debug logging: `export LOG_LEVEL=DEBUG`
