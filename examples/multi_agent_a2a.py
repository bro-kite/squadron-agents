#!/usr/bin/env python3
"""
Multi-Agent A2A Example

Demonstrates the Agent-to-Agent (A2A) protocol for multi-agent coordination:
- Creating agents with capabilities
- Agent discovery via Agent Cards
- Task delegation between agents
- Running an A2A server
"""

import asyncio
from uuid import uuid4

from squadron.connectivity.a2a import (
    A2AAgent,
    AgentCard,
    AgentCapability,
    A2ATask,
    TaskState,
    A2AClient,
)


# =============================================================================
# Define Agent Capabilities
# =============================================================================

async def search_handler(task: A2ATask) -> dict:
    """Handle search capability."""
    query = task.input_data.get("query", "")
    print(f"  🔍 Searching for: {query}")
    
    # Simulate search results
    await asyncio.sleep(0.5)
    
    return {
        "results": [
            {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
            {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"},
            {"title": f"Result 3 for '{query}'", "url": "https://example.com/3"},
        ],
        "total": 3,
    }


async def summarize_handler(task: A2ATask) -> dict:
    """Handle summarization capability."""
    text = task.input_data.get("text", "")
    max_length = task.input_data.get("max_length", 100)
    print(f"  📝 Summarizing {len(text)} characters...")
    
    # Simulate summarization
    await asyncio.sleep(0.3)
    
    summary = text[:max_length] + "..." if len(text) > max_length else text
    return {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
    }


async def analyze_handler(task: A2ATask) -> dict:
    """Handle analysis capability."""
    data = task.input_data.get("data", {})
    analysis_type = task.input_data.get("type", "general")
    print(f"  📊 Analyzing data ({analysis_type})...")
    
    # Simulate analysis
    await asyncio.sleep(0.4)
    
    return {
        "analysis_type": analysis_type,
        "findings": [
            "Finding 1: Data shows positive trend",
            "Finding 2: Anomaly detected in subset",
            "Finding 3: Correlation with external factors",
        ],
        "confidence": 0.85,
    }


async def code_review_handler(task: A2ATask) -> dict:
    """Handle code review capability."""
    code = task.input_data.get("code", "")
    language = task.input_data.get("language", "python")
    print(f"  🔍 Reviewing {language} code...")
    
    # Simulate code review
    await asyncio.sleep(0.5)
    
    return {
        "language": language,
        "issues": [
            {"severity": "warning", "message": "Consider adding type hints"},
            {"severity": "info", "message": "Good use of docstrings"},
        ],
        "score": 8.5,
        "approved": True,
    }


# =============================================================================
# Example: Creating A2A Agents
# =============================================================================

async def example_create_agents():
    """Example creating A2A agents with capabilities."""
    print("\n" + "=" * 50)
    print("🤖 Creating A2A Agents")
    print("=" * 50)
    
    # Create a research agent
    research_agent = A2AAgent(
        card=AgentCard(
            id="research-agent",
            name="Research Agent",
            description="Performs web research and information gathering",
            url="http://localhost:8001",
            capabilities=[
                AgentCapability(
                    name="search",
                    description="Search the web for information",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
                AgentCapability(
                    name="summarize",
                    description="Summarize text content",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "max_length": {"type": "integer", "default": 100},
                        },
                        "required": ["text"],
                    },
                ),
            ],
        )
    )
    
    # Register capability handlers
    research_agent.register_capability("search", search_handler)
    research_agent.register_capability("summarize", summarize_handler)
    
    # Create an analysis agent
    analysis_agent = A2AAgent(
        card=AgentCard(
            id="analysis-agent",
            name="Analysis Agent",
            description="Performs data analysis and insights generation",
            url="http://localhost:8002",
            capabilities=[
                AgentCapability(
                    name="analyze",
                    description="Analyze data and generate insights",
                ),
            ],
        )
    )
    
    analysis_agent.register_capability("analyze", analyze_handler)
    
    # Create a code review agent using decorator syntax
    code_agent = A2AAgent(
        card=AgentCard(
            id="code-agent",
            name="Code Review Agent",
            description="Reviews code for quality and best practices",
            url="http://localhost:8003",
            capabilities=[
                AgentCapability(name="review", description="Review code"),
            ],
        )
    )
    
    @code_agent.capability("review")
    async def review(task):
        return await code_review_handler(task)
    
    print(f"\n✅ Created {research_agent.card.name}")
    print(f"   Capabilities: {[c.name for c in research_agent.card.capabilities]}")
    
    print(f"\n✅ Created {analysis_agent.card.name}")
    print(f"   Capabilities: {[c.name for c in analysis_agent.card.capabilities]}")
    
    print(f"\n✅ Created {code_agent.card.name}")
    print(f"   Capabilities: {[c.name for c in code_agent.card.capabilities]}")
    
    return research_agent, analysis_agent, code_agent


# =============================================================================
# Example: Task Handling
# =============================================================================

async def example_task_handling(agents):
    """Example handling tasks with A2A agents."""
    print("\n" + "=" * 50)
    print("📋 Task Handling Example")
    print("=" * 50)
    
    research_agent, analysis_agent, code_agent = agents
    
    # Create and handle a search task
    print("\n1️⃣ Search Task:")
    search_task = A2ATask(
        capability="search",
        input_data={"query": "Python async best practices"},
    )
    
    result = await research_agent.handle_task(search_task)
    print(f"   State: {result.state.value}")
    print(f"   Results: {len(result.output_data.get('results', []))} items")
    
    # Create and handle a summarization task
    print("\n2️⃣ Summarization Task:")
    summarize_task = A2ATask(
        capability="summarize",
        input_data={
            "text": "This is a long document about artificial intelligence and machine learning. " * 10,
            "max_length": 50,
        },
    )
    
    result = await research_agent.handle_task(summarize_task)
    print(f"   State: {result.state.value}")
    print(f"   Summary: {result.output_data.get('summary', '')[:60]}...")
    
    # Create and handle an analysis task
    print("\n3️⃣ Analysis Task:")
    analysis_task = A2ATask(
        capability="analyze",
        input_data={
            "data": {"values": [1, 2, 3, 4, 5]},
            "type": "statistical",
        },
    )
    
    result = await analysis_agent.handle_task(analysis_task)
    print(f"   State: {result.state.value}")
    print(f"   Confidence: {result.output_data.get('confidence', 0):.0%}")
    
    # Create and handle a code review task
    print("\n4️⃣ Code Review Task:")
    review_task = A2ATask(
        capability="review",
        input_data={
            "code": "def hello():\n    print('Hello, World!')",
            "language": "python",
        },
    )
    
    result = await code_agent.handle_task(review_task)
    print(f"   State: {result.state.value}")
    print(f"   Score: {result.output_data.get('score', 0)}/10")
    print(f"   Approved: {result.output_data.get('approved', False)}")


# =============================================================================
# Example: Agent Discovery
# =============================================================================

async def example_agent_discovery():
    """Example of agent discovery via Agent Cards."""
    print("\n" + "=" * 50)
    print("🔍 Agent Discovery Example")
    print("=" * 50)
    
    # Agent Cards can be published to a registry or discovered via well-known URLs
    print("""
Agent Discovery Methods:

1. Well-Known URL:
   GET https://agent.example.com/.well-known/agent.json
   Returns the Agent Card JSON

2. Registry:
   POST https://registry.example.com/agents
   Body: Agent Card JSON
   
   GET https://registry.example.com/agents?capability=search
   Returns matching agents

3. Direct URL:
   Agent Card includes 'url' field pointing to the agent's endpoint
""")
    
    # Example Agent Card JSON
    card = AgentCard(
        id="example-agent",
        name="Example Agent",
        description="An example agent for demonstration",
        url="https://agent.example.com",
        capabilities=[
            AgentCapability(name="search", description="Search capability"),
            AgentCapability(name="analyze", description="Analysis capability"),
        ],
    )
    
    print("Example Agent Card JSON:")
    import json
    print(json.dumps(card.to_dict(), indent=2))


# =============================================================================
# Example: Task Delegation
# =============================================================================

async def example_task_delegation():
    """Example of delegating tasks between agents."""
    print("\n" + "=" * 50)
    print("🔄 Task Delegation Example")
    print("=" * 50)
    
    print("""
Task Delegation Flow:

1. Orchestrator Agent receives a complex task
2. Breaks it down into subtasks
3. Discovers agents with required capabilities
4. Delegates subtasks to appropriate agents
5. Aggregates results

Example Code:
```python
# Orchestrator agent
orchestrator = A2AAgent(card=AgentCard(...))

# Discover agents
research_agent = await orchestrator.discover_agent(
    registry_url="https://registry.example.com",
    capability="search",
)

# Delegate task
result = await orchestrator.delegate(
    agent_url=research_agent.url,
    capability="search",
    input_data={"query": "AI trends 2024"},
)

# Process result
if result.state == TaskState.COMPLETED:
    search_results = result.output_data["results"]
    # Continue with next step...
```
""")


# =============================================================================
# Example: Running A2A Server
# =============================================================================

async def example_a2a_server():
    """Example of running an A2A server."""
    print("\n" + "=" * 50)
    print("🌐 A2A Server Example")
    print("=" * 50)
    
    print("""
To run an A2A agent as a server:

```python
from squadron.connectivity.a2a import A2AAgent, AgentCard

# Create agent
agent = A2AAgent(
    card=AgentCard(
        id="my-agent",
        name="My Agent",
        description="My A2A agent",
        url="http://localhost:8000",
        capabilities=[...],
    )
)

# Register handlers
@agent.capability("my_capability")
async def handle_my_capability(task):
    return {"result": "done"}

# Start server
await agent.serve(host="0.0.0.0", port=8000)
```

The server exposes:
- GET  /.well-known/agent.json  - Agent Card
- POST /tasks                    - Create task
- GET  /tasks/{id}              - Get task status
- POST /tasks/{id}/cancel       - Cancel task
""")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all A2A examples."""
    print("🚀 Squadron A2A Multi-Agent Examples")
    print("=" * 50)
    
    try:
        # Create agents
        agents = await example_create_agents()
        
        # Handle tasks
        await example_task_handling(agents)
        
        # Discovery
        await example_agent_discovery()
        
        # Delegation
        await example_task_delegation()
        
        # Server
        await example_a2a_server()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ A2A examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
