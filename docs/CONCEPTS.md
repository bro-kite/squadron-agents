---
date created: 2025-12-11
date updated: 2025-12-11 14:22 UTC
---

# Core Concepts

This document explains the key concepts in Squadron to help you understand how the framework works.

## Agents

An **Agent** is the central entity in Squadron. It combines:
- An LLM for reasoning
- Memory for context
- Tools for taking actions
- A reasoning strategy for planning

```python
agent = Agent(
    name="developer",
    llm=create_llm(model="gpt-4o"),
    memory=GraphitiMemory(),
    reasoner=LATSReasoner(),
)
```

### Agent Lifecycle

1. **Initialization**: Agent is created with config, memory, and reasoner
2. **Task Assignment**: User provides a task via `agent.run(task)`
3. **Reasoning Loop**: Agent thinks, plans, and acts iteratively
4. **Completion**: Agent returns final state with results

---

## State Management

Squadron uses **immutable state** to track agent progress.

### AgentState

The `AgentState` contains everything about the current agent run:

```python
@dataclass
class AgentState:
    phase: AgentPhase        # current, thinking, acting, completed, error
    messages: tuple          # conversation history
    tool_results: tuple      # results from tool calls
    thought_tree: dict       # MCTS tree for reasoning
    iteration: int           # current iteration count
    errors: list             # any errors encountered
```

### Why Immutable?

- **Predictable**: State changes are explicit
- **Debuggable**: Can trace exactly how state evolved
- **Checkpointable**: Easy to save and restore

---

## Memory

Memory in Squadron is not just storage - it's a **temporal knowledge graph**.

### Facts

Facts are the basic unit of memory:

```python
fact = Fact(
    subject="user",
    predicate="prefers",
    object="Python",
    valid_from=datetime(2024, 1, 1),
    valid_to=None,  # Still valid
    source="conversation"
)
```

### Temporal Queries

You can query memory at different points in time:

```python
# Current knowledge
current = await memory.search("user preferences")

# Historical knowledge
past = await memory.search("user preferences", time="2023-06-01")
```

### Automatic Invalidation

When new facts contradict old ones, the old facts are automatically invalidated:

```python
# Old fact: user works at Acme
# New fact: user works at NewCo
# Result: Old fact marked as valid_to=now
```

---

## Reasoning

Squadron implements **tree search reasoning** for complex problem solving.

### LATS (Language Agent Tree Search)

Instead of linear reasoning, LATS explores multiple paths:

```
        [Task: Fix the bug]
              |
    +---------+---------+
    |                   |
[Check logs]      [Read code]
    |                   |
[Found error]     [Found issue]
    |                   |
[Fix attempt 1]   [Fix attempt 2]
```

### MCTS (Monte Carlo Tree Search)

MCTS balances exploration and exploitation:

- **Selection**: Choose promising nodes to explore
- **Expansion**: Generate new possible actions
- **Simulation**: Estimate outcome of actions
- **Backpropagation**: Update node values based on results

### Verification

Before committing to a solution, Squadron can verify:

```python
# Generate multiple solutions
solutions = await reasoner.generate_candidates(task, n=5)

# Rank them comparatively
ranked = await verifier.rank_solutions(solutions)

# Use the best one
best = ranked[0]
```

---

## Tools

Tools are functions that agents can call to interact with the world.

### Defining Tools

```python
from squadron import mcp_tool

@mcp_tool(description="Read a file from disk")
async def read_file(path: str) -> str:
    """Read and return file contents."""
    with open(path) as f:
        return f.read()
```

### Tool Packs

Pre-built collections of related tools:

```python
from squadron import CodingTools, ResearchTools, OpsTools

# Use a tool pack
tools = CodingTools(workspace_root="./project")
agent.register_tool_pack(tools)
```

### MCP Tools

External tools via Model Context Protocol:

```python
# Load from MCP server config
host = MCPHost()
await host.load_servers("mcp_servers.json")

# All tools from all servers
tools = host.get_all_tools()
```

---

## Multi-Agent Coordination

Squadron supports multiple agents working together via A2A.

### Agent Cards

Agents advertise capabilities via Agent Cards:

```python
card = AgentCard(
    id="researcher",
    name="Research Agent",
    description="Performs web research",
    capabilities=[
        AgentCapability(name="search", description="Search the web"),
        AgentCapability(name="summarize", description="Summarize content"),
    ],
    base_url="https://researcher.example.com"
)
```

### Discovery

Agents can discover each other:

```python
# Discover an agent by URL
card = await agent.discover("https://other-agent.com")

# Card contains capabilities, endpoints, auth info
print(card.capabilities)
```

### Delegation

Agents can delegate tasks to each other:

```python
result = await agent.delegate(
    agent_url="https://researcher.example.com",
    capability="search",
    input_data={"query": "latest AI research"}
)
```

---

## Governance

Governance ensures agents behave correctly and safely.

### Evaluation

Measure agent performance:

```python
evaluator = AgentEvaluator()

# Evaluate task completion
result = await evaluator.evaluate(
    agent_state=final_state,
    test_case=TestCase(
        task="Read the README",
        expected_tools=["read_file"],
    )
)

print(f"Score: {result.score}")
```

### Guardrails

Prevent dangerous actions:

```python
guardrails = SafetyGuardrails()

# Add custom guardrail
guardrails.add_guardrail(Guardrail(
    name="no_delete",
    description="Prevent file deletion",
    blocked_patterns=[r"rm\s+-rf"],
    action=GuardrailAction.BLOCK,
))

# Check before executing
result = await guardrails.check_tool_call(tool_call)
if not result.passed:
    raise SecurityError(result.reason)
```

---

## Self-Improvement

Squadron agents can improve themselves over time.

### The SICA Loop

1. **Mutate**: Generate variations of prompts/code
2. **Evaluate**: Test mutations against benchmarks
3. **Select**: Keep improvements, discard regressions

```python
sica = SICAEngine(evaluator=AgentEvaluator())

result = await sica.improve(
    agent=agent,
    test_cases=test_suite,
    mutation_types=[MutationType.PROMPT, MutationType.CONFIG],
)

if result.accepted:
    print(f"Improved by {result.improvement:.2%}")
```

### Sandboxing

Mutations run in isolated environments:

```python
sandbox = Sandbox(config=SandboxConfig(
    type="docker",
    timeout=30,
    memory_limit="512m",
))

result = await sandbox.execute(mutated_code)
```

### Evolution Archive

Track mutation history:

```python
archive = EvolutionArchive()

# Record successful mutation
archive.record(ArchiveEntry(
    mutation=mutation,
    before_score=0.75,
    after_score=0.82,
    accepted=True,
))

# Query history
improvements = archive.get_accepted()
```

---

## LLM Providers

Squadron abstracts LLM providers behind a unified interface.

### Creating LLMs

```python
from squadron import create_llm

# Auto-detect from environment
llm = create_llm()

# Specific provider
llm = create_llm(provider="anthropic", model="claude-4-5-sonnet")

# Local model
llm = create_llm(provider="ollama", model="llama3.2")
```

### Using LLMs

```python
from squadron import LLMMessage

response = await llm.generate([
    LLMMessage.system("You are a helpful assistant."),
    LLMMessage.user("Hello!"),
])

print(response.content)
```

### Streaming

```python
async for chunk in llm.stream([
    LLMMessage.user("Write a story...")
]):
    print(chunk.content, end="")
```

### Tool Calling

```python
response = await llm.generate(
    messages=[LLMMessage.user("What's the weather?")],
    tools=[weather_tool],
)

if response.tool_calls:
    for call in response.tool_calls:
        result = await execute_tool(call)
```
