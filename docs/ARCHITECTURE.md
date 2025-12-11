---
date created: 2025-12-11
date updated: 2025-12-11 14:22 UTC
---

# Squadron Architecture

This document provides a deep dive into the Squadron Agent Framework architecture.

## Table of Contents

- [Philosophy](#philosophy)
- [The Layered Architecture](#the-layered-architecture)
- [Layer 0: Orchestrator](#layer-0-orchestrator)
- [Layer 1: Memory Kernel](#layer-1-memory-kernel)
- [Layer 2: Reasoning Engine](#layer-2-reasoning-engine)
- [Layer 3: Connectivity](#layer-3-connectivity)
- [Layer 4: Governance](#layer-4-governance)
- [Layer 5: Evolution](#layer-5-evolution)
- [LLM Provider Abstraction](#llm-provider-abstraction)
- [Tool Packs](#tool-packs)

---

## Philosophy

### Inference-Time Compute over Pre-Training Scale

The core philosophy of Squadron is that **intelligent behavior emerges from sophisticated reasoning at inference time**, not just from larger pre-trained models.

Traditional approaches focus on training larger models. Squadron takes a different approach:

- **Tree search reasoning** to explore multiple solution paths
- **Temporal memory** to maintain context across sessions
- **Self-verification** to catch and correct mistakes
- **Self-improvement** to evolve agent capabilities over time

### Why a Layered Architecture?

1. **Separation of Concerns**: Each layer handles one aspect of agent behavior
2. **Modularity**: Layers can be swapped or upgraded independently
3. **Testability**: Each layer can be tested in isolation
4. **Flexibility**: Users can use only the layers they need

---

## The Layered Architecture

```
+-------------------------------------------------------------+
|  L5: Evolution (SICA)                                       |
|  Self-improvement via code/prompt mutation                  |
+-------------------------------------------------------------+
|  L4: Governance (DeepEval)                                  |
|  Evaluation, safety guardrails, regression testing          |
+-------------------------------------------------------------+
|  L3: Connectivity (MCP + A2A)                               |
|  Tool integration, multi-agent coordination                 |
+-------------------------------------------------------------+
|  L2: Reasoning (LATS/MCTS)                                  |
|  Tree search planning, verification, backtracking           |
+-------------------------------------------------------------+
|  L1: Memory (Graphiti)                                      |
|  Temporal knowledge graph, fact retrieval                   |
+-------------------------------------------------------------+
|  L0: Orchestrator (LangGraph)                               |
|  State management, execution flow, tool dispatch            |
+-------------------------------------------------------------+
```

---

## Layer 0: Orchestrator

**Location**: `src/squadron/core/`

### What It Does

The orchestrator manages agent state, execution flow, and tool dispatch.

### Why LangGraph?

1. **Cyclic Graphs**: Agents need loops (think -> act -> observe -> think)
2. **State Management**: Built-in immutable state transitions
3. **Checkpointing**: Save and restore agent state
4. **Human-in-the-Loop**: Native approval workflows

### Key Files

- `agent.py` - Main Agent class
- `state.py` - Immutable state objects (AgentState, Message, ToolCall)
- `config.py` - Configuration management

---

## Layer 1: Memory Kernel

**Location**: `src/squadron/memory/`

### What It Does

Provides **temporal knowledge graphs** - facts that change over time.

Unlike vector stores, the memory kernel:
- Tracks **when** facts were true
- Automatically **invalidates** outdated information
- Maintains **relationships** between entities

### Why Graphiti?

1. **Temporal Awareness**: Facts have valid_from/valid_to timestamps
2. **Graph Structure**: Entities and relationships, not just embeddings
3. **Incremental Updates**: Add facts without rebuilding
4. **Hybrid Search**: Graph traversal + vector similarity

### Key Files

- `graphiti.py` - Graphiti integration
- `types.py` - Entity, Edge, Fact types

---

## Layer 2: Reasoning Engine

**Location**: `src/squadron/reasoning/`

### What It Does

Implements **System 2 thinking** - deliberate reasoning with backtracking.

### Why LATS?

Traditional agents use linear reasoning. LATS uses tree search:

```
Linear:  Think -> Act -> Observe -> Think -> ...

Tree:           [Root]
               /      \
          [Plan A]  [Plan B]
          /    \        \
       [A1]   [A2]     [B1]
```

Benefits:
- **Exploration**: Try multiple approaches
- **Backtracking**: Abandon failing paths
- **Verification**: Compare solutions before committing

### Key Files

- `lats.py` - Language Agent Tree Search
- `mcts.py` - Monte Carlo Tree Search core
- `verifier.py` - List-wise verification

---

## Layer 3: Connectivity

**Location**: `src/squadron/connectivity/`

### What It Does

Connects agents to tools (MCP) and other agents (A2A).

### MCP (Model Context Protocol)

MCP is the "USB-C of AI" - a standard protocol for tool integration.

```python
# Any MCP server works instantly
host = MCPHost()
await host.load_servers("mcp_servers.json")
tools = host.get_all_tools()
```

### A2A (Agent-to-Agent)

Enables multi-agent coordination:

```python
# Discover and delegate to other agents
card = await agent.discover("https://other-agent.com")
result = await agent.delegate(
    agent_id=card.id,
    capability="analyze",
    input_data={"text": "..."}
)
```

### Key Files

- `mcp_host.py` - MCP server management
- `mcp_client.py` - Remote MCP client
- `a2a.py` - Agent-to-Agent protocol

---

## Layer 4: Governance

**Location**: `src/squadron/governance/`

### What It Does

Ensures agents behave correctly and safely:

1. **Evaluation**: Measure agent performance
2. **Guardrails**: Prevent dangerous actions
3. **Testing**: Regression tests for self-improvement

### Why DeepEval?

DeepEval provides:
- Multiple evaluation metrics
- Agentic evaluation (multi-turn)
- Custom metrics via GEval

### Key Files

- `evaluator.py` - Agent evaluation system
- `guardrails.py` - Safety guardrails

---

## Layer 5: Evolution

**Location**: `src/squadron/evolution/`

### What It Does

Enables agents to **improve themselves** through:

1. **Mutation**: Generate variations of prompts/code
2. **Evaluation**: Test mutations against benchmarks
3. **Selection**: Keep improvements, discard regressions

### Why SICA?

SICA (Self-Improving Coding Agent) provides:
- Sandboxed execution for safety
- Multiple mutation types
- Evolution archive for tracking

### Key Files

- `sica.py` - Self-Improving Coding Agent
- `sandbox.py` - Sandboxed execution
- `archive.py` - Mutation history

---

## LLM Provider Abstraction

**Location**: `src/squadron/llm/`

### What It Does

Unified interface for any LLM provider.

### Supported Providers

| Provider | Use Case |
|----------|----------|
| OpenAI | Cloud API (GPT-4, GPT-4o) |
| Anthropic | Cloud API (Claude 3.5) |
| Ollama | Local models |
| HuggingFace | HF Inference API or local |
| OpenAI-Compatible | vLLM, TGI, LocalAI, etc. |

### Key Files

- `base.py` - Abstract LLMProvider interface
- `providers.py` - All provider implementations
- `factory.py` - create_llm() factory

---

## Tool Packs

**Location**: `src/squadron/tools/`

### What They Do

Pre-built tool collections for common tasks.

### Available Packs

| Pack | Tools |
|------|-------|
| CodingTools | read_file, write_file, grep, find, git |
| ResearchTools | web_search, read_url, summarize |
| OpsTools | run_command, docker, system_info |

### Key Files

- `coding.py` - Software development tools
- `research.py` - Web research tools
- `ops.py` - System operations tools
