---
date created: 2025-12-11
date updated: 2025-12-11 14:22 UTC
---

# Squadron Documentation

Welcome to the Squadron Agent Framework documentation.

## Documentation Index

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep dive into the layered architecture and each component |
| [CONCEPTS.md](CONCEPTS.md) | Core concepts explained with code examples |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | Why we made the choices we did |
| [QUICKSTART_MULTIAGENT.md](QUICKSTART_MULTIAGENT.md) | Get multi-agent running in 5 minutes |
| [SETUP_NEO4J.md](SETUP_NEO4J.md) | Setting up Neo4j for persistent memory |

## Quick Links

- [Main README](../README.md) - Getting started guide
- [Examples](../examples/) - Working code examples
- [Contributing](../CONTRIBUTING.md) - How to contribute
- [Changelog](../CHANGELOG.md) - Version history

## Learning Path

### 1. Getting Started
Start with the [main README](../README.md) for installation and quick start.

### 2. Understand the Concepts
Read [CONCEPTS.md](CONCEPTS.md) to understand:
- Agents and their lifecycle
- State management
- Memory and temporal facts
- Reasoning with tree search
- Tools and tool packs
- Multi-agent coordination
- Governance and safety
- Self-improvement

### 3. Explore the Architecture
Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand:
- The 6-layer architecture (L0-L5)
- What each layer does
- Key files in each layer
- How layers interact

### 4. Understand Design Choices
Read [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) to understand:
- Why we chose LangGraph, Graphiti, LATS, MCP, A2A
- Architecture trade-offs
- Safety considerations
- Future plans

### 5. Try the Examples
Work through the [examples](../examples/):
1. `basic_agent.py` - Your first agent
2. `llm_providers.py` - Using different LLMs
3. `mcp_tools.py` - Tool integration
4. `multi_agent_a2a.py` - Multi-agent systems
5. `tool_packs.py` - Pre-built tools
6. `self_improvement.py` - Agent evolution

## API Reference

API documentation is generated from docstrings. Key modules:

### Core
- `squadron.Agent` - Main agent class
- `squadron.SquadronConfig` - Configuration
- `squadron.AgentState` - State management

### LLM
- `squadron.create_llm()` - LLM factory
- `squadron.LLMMessage` - Message types
- `squadron.LLMResponse` - Response types

### Memory
- `squadron.GraphitiMemory` - Temporal knowledge graph
- `squadron.Fact` - Fact representation

### Reasoning
- `squadron.LATSReasoner` - Tree search reasoning
- `squadron.MCTSNode` - Search tree nodes

### Connectivity
- `squadron.MCPHost` - MCP server management
- `squadron.A2AAgent` - Agent-to-agent protocol
- `squadron.AgentCard` - Agent discovery

### Governance
- `squadron.AgentEvaluator` - Evaluation system
- `squadron.SafetyGuardrails` - Safety checks
- `squadron.TestCase` - Test definitions

### Evolution
- `squadron.SICAEngine` - Self-improvement
- `squadron.Sandbox` - Safe execution
- `squadron.EvolutionArchive` - Mutation history

### Tools
- `squadron.CodingTools` - Development tools
- `squadron.ResearchTools` - Research tools
- `squadron.OpsTools` - Operations tools
