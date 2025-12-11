---
date created: 2025-12-11
date updated: 2025-12-11 14:22 UTC
---

# Design Decisions

This document explains the key design decisions made in Squadron and the reasoning behind them.

## Why These Technologies?

### LangGraph over LangChain

**Decision**: Use LangGraph for orchestration instead of LangChain's AgentExecutor.

**Reasoning**:
- LangChain's AgentExecutor is linear and doesn't support cycles well
- Agents need loops: think -> act -> observe -> think -> ...
- LangGraph provides native cyclic graph support
- Better state management with immutable transitions
- Built-in checkpointing for long-running tasks

### Graphiti over Vector Stores

**Decision**: Use Graphiti (temporal knowledge graph) instead of simple vector stores.

**Reasoning**:
- Vector stores treat all information as equally valid
- Real-world facts change over time (user changed jobs, prices updated)
- Graphiti tracks when facts were true and automatically invalidates old ones
- Graph structure captures relationships, not just similarity
- Enables queries like "what did we know about X on date Y?"

### LATS over ReAct

**Decision**: Use LATS (Language Agent Tree Search) instead of simple ReAct loops.

**Reasoning**:
- ReAct is linear: one thought, one action, one observation
- Complex tasks benefit from exploring multiple approaches
- LATS uses MCTS to explore a tree of possibilities
- Can backtrack when a path isn't working
- List-wise verification compares multiple solutions before committing

### MCP over Custom Tool Protocols

**Decision**: Adopt MCP (Model Context Protocol) for tool integration.

**Reasoning**:
- MCP is becoming an industry standard (Anthropic, others adopting)
- "USB-C of AI" - any MCP server works with any MCP client
- Growing ecosystem of pre-built MCP servers
- Standardized discovery, authentication, and execution
- Future-proof as more tools adopt MCP

### A2A over Custom Multi-Agent

**Decision**: Implement A2A protocol for agent-to-agent communication.

**Reasoning**:
- Need horizontal orchestration (agents coordinating as peers)
- A2A provides standard discovery via Agent Cards
- Task lifecycle management (requested -> running -> completed)
- Webhook support for async operations
- Enables building agent marketplaces/networks

---

## Architecture Decisions

### Layered Architecture

**Decision**: Organize code into 6 distinct layers (L0-L5).

**Reasoning**:
- Clear separation of concerns
- Each layer can be tested independently
- Users can adopt layers incrementally
- Easier to understand and maintain
- Matches conceptual model of agent capabilities

### Immutable State

**Decision**: Use immutable state objects throughout.

**Reasoning**:
- Predictable state transitions
- Easy to debug (can trace state history)
- Enables checkpointing and replay
- Prevents accidental mutations
- Works well with LangGraph's state model

### Provider Abstraction

**Decision**: Create unified LLM provider interface.

**Reasoning**:
- Users shouldn't be locked to one provider
- Easy to switch between cloud and local models
- Consistent API regardless of backend
- Enables cost optimization (use cheaper models for simple tasks)
- Future-proof as new providers emerge

### Tool Packs

**Decision**: Provide pre-built tool collections.

**Reasoning**:
- Common tasks shouldn't require custom tools
- Batteries-included experience
- Best practices built in (safety checks, error handling)
- Users can extend or replace as needed
- Reduces time to first working agent

---

## Safety Decisions

### Sandboxed Evolution

**Decision**: Run self-improvement mutations in sandboxes.

**Reasoning**:
- Self-modifying code is inherently risky
- Sandbox prevents mutations from affecting host system
- Can test mutations safely before accepting
- Supports both subprocess and Docker isolation
- Enables aggressive exploration without risk

### Guardrails by Default

**Decision**: Enable safety guardrails by default.

**Reasoning**:
- Agents can cause real harm (delete files, run commands)
- Safe defaults prevent accidents
- Users can disable for trusted environments
- Multiple guardrail types (content, rate, tool-specific)
- Approval workflows for dangerous operations

### Evaluation Before Evolution

**Decision**: Require evaluation suite before self-improvement.

**Reasoning**:
- Can't improve without measuring
- Prevents regressions (mutations must pass all tests)
- Provides objective improvement metrics
- Enables automated evolution pipelines
- Documents expected agent behavior

---

## API Decisions

### Async-First

**Decision**: All I/O operations are async.

**Reasoning**:
- LLM calls are slow (seconds to minutes)
- Async enables concurrent operations
- Better resource utilization
- Matches modern Python patterns
- Required for streaming support

### Configuration via Pydantic

**Decision**: Use Pydantic for all configuration.

**Reasoning**:
- Type safety and validation
- Environment variable support via pydantic-settings
- Clear documentation of options
- IDE autocomplete support
- Easy serialization/deserialization

### Factory Pattern for LLMs

**Decision**: Use factory function `create_llm()` for provider instantiation.

**Reasoning**:
- Hides provider-specific complexity
- Auto-detection from environment
- Consistent creation interface
- Easy to add new providers
- Supports presets for common configurations

---

## Trade-offs Acknowledged

### Complexity vs Simplicity

Squadron is more complex than simple agent frameworks. This is intentional:
- Simple frameworks hit walls quickly
- Complex tasks need sophisticated reasoning
- The layered architecture manages complexity
- Users can start simple and add layers as needed

### Performance vs Safety

We prioritize safety over raw performance:
- Guardrails add overhead but prevent disasters
- Sandboxing is slower but essential for evolution
- Evaluation takes time but ensures quality
- These can be disabled when appropriate

### Flexibility vs Opinions

We provide opinions but allow overrides:
- Default configurations work out of the box
- Every default can be changed
- Multiple providers supported
- Tool packs are optional

---

## Future Considerations

### Planned Improvements

1. **Vector Search in Memory**: Hybrid graph + vector for better retrieval
2. **Distributed Execution**: Run agents across multiple machines
3. **Visual Reasoning**: Support for image/video understanding
4. **Production Tools**: Monitoring, logging, deployment utilities

### Extension Points

The architecture is designed for extension:
- Custom LLM providers via `LLMProvider` base class
- Custom tools via `@mcp_tool` decorator
- Custom guardrails via `Guardrail` class
- Custom evaluation metrics via `EvalMetric`
