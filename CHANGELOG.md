---
date created: 2025-12-11
date updated: 2025-12-11 14:20 UTC
---

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-11

### Added

- **Core Agent Framework**
  - `Agent` class with LangGraph-based orchestration
  - Immutable state management with Pydantic
  - Tool registration and execution

- **LLM Provider Abstraction**
  - Unified interface for all LLM providers
  - OpenAI provider (GPT-4, GPT-4o, GPT-3.5)
  - Anthropic provider (Claude 3.5, Claude 3)
  - Ollama provider (local models)
  - Hugging Face provider (Inference API + local transformers)
  - OpenAI-compatible provider (vLLM, TGI, LocalAI, etc.)
  - Factory pattern with auto-detection
  - Streaming support
  - Tool/function calling support

- **Memory System (L1)**
  - Graphiti integration for temporal knowledge graphs
  - Entity, Edge, and Fact types
  - Time-aware fact retrieval
  - In-memory fallback

- **Reasoning Engine (L2)**
  - LATS (Language Agent Tree Search) implementation
  - MCTS (Monte Carlo Tree Search) for planning
  - List-wise verification for plan selection
  - Configurable exploration/exploitation

- **Connectivity Layer (L3)**
  - MCP Host for managing MCP servers
  - MCP Client for remote tool access
  - A2A (Agent-to-Agent) protocol
  - Agent Cards for discovery
  - Task delegation between agents

- **Governance Layer (L4)**
  - Agent evaluation system
  - Multiple evaluation metrics
  - Safety guardrails
  - Content filtering
  - Rate limiting

- **Evolution Layer (L5)**
  - SICA (Self-Improving Coding Agent) engine
  - Sandboxed code execution
  - Mutation types (prompt, config, code)
  - Evolution archive for tracking

- **Pre-built Tool Packs**
  - CodingTools: file ops, grep, git
  - ResearchTools: web search, summarization
  - OpsTools: shell commands, Docker, monitoring

- **Testing & Examples**
  - 115 passing tests
  - 6 working examples
  - Comprehensive documentation

### Security

- Guardrails for dangerous operations
- Sandboxed execution for self-improvement
- Secret redaction in logs

---

[Unreleased]: https://github.com/squadron-ai/squadron/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/squadron-ai/squadron/releases/tag/v0.1.0
