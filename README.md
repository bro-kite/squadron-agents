---
date created: 2025-12-11
date updated: 2025-12-28 16:40 UTC
---

<div align="center">

# 🚀 Squadron Agent Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-192%20passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**"Inference-Time Compute over Pre-Training Scale"**

*A next-generation AI agent framework where intelligent behavior emerges from sophisticated reasoning at inference time, not just from larger pre-trained models.*

[Getting Started](#quick-start) •
[Documentation](docs/) •
[Examples](examples/) •
[Contributing](#contributing)

</div>

---

## Architecture

The framework is built as a layered modular system:

| Layer | Component | Technology | Responsibility |
|-------|-----------|------------|----------------|
| **L5** | Meta-Optimizer | SICA / ADAS | Self-improvement via code/prompt rewriting |
| **L4** | Guardrails | DeepEval / Ragas | Regression testing & safety checks |
| **L3** | Connectivity | MCP + A2A | Tool integration & agent swarming |
| **L2** | Reasoning Engine | LATS (MCTS) | System 2 thinking with backtracking |
| **L1** | Memory Kernel | Graphiti (Zep) | Temporal Knowledge Graph |
| **L0** | Orchestrator | LangGraph | Cyclic graph execution & state |

## Key Features

- **Any LLM Provider**: OpenAI, Anthropic, Ollama, Hugging Face, or any OpenAI-compatible API (vLLM, DigitalOcean, RunPod, etc.)
- **Agent Skills**: Industry-standard modular capabilities following the [Agent Skills](https://github.com/agentskills/agentskills) specification - compatible with Claude, GitHub Copilot, and OpenAI Codex
- **LLM-Powered Reasoning**: Intelligent action generation and task completion detection using LLM evaluation
- **Temporal Memory**: Knowledge graph that tracks facts over time and invalidates outdated information
- **Tree Search Reasoning**: Full MCTS-based planning with LLM-powered expand/simulate functions
- **List-Wise Verification**: Superior plan selection through comparative ranking
- **Resume Capability**: Interrupt and resume agent execution with human-in-the-loop approval
- **MCP Integration**: Universal tool connectivity via Model Context Protocol
- **A2A Protocol**: Multi-agent coordination and task delegation
- **Self-Evolution**: Agents that improve their own code and prompts
- **Pre-built Tool Packs**: Coding, Research, and Ops tools ready to use

## Installation

```bash
# Core installation (minimal dependencies)
pip install squadron-agent

# With all optional features
pip install squadron-agent[all]

# Specific features
pip install squadron-agent[graphiti]   # Neo4j memory
pip install squadron-agent[eval]       # DeepEval integration
pip install squadron-agent[sandbox]    # Docker sandboxing
pip install squadron-agent[research]   # Web research tools

# From source
git clone https://github.com/squadron-agent/squadron
cd squadron
pip install -e ".[dev]"
```

## Quick Start

```python
from squadron import (
    Agent,
    GraphitiMemory,
    LATSReasoner,
    CodingTools,
)

# Initialize components
memory = GraphitiMemory()
reasoner = LATSReasoner(default_tool="read_file")
tools = CodingTools(workspace_root="./my-project")

# Create agent
agent = Agent(
    name="developer",
    memory=memory,
    reasoner=reasoner,
)

# Register tool pack
agent.register_tool_pack(tools)

# Run
result = await agent.run("Analyze the codebase structure")
```

### Using MCP Tools

```python
from squadron import Agent, MCPHost

# Load MCP servers from config
mcp = MCPHost()
await mcp.load_servers("mcp_servers.json")

# Create agent with MCP tools
agent = Agent(name="mcp-agent")
await agent.load_mcp_tools(mcp)

result = await agent.run("Query the database for recent orders")
```

### Multi-Agent with A2A

```python
from squadron import A2AAgent, AgentCard, AgentCapability

# Create an agent that can be discovered
agent = A2AAgent(
    card=AgentCard(
        id="researcher",
        name="Research Agent",
        description="Performs web research",
        capabilities=[
            AgentCapability(name="search", description="Search the web"),
        ],
    )
)

# Register capability handler
@agent.capability("search")
async def handle_search(task):
    # Perform search...
    return {"results": [...]}

# Delegate to another agent
result = await agent.delegate(
    agent_url="https://other-agent.example.com",
    capability="analyze",
    input_data={"text": "..."},
)
```

### Using Agent Skills

Agent Skills are modular capabilities that follow the [Agent Skills](https://github.com/agentskills/agentskills) open standard. Skills work across Claude, GitHub Copilot, OpenAI Codex, and Squadron.

```python
from squadron import Agent, SkillsManager, LATSReasoner

# Initialize skills manager
skills = SkillsManager(workspace_path="./my-project")
await skills.discover_skills()  # Finds skills in .github/skills/, .claude/skills/, etc.

# Create reasoner with skills support
reasoner = LATSReasoner(skills_manager=skills)

# Agent automatically uses relevant skills
agent = Agent(name="developer", reasoner=reasoner)
result = await agent.run("Create a PowerPoint presentation")  # Uses pptx skill if available
```

**Creating a Skill:**

```
.github/skills/my-skill/
├── SKILL.md          # Required: metadata + instructions
├── templates/        # Optional: code templates
└── examples/         # Optional: usage examples
```

```yaml
# SKILL.md
---
name: my-skill
description: What this skill does and when to use it
tags: [python, testing]
---
# My Skill

## Instructions
When performing this task:
1. Step one
2. Step two

## Examples
- Example usage here
```

Skills use **progressive disclosure** - only relevant content loads into context:
- **Level 1**: Name/description always loaded (for discovery)
- **Level 2**: Full instructions loaded when skill is triggered
- **Level 3**: Resources loaded on-demand

### Self-Improvement with SICA

```python
from squadron import Agent, SICAEngine, AgentEvaluator, TestCase

# Create evaluation suite
test_cases = [
    TestCase(
        name="file_read",
        task="Read the README file",
        expected_tools=["read_file"],
    ),
]

# Run self-improvement
sica = SICAEngine(evaluator=AgentEvaluator())
result = await sica.improve(
    agent=agent,
    test_cases=test_cases,
)

if result.accepted:
    print(f"Improvement: {result.improvement:.2%}")
```

### Using Any LLM Provider

Squadron works with any LLM - cloud APIs, local models, or self-hosted endpoints:

```python
from squadron import create_llm, Agent, LLMMessage

# OpenAI (auto-detected from OPENAI_API_KEY)
llm = create_llm(model="gpt-4o")

# Anthropic
llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Ollama (local)
llm = create_llm(provider="ollama", model="llama3.2")

# Hugging Face Inference API
llm = create_llm(
    provider="huggingface",
    model="meta-llama/Llama-3.2-3B-Instruct",
    api_key="hf_..."
)

# Local transformers model (runs on your GPU)
llm = create_llm(
    provider="huggingface",
    model="meta-llama/Llama-3.2-3B-Instruct",
    use_local=True
)

# Any OpenAI-compatible endpoint (vLLM, DigitalOcean, RunPod, Together AI, etc.)
llm = create_llm(
    provider="openai_compatible",
    model="llama-3.2-70b",
    base_url="https://your-gpu-server.com",
    api_key="your-api-key",
)

# Use with agent
agent = Agent(name="my-agent", llm=llm)

# Or use directly
response = await llm.generate([
    LLMMessage.system("You are a helpful assistant."),
    LLMMessage.user("Hello!"),
])
print(response.content)
```

**Supported Providers:**

| Provider | Models | Tool Calling | Streaming |
|----------|--------|--------------|-----------|
| OpenAI | GPT-4o, GPT-4, GPT-3.5 | ✅ | ✅ |
| Anthropic | Claude 3.5, Claude 3 | ✅ | ✅ |
| Ollama | Llama, Mistral, Phi, etc. | ✅ (some models) | ✅ |
| Hugging Face | Any HF model | ❌ | ✅ |
| OpenAI-Compatible | vLLM, TGI, LocalAI, etc. | ✅ | ✅ |

## Project Structure

```
squadron/
├── src/squadron/
│   ├── core/           # L0: Runtime & orchestration
│   │   ├── agent.py    # Main Agent class (with resume support)
│   │   ├── config.py   # Configuration management
│   │   └── state.py    # Immutable state objects
│   ├── memory/         # L1: Temporal knowledge graph
│   │   ├── graphiti.py # Graphiti integration
│   │   └── types.py    # Entity, Edge, Fact types
│   ├── reasoning/      # L2: LATS & MCTS implementation
│   │   ├── lats.py     # Language Agent Tree Search (LLM-powered)
│   │   ├── mcts.py     # Monte Carlo Tree Search
│   │   └── verifier.py # List-wise verification
│   ├── skills/         # Agent Skills (agentskills.io standard)
│   │   ├── manager.py  # SkillsManager with progressive disclosure
│   │   ├── models.py   # Skill, SkillMetadata data classes
│   │   └── parser.py   # SKILL.md parser
│   ├── connectivity/   # L3: MCP & A2A protocols
│   │   ├── mcp_host.py # MCP server management
│   │   ├── mcp_client.py # Remote MCP client
│   │   └── a2a.py      # Agent-to-Agent protocol
│   ├── governance/     # L4: Evaluation & safety
│   │   ├── evaluator.py # Agent evaluation
│   │   └── guardrails.py # Safety guardrails
│   ├── evolution/      # L5: Self-improvement
│   │   ├── sica.py     # Self-Improving Coding Agent
│   │   ├── sandbox.py  # Sandboxed execution
│   │   └── archive.py  # Mutation history
│   ├── llm/            # LLM provider abstraction
│   │   ├── base.py     # Abstract LLMProvider interface
│   │   ├── providers.py # OpenAI, Anthropic, Ollama, HuggingFace, etc.
│   │   └── factory.py  # create_llm() factory
│   └── tools/          # Pre-built tool packs
│       ├── coding.py   # File ops, git, code search
│       ├── research.py # Web search, summarization
│       └── ops.py      # Shell, Docker, monitoring
├── tests/
├── examples/
└── docs/
```

## Configuration

Create a `.env` file:

```env
# LLM Provider Configuration
LLM_PROVIDER=openai                    # openai, anthropic, ollama, huggingface, openai_compatible
LLM_MODEL=gpt-4o                       # Model name
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# API Keys (set the one for your provider)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...                        # Hugging Face token

# Custom Endpoint (for OpenAI-compatible APIs like vLLM, DigitalOcean GPU, etc.)
LLM_BASE_URL=https://your-gpu-server.com/v1

# Ollama (local models)
LLM_OLLAMA_BASE_URL=http://localhost:11434

# Hugging Face local models
LLM_USE_LOCAL_MODEL=false              # Set to true to run model locally

# Memory (optional - falls back to in-memory)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Search (for research tools)
SERPER_API_KEY=your-key

# Reasoning
REASONING_N_CANDIDATES=5
REASONING_MAX_DEPTH=10
REASONING_USE_LLM_GENERATION=true
REASONING_USE_LLM_COMPLETION=true
REASONING_COMPLETION_CONFIDENCE=0.8

# Agent Skills
SKILLS_ENABLED=true
SKILLS_MAX_IN_CONTEXT=5
SKILLS_RELEVANCE_THRESHOLD=0.2
SKILLS_AUTO_DISCOVER=true

# Governance
GOVERNANCE_ENABLE_GUARDRAILS=true
GOVERNANCE_MAX_ITERATIONS=50

# Evolution
EVOLUTION_ENABLE_SELF_IMPROVEMENT=false
```

## Examples

Check out the [examples/](examples/) directory for working code:

| Example | Description |
|---------|-------------|
| [basic_agent.py](examples/basic_agent.py) | Simple agent with custom tools |
| [quick_multiagent.py](examples/quick_multiagent.py) | **Fastest way to multi-agent** |
| [llm_providers.py](examples/llm_providers.py) | Using different LLM providers |
| [mcp_tools.py](examples/mcp_tools.py) | MCP tool integration |
| [multi_agent_a2a.py](examples/multi_agent_a2a.py) | Multi-agent with A2A protocol |
| [tool_packs.py](examples/tool_packs.py) | Pre-built tool packs |
| [self_improvement.py](examples/self_improvement.py) | SICA self-improvement |

> 📖 **New to multi-agent?** Start with [Quick Start: Multi-Agent](docs/QUICKSTART_MULTIAGENT.md)

## Development

```bash
# Clone the repository
git clone https://github.com/squadron-ai/squadron.git
cd squadron

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=squadron

# Type checking
mypy src/squadron

# Linting
ruff check src/squadron

# Format code
ruff format src/squadron
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Type hints are required for all public APIs
- Tests are required for new features

## Roadmap

- [x] **v0.1.1**: LLM-powered reasoning (action generation, completion detection)
- [x] **v0.1.2**: Agent Skills support (agentskills.io standard)
- [ ] **v0.2**: Enhanced memory with vector search
- [ ] **v0.3**: Distributed agent execution
- [ ] **v0.4**: Visual reasoning capabilities
- [ ] **v0.5**: Production deployment tools

## Acknowledgments

Squadron builds on the shoulders of giants:

- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based agent orchestration
- [Graphiti](https://github.com/getzep/graphiti) - Temporal knowledge graphs
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol
- [Agent Skills](https://github.com/agentskills/agentskills) - Open standard for agent capabilities


## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[⬆ Back to Top](#-squadron-agent-framework)**


</div>
