---
date created: 2025-12-11
date updated: 2025-12-11 13:12 UTC
---

# Squadron Examples

This directory contains working examples demonstrating various features of the Squadron Agent Framework.

## Examples

### 1. Basic Agent (`basic_agent.py`)

A simple introduction to creating and running a Squadron agent with custom tools.

```bash
python examples/basic_agent.py
```

**Demonstrates:**
- Creating an agent with memory and reasoning
- Registering custom tools
- Running tasks and viewing results

---

### 2. LLM Providers (`llm_providers.py`)

Shows how to use Squadron with different LLM providers.

```bash
python examples/llm_providers.py
```

**Demonstrates:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models)
- Hugging Face (Inference API, local transformers)
- OpenAI-compatible endpoints (vLLM, DigitalOcean, RunPod)
- Streaming responses
- Tool/function calling
- Factory presets

**Required env vars:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN` (depending on provider)

---

### 3. MCP Tools (`mcp_tools.py`)

Demonstrates the Model Context Protocol (MCP) for tool integration.

```bash
python examples/mcp_tools.py
```

**Demonstrates:**
- Creating local MCP tools with `@mcp_tool` decorator
- Loading tools from MCP server configurations
- Using MCP tools with agents
- Tool approval workflows

---

### 4. Multi-Agent A2A (`multi_agent_a2a.py`)

Shows the Agent-to-Agent (A2A) protocol for multi-agent coordination.

```bash
python examples/multi_agent_a2a.py
```

**Demonstrates:**
- Creating agents with capabilities
- Agent Cards for discovery
- Task handling and state management
- Task delegation between agents
- Running A2A servers

---

### 5. Tool Packs (`tool_packs.py`)

Demonstrates the pre-built tool packs for common domains.

```bash
python examples/tool_packs.py
```

**Demonstrates:**
- **CodingTools**: File operations, code search, git
- **ResearchTools**: Web search, URL reading, summarization
- **OpsTools**: Shell commands, Docker, system monitoring
- Creating custom tool packs

---

### 6. Self-Improvement (`self_improvement.py`)

Shows the SICA (Self-Improving Coding Agent) system.

```bash
python examples/self_improvement.py
```

**Demonstrates:**
- Creating test cases for evaluation
- Agent evaluation metrics
- Sandbox execution for safe testing
- Evolution archive for tracking mutations
- The full improvement cycle

---

## Quick Start

1. **Install Squadron:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run an example:**
   ```bash
   python examples/basic_agent.py
   ```

## Environment Variables

Create a `.env` file with the following (as needed):

```env
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...

# For OpenAI-compatible endpoints
LLM_BASE_URL=https://your-server.com

# For research tools
SERPER_API_KEY=...

# For memory (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Creating Your Own Examples

Use these examples as templates for your own agents:

```python
import asyncio
from squadron import Agent, create_llm, CodingTools

async def main():
    # Create LLM
    llm = create_llm(model="gpt-4o")
    
    # Create agent
    agent = Agent(name="my-agent", llm=llm)
    
    # Add tools
    agent.register_tool_pack(CodingTools())
    
    # Run
    result = await agent.run("Your task here")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```
