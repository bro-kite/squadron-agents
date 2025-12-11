"""
Squadron Agent Framework

A next-generation AI agent framework built on the principle that intelligent
behavior emerges from sophisticated reasoning at inference time.

Core Philosophy: "Inference-Time Compute over Pre-Training Scale"

Architecture:
- L0: Orchestrator (LangGraph) - Cyclic graph execution & state
- L1: Memory Kernel (Graphiti) - Temporal Knowledge Graph
- L2: Reasoning Engine (LATS) - System 2 thinking with backtracking
- L3: Connectivity (MCP + A2A) - Tool integration & agent swarming
- L4: Guardrails (DeepEval) - Regression testing & safety checks
- L5: Meta-Optimizer (SICA) - Self-improvement via code/prompt rewriting
"""

# Core
from squadron.core.agent import Agent
from squadron.core.config import SquadronConfig
from squadron.core.state import AgentState

# Memory (L1)
from squadron.memory import GraphitiMemory

# Reasoning (L2)
from squadron.reasoning import LATSReasoner, ListWiseVerifier, MCTSController

# Connectivity (L3)
from squadron.connectivity import MCPHost, MCPClient, A2AAgent, AgentCard

# Governance (L4)
from squadron.governance import AgentEvaluator, SafetyGuardrails, TestCase

# Evolution (L5)
from squadron.evolution import SICAEngine, Sandbox, EvolutionArchive

# LLM Providers
from squadron.llm import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    create_llm,
    LLMFactory,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
)

# Tools
from squadron.tools import CodingTools, ResearchTools, OpsTools

__version__ = "0.1.0"
__all__ = [
    # Core
    "Agent",
    "AgentState",
    "SquadronConfig",
    # Memory
    "GraphitiMemory",
    # Reasoning
    "LATSReasoner",
    "ListWiseVerifier",
    "MCTSController",
    # Connectivity
    "MCPHost",
    "MCPClient",
    "A2AAgent",
    "AgentCard",
    # Governance
    "AgentEvaluator",
    "SafetyGuardrails",
    "TestCase",
    # Evolution
    "SICAEngine",
    "Sandbox",
    "EvolutionArchive",
    # LLM Providers
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "create_llm",
    "LLMFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "OpenAICompatibleProvider",
    # Tools
    "CodingTools",
    "ResearchTools",
    "OpsTools",
    # Version
    "__version__",
]