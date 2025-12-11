"""
LLM Provider Abstraction Layer

Provides a unified interface for working with any LLM:
- OpenAI API (GPT-4, GPT-3.5, etc.)
- Anthropic API (Claude)
- Ollama (local models)
- Hugging Face (transformers, Inference API)
- OpenAI-compatible APIs (vLLM, LiteLLM, LocalAI, etc.)
- Custom cloud deployments (DigitalOcean, RunPod, etc.)
"""

from squadron.llm.base import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    MessageRole,
    ToolDefinition,
    ToolCall,
)
from squadron.llm.factory import create_llm, LLMFactory
from squadron.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
)

__all__ = [
    # Base
    "LLMProvider",
    "LLMResponse",
    "LLMMessage",
    "MessageRole",
    "ToolDefinition",
    "ToolCall",
    # Factory
    "create_llm",
    "LLMFactory",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "OpenAICompatibleProvider",
]
