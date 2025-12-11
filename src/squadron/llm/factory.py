"""
LLM Factory

Provides a unified factory for creating LLM providers from configuration.
Supports automatic provider detection and easy switching between models.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from squadron.llm.base import LLMProvider

logger = structlog.get_logger(__name__)


# Provider registry
_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """Register a provider class."""
    _PROVIDERS[name] = provider_class


def get_provider_class(name: str) -> type[LLMProvider] | None:
    """Get a provider class by name."""
    return _PROVIDERS.get(name)


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> LLMProvider:
    """
    Create an LLM provider from configuration.
    
    This is the primary entry point for creating LLM instances.
    It automatically detects the provider from environment variables
    or explicit configuration.
    
    Args:
        provider: Provider name (openai, anthropic, ollama, huggingface, openai_compatible)
        model: Model name/identifier
        api_key: API key (or uses environment variable)
        base_url: Base URL for API (for custom deployments)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional provider-specific options
        
    Returns:
        Configured LLM provider
        
    Examples:
        ```python
        # OpenAI (auto-detected from OPENAI_API_KEY)
        llm = create_llm(model="gpt-4o")
        
        # Anthropic
        llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")
        
        # Ollama (local)
        llm = create_llm(provider="ollama", model="llama3.2")
        
        # Hugging Face Inference API
        llm = create_llm(provider="huggingface", model="meta-llama/Llama-3.2-3B-Instruct")
        
        # Custom OpenAI-compatible endpoint (vLLM, DigitalOcean, etc.)
        llm = create_llm(
            provider="openai_compatible",
            model="llama-3.2-70b",
            base_url="https://your-gpu-server.com",
            api_key="your-key",
        )
        ```
    """
    # Import providers lazily to avoid circular imports
    from squadron.llm.providers import (
        OpenAIProvider,
        AnthropicProvider,
        OllamaProvider,
        HuggingFaceProvider,
        OpenAICompatibleProvider,
    )
    
    # Register providers if not already done
    if not _PROVIDERS:
        register_provider("openai", OpenAIProvider)
        register_provider("anthropic", AnthropicProvider)
        register_provider("ollama", OllamaProvider)
        register_provider("huggingface", HuggingFaceProvider)
        register_provider("openai_compatible", OpenAICompatibleProvider)
    
    # Auto-detect provider if not specified
    if provider is None:
        provider = _detect_provider(api_key, base_url, model)
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = _get_api_key_for_provider(provider)
    
    # Set default model if not specified
    if model is None:
        model = _get_default_model(provider)
    
    # Create the provider
    provider_class = _PROVIDERS.get(provider)
    if provider_class is None:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )
    
    logger.info("Creating LLM provider", provider=provider, model=model)
    
    # Build kwargs based on provider
    provider_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    
    if api_key:
        provider_kwargs["api_key"] = api_key
    
    if base_url:
        provider_kwargs["base_url"] = base_url
    
    return provider_class(**provider_kwargs)


def _detect_provider(
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> str:
    """Auto-detect the provider based on available configuration."""
    
    # If base_url is provided, assume OpenAI-compatible
    if base_url:
        return "openai_compatible"
    
    # Check for API keys in environment
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"):
        return "huggingface"
    
    # Check model name patterns
    if model:
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "llama" in model_lower or "mistral" in model_lower or "phi" in model_lower:
            # Could be Ollama or HuggingFace
            # Check if Ollama is running
            if _is_ollama_available():
                return "ollama"
            return "huggingface"
    
    # Default to OpenAI
    return "openai"


def _get_api_key_for_provider(provider: str) -> str | None:
    """Get the API key for a provider from environment."""
    env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        "openai_compatible": ["OPENAI_API_KEY", "API_KEY"],
    }
    
    for var in env_vars.get(provider, []):
        value = os.getenv(var)
        if value:
            return value
    
    return None


def _get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.2",
        "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
        "openai_compatible": "default",
    }
    return defaults.get(provider, "default")


def _is_ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


class LLMFactory:
    """
    Factory class for creating and managing LLM providers.
    
    Provides a more object-oriented interface for LLM creation
    with support for caching and configuration presets.
    
    Example:
        ```python
        factory = LLMFactory()
        
        # Register a preset
        factory.register_preset("fast", provider="openai", model="gpt-4o-mini")
        factory.register_preset("smart", provider="anthropic", model="claude-3-5-sonnet-20241022")
        
        # Create from preset
        llm = factory.create("fast")
        
        # Or create directly
        llm = factory.create(provider="ollama", model="llama3.2")
        ```
    """
    
    def __init__(self):
        self._presets: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, LLMProvider] = {}
    
    def register_preset(
        self,
        name: str,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> None:
        """
        Register a configuration preset.
        
        Args:
            name: Preset name
            provider: Provider name
            model: Model name
            **kwargs: Additional configuration
        """
        self._presets[name] = {
            "provider": provider,
            "model": model,
            **kwargs,
        }
    
    def create(
        self,
        preset: str | None = None,
        cache: bool = False,
        **kwargs: Any,
    ) -> LLMProvider:
        """
        Create an LLM provider.
        
        Args:
            preset: Preset name to use
            cache: Whether to cache the provider
            **kwargs: Override preset or provide direct configuration
            
        Returns:
            LLM provider instance
        """
        # Merge preset with kwargs
        config = {}
        if preset and preset in self._presets:
            config.update(self._presets[preset])
        config.update(kwargs)
        
        # Check cache
        cache_key = str(sorted(config.items()))
        if cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create provider
        provider = create_llm(**config)
        
        # Cache if requested
        if cache:
            self._cache[cache_key] = provider
        
        return provider
    
    def get_preset(self, name: str) -> dict[str, Any] | None:
        """Get a preset configuration."""
        return self._presets.get(name)
    
    def list_presets(self) -> list[str]:
        """List available presets."""
        return list(self._presets.keys())
    
    async def close_all(self) -> None:
        """Close all cached providers."""
        for provider in self._cache.values():
            await provider.close()
        self._cache.clear()
