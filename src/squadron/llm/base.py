"""
Base LLM Provider Interface

Defines the abstract interface that all LLM providers must implement.
This allows Squadron to work with any LLM regardless of the underlying API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """
    A message in the conversation.
    
    Unified format that works across all providers.
    """
    role: MessageRole
    content: str
    
    # For tool messages
    name: str | None = None
    tool_call_id: str | None = None
    
    # For assistant messages with tool calls
    tool_calls: list[ToolCall] | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message format."""
        msg: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        
        if self.name:
            msg["name"] = self.name
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]
        
        return msg
    
    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic message format."""
        if self.role == MessageRole.SYSTEM:
            # Anthropic handles system messages separately
            return {"role": "user", "content": f"[System]: {self.content}"}
        
        msg: dict[str, Any] = {
            "role": self.role.value if self.role != MessageRole.TOOL else "user",
            "content": self.content,
        }
        
        if self.role == MessageRole.TOOL:
            msg["content"] = f"[Tool Result - {self.name}]: {self.content}"
        
        return msg
    
    @classmethod
    def system(cls, content: str) -> LLMMessage:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> LLMMessage:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> LLMMessage:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool(cls, content: str, name: str, tool_call_id: str) -> LLMMessage:
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )


@dataclass
class ToolCall:
    """
    A tool call requested by the LLM.
    """
    id: str
    name: str
    arguments: dict[str, Any]
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool call format."""
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }
    
    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> ToolCall:
        """Create from OpenAI tool call format."""
        import json
        func = data.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        
        return cls(
            id=data.get("id", str(uuid4())),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class ToolDefinition:
    """
    Definition of a tool that can be called by the LLM.
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
    
    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class LLMResponse:
    """
    Response from an LLM.
    
    Unified format across all providers.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Content
    content: str = ""
    
    # Tool calls (if any)
    tool_calls: list[ToolCall] = field(default_factory=list)
    
    # Finish reason
    finish_reason: str = "stop"  # stop, tool_calls, length, content_filter
    
    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Model info
    model: str = ""
    provider: str = ""
    
    # Timing
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Raw response (for debugging)
    raw_response: Any = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0
    
    def to_message(self) -> LLMMessage:
        """Convert to an LLMMessage for conversation history."""
        return LLMMessage.assistant(
            content=self.content,
            tool_calls=self.tool_calls if self.tool_calls else None,
        )


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement this interface to work with Squadron.
    
    Example implementation:
        ```python
        class MyProvider(LLMProvider):
            async def generate(self, messages, **kwargs):
                # Call your LLM API
                response = await my_api.chat(messages)
                return LLMResponse(content=response.text)
        ```
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """
        Initialize the provider.
        
        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific options
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.options = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...
    
    @property
    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling."""
        return True
    
    @property
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True
    
    @property
    def supports_vision(self) -> bool:
        """Check if provider supports vision/images."""
        return False
    
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Conversation history
            tools: Available tools
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            **kwargs: Additional provider-specific options
            
        Returns:
            LLM response
        """
        ...
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.
        
        Args:
            messages: Conversation history
            tools: Available tools
            **kwargs: Additional options
            
        Yields:
            Response chunks
        """
        # Default implementation: fall back to non-streaming
        response = await self.generate(messages, tools, **kwargs)
        yield response.content
    
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        # Default: rough estimate (4 chars per token)
        return len(text) // 4
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError(f"{self.provider_name} does not support embeddings")
    
    def format_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """
        Format messages for this provider.
        
        Override in subclasses for provider-specific formatting.
        """
        return [msg.to_openai_format() for msg in messages]
    
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """
        Format tools for this provider.
        
        Override in subclasses for provider-specific formatting.
        """
        return [tool.to_openai_format() for tool in tools]
    
    async def close(self) -> None:
        """Close any open connections."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
