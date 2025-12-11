"""
Pytest configuration and shared fixtures for Squadron tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from squadron.core.config import SquadronConfig
from squadron.core.state import AgentState, Message, MessageRole
from squadron.llm.base import LLMProvider, LLMResponse, LLMMessage, ToolCall


@pytest.fixture
def config():
    """Default test configuration."""
    cfg = SquadronConfig()
    cfg.governance.max_iterations = 5
    return cfg


@pytest.fixture
def mock_llm():
    """Mock LLM provider for testing."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"
    llm.supports_tools = True
    llm.supports_streaming = True
    
    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content="Mock response",
            model="mock-model",
            provider="mock",
        )
    
    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()
    
    return llm


@pytest.fixture
def mock_llm_with_tool_call():
    """Mock LLM that returns a tool call."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"
    llm.supports_tools = True
    
    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="test_tool",
                    arguments={"arg": "value"},
                )
            ],
            finish_reason="tool_calls",
            model="mock-model",
            provider="mock",
        )
    
    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()
    
    return llm


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        LLMMessage.system("You are a helpful assistant."),
        LLMMessage.user("Hello!"),
        LLMMessage.assistant("Hi there! How can I help?"),
    ]


@pytest.fixture
def agent_state():
    """Sample agent state."""
    state = AgentState(task="Test task")
    state = state.add_message(Message(role=MessageRole.USER, content="Test task"))
    return state


@pytest.fixture
def simple_tool():
    """A simple test tool."""
    async def echo_tool(text: str) -> str:
        """Echo the input text."""
        return f"Echo: {text}"
    
    echo_tool.__name__ = "echo_tool"
    return echo_tool


@pytest.fixture
def completion_tool():
    """A tool that signals completion."""
    async def complete_tool(result: str) -> str:
        """Complete the task with a result."""
        return f"complete: {result}"
    
    complete_tool.__name__ = "complete_tool"
    return complete_tool
