"""
Tests for LLM Provider Abstraction Layer.

Tests the unified LLM interface and provider implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from squadron.llm.base import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    MessageRole,
    ToolDefinition,
    ToolCall,
)
from squadron.llm.factory import create_llm, LLMFactory, _detect_provider
from squadron.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
)


class TestLLMMessage:
    """Tests for LLMMessage."""
    
    def test_system_message(self):
        msg = LLMMessage.system("You are helpful.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful."
    
    def test_user_message(self):
        msg = LLMMessage.user("Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
    
    def test_assistant_message(self):
        msg = LLMMessage.assistant("Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_tool_message(self):
        msg = LLMMessage.tool("result", name="my_tool", tool_call_id="123")
        assert msg.role == MessageRole.TOOL
        assert msg.content == "result"
        assert msg.name == "my_tool"
        assert msg.tool_call_id == "123"
    
    def test_to_openai_format(self):
        msg = LLMMessage.user("Hello!")
        formatted = msg.to_openai_format()
        assert formatted["role"] == "user"
        assert formatted["content"] == "Hello!"
    
    def test_assistant_with_tool_calls(self):
        tool_calls = [ToolCall(id="1", name="test", arguments={"a": 1})]
        msg = LLMMessage.assistant("", tool_calls=tool_calls)
        formatted = msg.to_openai_format()
        assert "tool_calls" in formatted
        assert len(formatted["tool_calls"]) == 1


class TestToolCall:
    """Tests for ToolCall."""
    
    def test_to_openai_format(self):
        tc = ToolCall(id="123", name="my_func", arguments={"x": 1})
        formatted = tc.to_openai_format()
        assert formatted["id"] == "123"
        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "my_func"
    
    def test_from_openai_format(self):
        data = {
            "id": "456",
            "function": {
                "name": "other_func",
                "arguments": '{"y": 2}',
            },
        }
        tc = ToolCall.from_openai_format(data)
        assert tc.id == "456"
        assert tc.name == "other_func"
        assert tc.arguments == {"y": 2}


class TestToolDefinition:
    """Tests for ToolDefinition."""
    
    def test_to_openai_format(self):
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        formatted = tool.to_openai_format()
        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "search"
        assert formatted["function"]["description"] == "Search the web"
    
    def test_to_anthropic_format(self):
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        formatted = tool.to_anthropic_format()
        assert formatted["name"] == "search"
        assert formatted["description"] == "Search the web"
        assert "input_schema" in formatted


class TestLLMResponse:
    """Tests for LLMResponse."""
    
    def test_has_tool_calls(self):
        response = LLMResponse(content="Hello")
        assert not response.has_tool_calls
        
        response = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert response.has_tool_calls
    
    def test_to_message(self):
        response = LLMResponse(content="Hello!")
        msg = response.to_message()
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hello!"


class TestProviderDetection:
    """Tests for automatic provider detection."""
    
    def test_detect_from_base_url(self):
        provider = _detect_provider(None, "https://my-server.com", None)
        assert provider == "openai_compatible"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_detect_from_openai_env(self):
        provider = _detect_provider(None, None, None)
        assert provider == "openai"
    
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_detect_from_anthropic_env(self):
        provider = _detect_provider(None, None, None)
        assert provider == "anthropic"
    
    def test_detect_from_model_name_gpt(self):
        provider = _detect_provider(None, None, "gpt-4o")
        assert provider == "openai"
    
    def test_detect_from_model_name_claude(self):
        provider = _detect_provider(None, None, "claude-3-5-sonnet")
        assert provider == "anthropic"


class TestLLMFactory:
    """Tests for LLMFactory."""
    
    def test_register_preset(self):
        factory = LLMFactory()
        factory.register_preset("fast", provider="openai", model="gpt-4o-mini")
        
        preset = factory.get_preset("fast")
        assert preset["provider"] == "openai"
        assert preset["model"] == "gpt-4o-mini"
    
    def test_list_presets(self):
        factory = LLMFactory()
        factory.register_preset("fast", provider="openai", model="gpt-4o-mini")
        factory.register_preset("smart", provider="anthropic", model="claude-3-5-sonnet")
        
        presets = factory.list_presets()
        assert "fast" in presets
        assert "smart" in presets


class TestOpenAIProvider:
    """Tests for OpenAI provider."""
    
    def test_init(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        assert provider.model == "gpt-4o"
        assert provider.provider_name == "openai"
        assert provider.api_key == "test-key"
    
    def test_supports_vision(self):
        provider = OpenAIProvider(model="gpt-4o")
        assert provider.supports_vision
        
        provider = OpenAIProvider(model="gpt-3.5-turbo")
        assert not provider.supports_vision
    
    @pytest.mark.asyncio
    async def test_generate_mocked(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test")
        
        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.model = "gpt-4o"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        
        response = await provider.generate([LLMMessage.user("Hi")])
        
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.total_tokens == 15


class TestAnthropicProvider:
    """Tests for Anthropic provider."""
    
    def test_init(self):
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test")
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.provider_name == "anthropic"
    
    def test_supports_vision(self):
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        assert provider.supports_vision
        
        provider = AnthropicProvider(model="claude-2")
        assert not provider.supports_vision
    
    def test_format_messages(self):
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [
            LLMMessage.system("Be helpful"),
            LLMMessage.user("Hello"),
        ]
        system, formatted = provider.format_messages(messages)
        
        assert system == "Be helpful"
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"


class TestOllamaProvider:
    """Tests for Ollama provider."""
    
    def test_init(self):
        provider = OllamaProvider(model="llama3.2")
        assert provider.model == "llama3.2"
        assert provider.provider_name == "ollama"
        assert provider.base_url == "http://localhost:11434"
    
    def test_custom_base_url(self):
        provider = OllamaProvider(model="llama3.2", base_url="http://remote:11434/")
        assert provider.base_url == "http://remote:11434"


class TestHuggingFaceProvider:
    """Tests for HuggingFace provider."""
    
    def test_init(self):
        provider = HuggingFaceProvider(model="meta-llama/Llama-3.2-3B-Instruct")
        assert provider.model == "meta-llama/Llama-3.2-3B-Instruct"
        assert provider.provider_name == "huggingface"
        assert not provider.supports_tools
    
    def test_format_chat_prompt(self):
        provider = HuggingFaceProvider(model="test")
        messages = [
            LLMMessage.system("Be helpful"),
            LLMMessage.user("Hello"),
        ]
        prompt = provider._format_chat_prompt(messages)
        assert "SYSTEM:" in prompt
        assert "USER:" in prompt
        assert "ASSISTANT:" in prompt


class TestOpenAICompatibleProvider:
    """Tests for OpenAI-compatible provider."""
    
    def test_init(self):
        provider = OpenAICompatibleProvider(
            model="llama-70b",
            base_url="https://my-server.com",
            api_key="test-key",
        )
        assert provider.model == "llama-70b"
        assert provider.provider_name == "openai_compatible"
        assert provider.base_url == "https://my-server.com"
    
    @pytest.mark.asyncio
    async def test_generate_mocked(self):
        provider = OpenAICompatibleProvider(
            model="test-model",
            base_url="https://test.com",
        )
        
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "resp-123",
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        provider._session = mock_session
        
        response = await provider.generate([LLMMessage.user("Hi")])
        
        assert response.content == "Hello!"
        assert response.provider == "openai_compatible"
