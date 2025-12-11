"""
LLM Provider Implementations

Concrete implementations for various LLM providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Ollama (local models)
- Hugging Face (transformers, Inference API)
- OpenAI-compatible APIs (vLLM, LiteLLM, LocalAI, etc.)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import structlog

from squadron.llm.base import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    MessageRole,
    ToolDefinition,
    ToolCall,
)

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for GPT-4, GPT-3.5, etc."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self._client: Any = None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def supports_vision(self) -> bool:
        return "vision" in self.model or "gpt-4o" in self.model
    
    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required. Run: pip install openai")
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        return self._client
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        start_time = time.time()
        
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if tools:
            request["tools"] = self.format_tools(tools)
            if tool_choice:
                request["tool_choice"] = tool_choice
        
        response = await client.chat.completions.create(**request)
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall.from_openai_format({
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }))
        
        return LLMResponse(
            id=response.id,
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            model=response.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=response,
        )
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        if tools:
            request["tools"] = self.format_tools(tools)
        
        stream = await client.chat.completions.create(**request)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        response = await client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]
    
    async def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            return len(text) // 4
    
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude models."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def supports_vision(self) -> bool:
        return "claude-3" in self.model
    
    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required. Run: pip install anthropic")
            self._client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def format_messages(self, messages: list[LLMMessage]) -> tuple[str | None, list[dict]]:
        system_message = None
        formatted = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            elif msg.role == MessageRole.TOOL:
                formatted.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}],
                })
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                formatted.append({"role": "assistant", "content": content})
            else:
                formatted.append({"role": msg.role.value, "content": msg.content})
        return system_message, formatted
    
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [tool.to_anthropic_format() for tool in tools]
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = await self._get_client()
        start_time = time.time()
        system_message, formatted_messages = self.format_messages(messages)
        
        request: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system_message:
            request["system"] = system_message
        if tools:
            request["tools"] = self.format_tools(tools)
            if tool_choice == "auto":
                request["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                request["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict):
                request["tool_choice"] = tool_choice
        
        response = await client.messages.create(**request)
        latency_ms = (time.time() - start_time) * 1000
        
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
        
        return LLMResponse(
            id=response.id,
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=response.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=response,
        )
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        system_message, formatted_messages = self.format_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system_message:
            request["system"] = system_message
        if tools:
            request["tools"] = self.format_tools(tools)
        async with client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._session: Any = None
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    async def _get_session(self) -> Any:
        if self._session is None:
            try:
                import httpx
                self._session = httpx.AsyncClient(timeout=120.0)
            except ImportError:
                raise ImportError("httpx package required. Run: pip install httpx")
        return self._session
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        session = await self._get_session()
        start_time = time.time()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_openai_format() for msg in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        if tools:
            request["tools"] = self.format_tools(tools)
        
        response = await session.post(self.base_url + "/api/chat", json=request)
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        message = data.get("message", {})
        tool_calls = []
        if message.get("tool_calls"):
            for i, tc in enumerate(message["tool_calls"]):
                tool_calls.append(ToolCall(
                    id=tc.get("id", str(i)),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", {}),
                ))
        
        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=tool_calls,
            finish_reason=data.get("done_reason", "stop"),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            model=self.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_openai_format() for msg in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        async with session.stream("POST", self.base_url + "/api/chat", json=request) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content")
                    if content:
                        yield content
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        session = await self._get_session()
        embeddings = []
        for text in texts:
            response = await session.post(
                self.base_url + "/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json().get("embedding", []))
        return embeddings
    
    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None


class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider for Inference API, TGI, and local models."""
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_local: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.use_local = use_local
        self._session: Any = None
        self._local_pipeline: Any = None
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    @property
    def supports_tools(self) -> bool:
        return False
    
    async def _get_session(self) -> Any:
        if self._session is None:
            try:
                import httpx
                self._session = httpx.AsyncClient(timeout=120.0)
            except ImportError:
                raise ImportError("httpx package required. Run: pip install httpx")
        return self._session
    
    async def _get_local_pipeline(self) -> Any:
        if self._local_pipeline is None:
            try:
                from transformers import pipeline
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                self._local_pipeline = pipeline("text-generation", model=self.model, device=device, torch_dtype=dtype)
            except ImportError:
                raise ImportError("transformers package required. Run: pip install transformers torch")
        return self._local_pipeline
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.use_local:
            return await self._generate_local(messages, **kwargs)
        elif self.base_url:
            return await self._generate_tgi(messages, **kwargs)
        else:
            return await self._generate_inference_api(messages, **kwargs)
    
    async def _generate_inference_api(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        session = await self._get_session()
        start_time = time.time()
        formatted = [msg.to_openai_format() for msg in messages]
        
        url = "https://api-inference.huggingface.co/models/" + self.model + "/v1/chat/completions"
        response = await session.post(
            url,
            headers={"Authorization": "Bearer " + (self.api_key or "")},
            json={
                "messages": formatted,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return LLMResponse(
            id=data.get("id", ""),
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            model=self.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def _generate_tgi(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        session = await self._get_session()
        start_time = time.time()
        formatted = [msg.to_openai_format() for msg in messages]
        
        response = await session.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": formatted,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return LLMResponse(
            id=data.get("id", ""),
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            model=self.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def _generate_local(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        pipeline_obj = await self._get_local_pipeline()
        start_time = time.time()
        prompt = self._format_chat_prompt(messages)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline_obj(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
                return_full_text=False,
            ),
        )
        latency_ms = (time.time() - start_time) * 1000
        content = result[0]["generated_text"] if result else ""
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=result,
        )
    
    def _format_chat_prompt(self, messages: list[LLMMessage]) -> str:
        """Format messages as a chat prompt for local models."""
        parts = []
        for msg in messages:
            role = msg.role.value
            content = msg.content
            parts.append(role.upper() + ": " + content)
        return "\n".join(parts) + "\nASSISTANT:"
    
    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for OpenAI-compatible APIs.
    
    Works with: vLLM, LiteLLM, LocalAI, text-generation-webui, 
    DigitalOcean GPU Droplets, RunPod, Together AI, Anyscale, etc.
    """
    
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: Any = None
    
    @property
    def provider_name(self) -> str:
        return "openai_compatible"
    
    async def _get_session(self) -> Any:
        if self._session is None:
            try:
                import httpx
                self._session = httpx.AsyncClient(timeout=120.0)
            except ImportError:
                raise ImportError("httpx package required. Run: pip install httpx")
        return self._session
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        session = await self._get_session()
        start_time = time.time()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_openai_format() for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if tools:
            request_body["tools"] = self.format_tools(tools)
            if tool_choice:
                request_body["tool_choice"] = tool_choice
        
        response = await session.post(
            self.base_url + "/v1/chat/completions",
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall.from_openai_format(tc))
        
        return LLMResponse(
            id=data.get("id", ""),
            content=message.get("content", ""),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            model=data.get("model", self.model),
            provider=self.provider_name,
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_openai_format() for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        
        if tools:
            request_body["tools"] = self.format_tools(tools)
        
        async with session.stream(
            "POST",
            self.base_url + "/v1/chat/completions",
            headers=headers,
            json=request_body,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        session = await self._get_session()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        
        response = await session.post(
            self.base_url + "/v1/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        
        return [item["embedding"] for item in data.get("data", [])]
    
    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None
