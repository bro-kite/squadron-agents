#!/usr/bin/env python3
"""
LLM Provider Examples

Demonstrates how to use Squadron with different LLM providers:
- OpenAI
- Anthropic
- Ollama (local)
- Hugging Face
- OpenAI-compatible endpoints (vLLM, DigitalOcean, etc.)
"""

import asyncio
import os

from squadron import (
    create_llm,
    LLMMessage,
    LLMFactory,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
)


async def example_openai():
    """Example using OpenAI."""
    print("\n" + "=" * 50)
    print("🤖 OpenAI Example")
    print("=" * 50)
    
    # Method 1: Using create_llm factory (recommended)
    llm = create_llm(
        provider="openai",
        model="gpt-4o",
        # api_key is auto-loaded from OPENAI_API_KEY env var
    )
    
    messages = [
        LLMMessage.system("You are a helpful coding assistant."),
        LLMMessage.user("Write a Python function to calculate fibonacci numbers."),
    ]
    
    response = await llm.generate(messages)
    
    print(f"Model: {response.model}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nResponse:\n{response.content[:500]}...")
    
    await llm.close()


async def example_anthropic():
    """Example using Anthropic Claude."""
    print("\n" + "=" * 50)
    print("🧠 Anthropic Claude Example")
    print("=" * 50)
    
    llm = create_llm(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        # api_key is auto-loaded from ANTHROPIC_API_KEY env var
    )
    
    messages = [
        LLMMessage.system("You are an expert software architect."),
        LLMMessage.user("Explain the benefits of event-driven architecture."),
    ]
    
    response = await llm.generate(messages)
    
    print(f"Model: {response.model}")
    print(f"Tokens: {response.total_tokens}")
    print(f"\nResponse:\n{response.content[:500]}...")
    
    await llm.close()


async def example_ollama():
    """Example using Ollama for local models."""
    print("\n" + "=" * 50)
    print("🦙 Ollama (Local) Example")
    print("=" * 50)
    
    llm = create_llm(
        provider="ollama",
        model="llama3.2",  # Or any model you have pulled
        base_url="http://localhost:11434",
    )
    
    messages = [
        LLMMessage.user("What is the capital of France?"),
    ]
    
    response = await llm.generate(messages)
    
    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nResponse:\n{response.content}")
    
    await llm.close()


async def example_huggingface_api():
    """Example using Hugging Face Inference API."""
    print("\n" + "=" * 50)
    print("🤗 Hugging Face Inference API Example")
    print("=" * 50)
    
    llm = create_llm(
        provider="huggingface",
        model="meta-llama/Llama-3.2-3B-Instruct",
        # api_key is auto-loaded from HF_TOKEN env var
    )
    
    messages = [
        LLMMessage.user("Explain quantum computing in simple terms."),
    ]
    
    response = await llm.generate(messages)
    
    print(f"Model: {response.model}")
    print(f"\nResponse:\n{response.content[:500]}...")
    
    await llm.close()


async def example_openai_compatible():
    """Example using OpenAI-compatible endpoint (vLLM, DigitalOcean, etc.)."""
    print("\n" + "=" * 50)
    print("🔌 OpenAI-Compatible Endpoint Example")
    print("=" * 50)
    
    # This works with:
    # - vLLM servers
    # - Text Generation Inference (TGI)
    # - LocalAI
    # - DigitalOcean GPU Droplets
    # - RunPod
    # - Together AI
    # - Anyscale
    # - Any OpenAI-compatible API
    
    llm = create_llm(
        provider="openai_compatible",
        model="llama-3.2-70b",  # Model name depends on your deployment
        base_url="https://your-server.example.com",  # Your server URL
        api_key="your-api-key",  # If required
    )
    
    messages = [
        LLMMessage.system("You are a helpful assistant."),
        LLMMessage.user("Hello!"),
    ]
    
    response = await llm.generate(messages)
    
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"\nResponse:\n{response.content}")
    
    await llm.close()


async def example_streaming():
    """Example of streaming responses."""
    print("\n" + "=" * 50)
    print("📡 Streaming Example")
    print("=" * 50)
    
    llm = create_llm(model="gpt-4o")
    
    messages = [
        LLMMessage.user("Count from 1 to 10, one number per line."),
    ]
    
    print("Streaming response:")
    async for chunk in llm.generate_stream(messages):
        print(chunk, end="", flush=True)
    print()
    
    await llm.close()


async def example_tool_calling():
    """Example of tool/function calling."""
    print("\n" + "=" * 50)
    print("🔧 Tool Calling Example")
    print("=" * 50)
    
    from squadron.llm.base import ToolDefinition
    
    llm = create_llm(model="gpt-4o")
    
    # Define tools
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["location"],
            },
        ),
    ]
    
    messages = [
        LLMMessage.user("What's the weather like in Tokyo?"),
    ]
    
    response = await llm.generate(messages, tools=tools)
    
    if response.has_tool_calls:
        print("Tool calls requested:")
        for tc in response.tool_calls:
            print(f"  - {tc.name}({tc.arguments})")
    else:
        print(f"Response: {response.content}")
    
    await llm.close()


async def example_factory_presets():
    """Example using LLMFactory with presets."""
    print("\n" + "=" * 50)
    print("🏭 Factory Presets Example")
    print("=" * 50)
    
    factory = LLMFactory()
    
    # Register presets for different use cases
    factory.register_preset(
        "fast",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
    )
    
    factory.register_preset(
        "smart",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
    )
    
    factory.register_preset(
        "local",
        provider="ollama",
        model="llama3.2",
    )
    
    print("Available presets:", factory.list_presets())
    
    # Create from preset
    llm = factory.create("fast")
    print(f"Created: {llm}")
    
    # Override preset values
    llm = factory.create("fast", temperature=0.9)
    print(f"Created with override: {llm}")
    
    await factory.close_all()


async def example_auto_detection():
    """Example of automatic provider detection."""
    print("\n" + "=" * 50)
    print("🔍 Auto-Detection Example")
    print("=" * 50)
    
    # create_llm automatically detects the provider based on:
    # 1. Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    # 2. Model name patterns (gpt-*, claude-*, llama*, etc.)
    # 3. base_url presence (assumes OpenAI-compatible)
    
    # Auto-detect from env vars
    llm = create_llm()
    print(f"Auto-detected: {llm}")
    
    # Auto-detect from model name
    llm = create_llm(model="claude-3-5-sonnet-20241022")
    print(f"From model name: {llm}")
    
    # Auto-detect from base_url
    llm = create_llm(base_url="https://my-server.com", model="custom-model")
    print(f"From base_url: {llm}")


async def main():
    """Run examples based on available API keys."""
    print("🚀 Squadron LLM Provider Examples")
    print("=" * 50)
    
    # Check which providers are available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_huggingface = bool(os.getenv("HF_TOKEN"))
    
    print(f"OpenAI available: {'✅' if has_openai else '❌'}")
    print(f"Anthropic available: {'✅' if has_anthropic else '❌'}")
    print(f"Hugging Face available: {'✅' if has_huggingface else '❌'}")
    print(f"Ollama: Check if running locally")
    
    # Run available examples
    try:
        if has_openai:
            await example_openai()
            await example_streaming()
            await example_tool_calling()
        
        if has_anthropic:
            await example_anthropic()
        
        # Ollama example (uncomment if Ollama is running)
        # await example_ollama()
        
        if has_huggingface:
            await example_huggingface_api()
        
        # Factory example
        await example_factory_presets()
        
        # Auto-detection example
        await example_auto_detection()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
