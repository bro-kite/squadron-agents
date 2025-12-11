#!/usr/bin/env python3
"""
General-purpose Squadron Agent Example

Demonstrates domain-agnostic usage with simple, non-coding tools.
"""

import asyncio

from squadron import Agent, SquadronConfig
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


async def main():
    """Run a general-purpose Squadron agent."""

    print("🤖 Squadron Agent Framework - General Example")
    print("=" * 55)

    config = SquadronConfig()
    config.governance.max_iterations = 10

    memory = GraphitiMemory()
    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="echo",
    )

    agent = Agent(
        name="generalist",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )

    _register_general_tools(agent)

    tasks = [
        "Say hello and tell me what this agent can do.",
        "Classify this request: please write a short blog outline.",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task}")
        print("-" * 50)

        final_state = await agent.run(task)
        print(f"Phase: {final_state.phase}")
        print(f"Iterations: {final_state.iteration}")

        if final_state.messages:
            print("\n💬 Conversation (last 3):")
            for msg in final_state.messages[-3:]:
                print(f"  {msg.role.value.upper()}: {msg.content}")

        if final_state.tool_results:
            print("\n🔧 Tool Results:")
            for result in final_state.tool_results:
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.tool_name} -> {result.result}")


def _register_general_tools(agent: Agent):
    """Register simple, domain-agnostic tools."""

    async def echo(text: str) -> str:
        return f"Echo: {text}"

    async def classify_intent(text: str) -> str:
        lowered = text.lower()
        if "write" in lowered or "draft" in lowered:
            return "writing"
        if "bug" in lowered or "error" in lowered:
            return "debugging"
        if "summarize" in lowered or "research" in lowered:
            return "research"
        return "general"

    agent.register_tool(echo)
    agent.register_tool(classify_intent)
    print("🔧 Registered general tools (echo, classify_intent)")


if __name__ == "__main__":
    asyncio.run(main())
