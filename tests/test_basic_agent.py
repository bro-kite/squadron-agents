import pytest

from squadron import Agent, SquadronConfig
from squadron.core.state import AgentPhase
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


def test_agent_initialization():
    """Test that agent can be initialized with basic config."""
    config = SquadronConfig()
    config.governance.max_iterations = 5

    memory = GraphitiMemory()

    reasoner = LATSReasoner(
        config=config.reasoning,
        memory=memory,
        default_tool="complete_tool",
    )

    agent = Agent(
        name="test-agent",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )

    assert agent.name == "test-agent"
    assert agent.config == config


def test_agent_register_tool():
    """Test that tools can be registered with the agent."""
    config = SquadronConfig()
    memory = GraphitiMemory()
    reasoner = LATSReasoner(config=config.reasoning, memory=memory)

    agent = Agent(
        name="test-agent",
        config=config,
        memory=memory,
        reasoner=reasoner,
    )

    async def my_tool(text: str) -> str:
        return text

    agent.register_tool(my_tool)
    
    assert "my_tool" in agent._tool_registry
