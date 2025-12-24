"""
Tests for MCTS integration with LATS reasoning.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from squadron.core.config import ReasoningConfig
from squadron.core.state import AgentState, AgentPhase, Message, MessageRole, ToolResult
from squadron.llm.base import LLMProvider, LLMResponse, LLMMessage, ToolDefinition
from squadron.reasoning.lats import LATSReasoner
from squadron.reasoning.mcts import MCTSController, MCTSNode


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
    ]


@pytest.fixture
def mock_llm_for_mcts():
    """Mock LLM for MCTS testing."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    call_count = [0]

    async def mock_generate(messages, tools=None, **kwargs):
        call_count[0] += 1

        # Check if this is an expansion call or evaluation call
        prompt = messages[0].content if messages else ""

        if "Evaluate" in prompt:
            # State evaluation - return a value
            return LLMResponse(
                content="0.75",
                model="mock-model",
                provider="mock",
            )
        else:
            # Candidate generation
            candidates = json.dumps([
                {
                    "thought": f"Read file to gather information (call {call_count[0]})",
                    "tool_name": "read_file",
                    "arguments": {"path": "/src/main.py"},
                    "expected_outcome": "Get file contents",
                },
                {
                    "thought": f"Write a test file (call {call_count[0]})",
                    "tool_name": "write_file",
                    "arguments": {"path": "/test.txt", "content": "test"},
                    "expected_outcome": "Create test file",
                },
            ])
            return LLMResponse(
                content=candidates,
                model="mock-model",
                provider="mock",
            )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


class TestMCTSControllerBasic:
    """Basic tests for MCTS controller."""

    def test_init(self):
        """Test MCTS controller initialization."""
        def expand_fn(state):
            return []

        def simulate_fn(state):
            return 0.5

        mcts = MCTSController(
            expand_fn=expand_fn,
            simulate_fn=simulate_fn,
        )

        assert mcts.exploration_constant == 1.414
        assert mcts.max_depth == 10

    @pytest.mark.asyncio
    async def test_search_empty_expansions(self):
        """Test MCTS search with no expansions."""
        def expand_fn(state):
            return []

        def simulate_fn(state):
            return 0.5

        mcts = MCTSController(
            expand_fn=expand_fn,
            simulate_fn=simulate_fn,
        )

        state = AgentState(task="Test")
        best_action, trajectory = await mcts.search(state, budget=10)

        assert best_action is None
        assert len(trajectory) == 1  # Just root

    @pytest.mark.asyncio
    async def test_search_with_expansions(self):
        """Test MCTS search with valid expansions."""
        def expand_fn(state):
            return [
                ({"action": "a1"}, "Action 1", state),
                ({"action": "a2"}, "Action 2", state),
            ]

        def simulate_fn(state):
            return 0.7

        mcts = MCTSController(
            expand_fn=expand_fn,
            simulate_fn=simulate_fn,
        )

        state = AgentState(task="Test")
        best_action, trajectory = await mcts.search(state, budget=20)

        assert best_action is not None
        assert "action" in best_action

    def test_tree_stats(self):
        """Test tree statistics."""
        mcts = MCTSController(
            expand_fn=lambda s: [],
            simulate_fn=lambda s: 0.5,
        )

        # Before search
        stats = mcts.tree_stats
        assert stats["nodes"] == 0

    def test_get_top_actions(self):
        """Test getting top actions after search."""
        mcts = MCTSController(
            expand_fn=lambda s: [],
            simulate_fn=lambda s: 0.5,
        )

        # Before search
        top = mcts.get_top_actions()
        assert top == []


class TestMCTSNode:
    """Tests for MCTS nodes."""

    def test_node_creation(self):
        """Test node creation."""
        node = MCTSNode(
            state=AgentState(task="Test"),
            depth=0,
        )

        assert node.visits == 0
        assert node.value == 0.0
        assert not node.is_terminal

    def test_node_value_calculation(self):
        """Test node value calculation."""
        node = MCTSNode(state=None)
        node.visits = 10
        node.total_value = 7.5

        assert node.value == 0.75

    def test_node_ucb_unexplored(self):
        """Test UCB score for unexplored nodes."""
        node = MCTSNode(state=None)
        assert node.ucb_score == float("inf")


class TestLATSMCTSIntegration:
    """Tests for LATS MCTS integration."""

    def test_init_mcts_with_llm(self, mock_llm_for_mcts, sample_tools):
        """Test initializing MCTS when LLM is set."""
        reasoner = LATSReasoner(
            tools=sample_tools,
        )

        # Initially uses stub functions
        assert reasoner.mcts.expand_fn == reasoner._expand_stub

        # After setting LLM, should use real functions
        reasoner.set_llm(mock_llm_for_mcts)

        # MCTS should be reinitialized
        assert reasoner.mcts.expand_fn == reasoner._mcts_expand

    @pytest.mark.asyncio
    async def test_plan_with_mcts_no_llm(self, sample_tools):
        """Test MCTS planning falls back when no LLM."""
        reasoner = LATSReasoner(
            tools=sample_tools,
            default_tool="read_file",
        )

        state = AgentState(task="Test task")
        result = await reasoner.plan_with_mcts(state)

        # Should fall back to simple plan
        assert len(result.pending_tool_calls) == 1

    @pytest.mark.asyncio
    async def test_generate_mcts_expansions(self, mock_llm_for_mcts, sample_tools):
        """Test generating expansions for MCTS."""
        reasoner = LATSReasoner(
            llm=mock_llm_for_mcts,
            tools=sample_tools,
        )

        state = AgentState(task="Analyze the codebase")
        expansions = await reasoner._generate_mcts_expansions(state)

        assert len(expansions) >= 1
        for action, description, new_state in expansions:
            assert "tool_name" in action
            assert isinstance(description, str)
            assert isinstance(new_state, AgentState)


class TestHeuristicStateValue:
    """Tests for heuristic state evaluation."""

    def test_empty_state(self):
        """Test evaluation of empty state."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")

        value = reasoner._heuristic_state_value(state)

        assert value == 0.0

    def test_state_with_successful_results(self):
        """Test evaluation with successful tool results."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")

        # Add successful result
        result = ToolResult(
            tool_call_id=uuid4(),
            tool_name="test_tool",
            result="success",
        )
        state = state.model_copy(
            update={"tool_results": (*state.tool_results, result)}
        )

        value = reasoner._heuristic_state_value(state)

        # Should have positive value for successful result
        assert value > 0.0

    def test_state_with_errors(self):
        """Test evaluation with errors."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")
        state = state.add_error("Something went wrong")

        value = reasoner._heuristic_state_value(state)

        # Should be clamped to 0
        assert value == 0.0

    def test_state_with_completion_message(self):
        """Test evaluation with completion indicator."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")
        state = state.add_message(
            Message(role=MessageRole.ASSISTANT, content="Task complete!")
        )

        # Add a tool result so base value is > 0
        result = ToolResult(
            tool_call_id=uuid4(),
            tool_name="test_tool",
            result="done",
        )
        state = state.model_copy(
            update={"tool_results": (*state.tool_results, result)}
        )

        value = reasoner._heuristic_state_value(state)

        # Should have high value for completion indicator
        assert value >= 0.5


class TestMCTSTerminalCheck:
    """Tests for terminal state detection."""

    def test_completed_state_is_terminal(self):
        """Test that completed state is terminal."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test", phase=AgentPhase.COMPLETED)

        assert reasoner._mcts_is_terminal(state) is True

    def test_error_state_is_terminal(self):
        """Test that error state is terminal."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test", phase=AgentPhase.ERROR)

        assert reasoner._mcts_is_terminal(state) is True

    def test_max_iterations_is_terminal(self):
        """Test that max iterations is terminal."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test", iteration=50, max_iterations=50)

        assert reasoner._mcts_is_terminal(state) is True

    def test_completion_keyword_is_terminal(self):
        """Test that completion keyword in message is terminal."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")
        state = state.add_message(
            Message(role=MessageRole.ASSISTANT, content="The task is done.")
        )

        assert reasoner._mcts_is_terminal(state) is True

    def test_normal_state_not_terminal(self):
        """Test that normal state is not terminal."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test", phase=AgentPhase.ACTING)

        assert reasoner._mcts_is_terminal(state) is False


class TestLLMStateValue:
    """Tests for LLM-based state evaluation."""

    @pytest.mark.asyncio
    async def test_llm_state_value(self, mock_llm_for_mcts, sample_tools):
        """Test LLM-based state evaluation."""
        reasoner = LATSReasoner(
            llm=mock_llm_for_mcts,
            tools=sample_tools,
        )

        state = AgentState(task="Analyze code")
        value = await reasoner._llm_state_value(state)

        # Mock returns 0.75
        assert value == 0.75

    @pytest.mark.asyncio
    async def test_llm_state_value_parses_number(self, sample_tools):
        """Test that various LLM response formats are parsed."""
        llm = MagicMock(spec=LLMProvider)
        llm.provider_name = "mock"

        async def mock_generate(messages, **kwargs):
            return LLMResponse(
                content="Based on my analysis, I'd rate this state at 0.85",
                model="mock-model",
                provider="mock",
            )

        llm.generate = AsyncMock(side_effect=mock_generate)

        reasoner = LATSReasoner(llm=llm, tools=sample_tools)
        state = AgentState(task="Test")

        value = await reasoner._llm_state_value(state)

        assert value == 0.85

    @pytest.mark.asyncio
    async def test_llm_state_value_clamps(self, sample_tools):
        """Test that values are clamped to [0, 1]."""
        llm = MagicMock(spec=LLMProvider)
        llm.provider_name = "mock"

        async def mock_generate(messages, **kwargs):
            return LLMResponse(
                content="1.5",  # Out of range
                model="mock-model",
                provider="mock",
            )

        llm.generate = AsyncMock(side_effect=mock_generate)

        reasoner = LATSReasoner(llm=llm, tools=sample_tools)
        state = AgentState(task="Test")

        value = await reasoner._llm_state_value(state)

        assert value == 1.0  # Clamped
