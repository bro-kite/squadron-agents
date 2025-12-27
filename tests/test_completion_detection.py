"""
Tests for LLM-based task completion detection.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from squadron.core.agent import Agent, COMPLETION_DETECTION_PROMPT
from squadron.core.config import SquadronConfig
from squadron.core.state import AgentState, Message, MessageRole, ToolResult
from squadron.llm.base import LLMProvider, LLMResponse, LLMMessage


@pytest.fixture
def mock_llm_complete():
    """Mock LLM that indicates task is complete."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    response_content = json.dumps({
        "completed": True,
        "confidence": 0.95,
        "reason": "All required files have been read and analyzed successfully.",
    })

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content=f"Based on my analysis:\n```json\n{response_content}\n```",
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_incomplete():
    """Mock LLM that indicates task is incomplete."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    response_content = json.dumps({
        "completed": False,
        "confidence": 0.8,
        "reason": "The task asked to write a file, but no write operation was performed.",
    })

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_low_confidence():
    """Mock LLM that returns low confidence completion."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    response_content = json.dumps({
        "completed": True,
        "confidence": 0.4,
        "reason": "Might be complete but not certain.",
    })

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content=response_content,
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_invalid_response():
    """Mock LLM that returns invalid JSON."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content="I think the task is complete but I can't format JSON properly",
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_error():
    """Mock LLM that raises an error."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    llm.generate = AsyncMock(side_effect=Exception("LLM API error"))
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def state_with_successful_results():
    """Agent state with successful tool results."""
    state = AgentState(task="Read and analyze the main.py file")
    state = state.add_message(
        Message(role=MessageRole.USER, content="Read and analyze the main.py file")
    )
    state = state.add_message(
        Message(role=MessageRole.ASSISTANT, content="I'll read the file for you.")
    )

    # Add tool result
    tool_call_id = uuid4()
    result = ToolResult(
        tool_call_id=tool_call_id,
        tool_name="read_file",
        result="def main():\n    print('Hello')",
    )

    # Manually create new state with tool result
    state = state.model_copy(
        update={"tool_results": (*state.tool_results, result)}
    )

    state = state.add_message(
        Message(
            role=MessageRole.ASSISTANT,
            content="I have read the file. The analysis is complete.",
        )
    )

    return state


@pytest.fixture
def state_with_failed_results():
    """Agent state with failed tool results."""
    state = AgentState(task="Write to a protected file")
    state = state.add_message(
        Message(role=MessageRole.USER, content="Write to a protected file")
    )

    tool_call_id = uuid4()
    result = ToolResult(
        tool_call_id=tool_call_id,
        tool_name="write_file",
        result=None,
        error="Permission denied",
    )

    state = state.model_copy(
        update={"tool_results": (*state.tool_results, result)}
    )

    return state


@pytest.fixture
def state_with_no_results():
    """Agent state with no tool results."""
    state = AgentState(task="Test task")
    state = state.add_message(
        Message(role=MessageRole.USER, content="Test task")
    )
    return state


class TestCompletionDetectionWithLLM:
    """Tests for LLM-based completion detection."""

    @pytest.mark.asyncio
    async def test_llm_detects_completion(
        self, mock_llm_complete, state_with_successful_results
    ):
        """Test that LLM can detect task completion."""
        agent = Agent(name="test", llm=mock_llm_complete)

        result = await agent._is_task_complete_llm(state_with_successful_results)

        assert result is True
        mock_llm_complete.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_detects_incomplete(
        self, mock_llm_incomplete, state_with_successful_results
    ):
        """Test that LLM can detect incomplete tasks."""
        agent = Agent(name="test", llm=mock_llm_incomplete)

        result = await agent._is_task_complete_llm(state_with_successful_results)

        assert result is False

    @pytest.mark.asyncio
    async def test_low_confidence_not_complete(
        self, mock_llm_low_confidence, state_with_successful_results
    ):
        """Test that low confidence completions are not accepted."""
        agent = Agent(
            name="test",
            llm=mock_llm_low_confidence,
            completion_confidence_threshold=0.7,
        )

        result = await agent._is_task_complete_llm(state_with_successful_results)

        assert result is False

    @pytest.mark.asyncio
    async def test_custom_confidence_threshold(
        self, mock_llm_low_confidence, state_with_successful_results
    ):
        """Test custom confidence threshold."""
        # With a lower threshold, the 0.4 confidence should pass
        agent = Agent(
            name="test",
            llm=mock_llm_low_confidence,
            completion_confidence_threshold=0.3,
        )

        result = await agent._is_task_complete_llm(state_with_successful_results)

        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_json_returns_false(
        self, mock_llm_invalid_response, state_with_successful_results
    ):
        """Test that invalid JSON response falls back to heuristic."""
        agent = Agent(name="test", llm=mock_llm_invalid_response)

        result = await agent._is_task_complete_llm(state_with_successful_results)

        # Invalid JSON parsing returns False by default
        assert result is False

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_heuristic(
        self, mock_llm_error, state_with_successful_results
    ):
        """Test that LLM errors fall back to heuristic detection."""
        agent = Agent(name="test", llm=mock_llm_error)

        # Should use heuristic fallback and find "complete" in message
        result = await agent._is_task_complete_llm(state_with_successful_results)

        # Heuristic should return True due to "complete" keyword
        assert result is True


class TestHeuristicCompletionDetection:
    """Tests for heuristic-based completion detection."""

    def test_no_results_not_complete(self, state_with_no_results):
        """Test that no tool results means not complete."""
        agent = Agent(name="test")

        result = agent._is_task_complete_heuristic(state_with_no_results)

        assert result is False

    def test_completion_keyword_in_message(self, state_with_successful_results):
        """Test detection via completion keywords."""
        agent = Agent(name="test")

        result = agent._is_task_complete_heuristic(state_with_successful_results)

        assert result is True

    def test_multiple_successful_results_complete(self):
        """Test that multiple successful tool results indicate completion."""
        agent = Agent(name="test")

        state = AgentState(task="Multiple operations")

        # Add multiple successful results
        for i in range(3):
            result = ToolResult(
                tool_call_id=uuid4(),
                tool_name=f"tool_{i}",
                result=f"Result {i}",
            )
            state = state.model_copy(
                update={"tool_results": (*state.tool_results, result)}
            )

        result = agent._is_task_complete_heuristic(state)

        assert result is True

    def test_failed_results_not_complete(self, state_with_failed_results):
        """Test that failed tool results don't indicate completion."""
        agent = Agent(name="test")

        result = agent._is_task_complete_heuristic(state_with_failed_results)

        assert result is False


class TestCompletionDetectionIntegration:
    """Integration tests for completion detection."""

    def test_uses_llm_when_available(self, mock_llm_complete, state_with_successful_results):
        """Test that _is_task_complete uses LLM when available."""
        # Note: This is tricky to test because _is_task_complete is sync
        # but calls async _is_task_complete_llm. We test the async method directly.
        agent = Agent(name="test", llm=mock_llm_complete)

        # Without results, should return False immediately
        state_no_results = AgentState(task="Test")
        assert agent._is_task_complete(state_no_results) is False

    def test_uses_heuristic_without_llm(self, state_with_successful_results):
        """Test that _is_task_complete uses heuristic without LLM."""
        agent = Agent(name="test")  # No LLM

        result = agent._is_task_complete(state_with_successful_results)

        # Should use heuristic and find "complete" keyword
        assert result is True

    def test_set_llm_method(self, mock_llm_complete):
        """Test setting LLM after initialization."""
        agent = Agent(name="test")
        assert agent.llm is None

        agent.set_llm(mock_llm_complete)

        assert agent.llm is mock_llm_complete


class TestParseCompletionResponse:
    """Tests for parsing completion response."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        agent = Agent(name="test")

        content = json.dumps({
            "completed": True,
            "confidence": 0.9,
            "reason": "Task done",
        })

        result = agent._parse_completion_response(content)

        assert result["completed"] is True
        assert result["confidence"] == 0.9
        assert result["reason"] == "Task done"

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        agent = Agent(name="test")

        content = '''Here is my analysis:
```json
{
    "completed": true,
    "confidence": 0.85,
    "reason": "All steps completed"
}
```
'''
        result = agent._parse_completion_response(content)

        assert result["completed"] is True
        assert result["confidence"] == 0.85

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns defaults."""
        agent = Agent(name="test")

        result = agent._parse_completion_response("not json at all")

        assert result["completed"] is False
        assert result["confidence"] == 0.0

    def test_parse_no_json(self):
        """Test parsing response with no JSON."""
        agent = Agent(name="test")

        result = agent._parse_completion_response(
            "The task appears to be complete based on the evidence."
        )

        assert result["completed"] is False
        assert result["confidence"] == 0.0
