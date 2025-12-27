"""
Tests for LLM-based action generation in LATS reasoning.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from squadron.core.config import ReasoningConfig
from squadron.core.state import AgentState, Message, MessageRole
from squadron.llm.base import LLMProvider, LLMResponse, LLMMessage, ToolDefinition
from squadron.reasoning.lats import LATSReasoner, ACTION_GENERATION_PROMPT
from squadron.reasoning.verifier import CandidatePlan


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
                    "path": {"type": "string", "description": "File path to read"},
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
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        ),
        ToolDefinition(
            name="run_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"},
                },
                "required": ["command"],
            },
        ),
    ]


@pytest.fixture
def mock_llm_with_candidates():
    """Mock LLM that returns candidate actions."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    # Return a valid JSON response with candidates
    candidate_response = json.dumps([
        {
            "thought": "First, I should read the file to understand its contents",
            "tool_name": "read_file",
            "arguments": {"path": "/src/main.py"},
            "expected_outcome": "Get the file contents to analyze",
        },
        {
            "thought": "I could also check what's in the directory",
            "tool_name": "run_command",
            "arguments": {"command": "ls -la /src"},
            "expected_outcome": "See all files in the source directory",
        },
        {
            "thought": "Another approach is to write a test file",
            "tool_name": "write_file",
            "arguments": {"path": "/test.txt", "content": "test"},
            "expected_outcome": "Create a test file for verification",
        },
    ])

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content=f"Here are the candidate actions:\n```json\n{candidate_response}\n```",
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_invalid_response():
    """Mock LLM that returns invalid response."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content="This is not valid JSON at all",
            model="mock-model",
            provider="mock",
        )

    llm.generate = AsyncMock(side_effect=mock_generate)
    llm.close = AsyncMock()

    return llm


@pytest.fixture
def mock_llm_invalid_tools():
    """Mock LLM that suggests invalid tool names."""
    llm = MagicMock(spec=LLMProvider)
    llm.provider_name = "mock"
    llm.model = "mock-model"

    candidate_response = json.dumps([
        {
            "thought": "Use a tool that doesn't exist",
            "tool_name": "nonexistent_tool",
            "arguments": {"arg": "value"},
            "expected_outcome": "This should be filtered out",
        },
        {
            "thought": "This one is valid",
            "tool_name": "read_file",
            "arguments": {"path": "/valid.txt"},
            "expected_outcome": "Read valid file",
        },
    ])

    async def mock_generate(messages, tools=None, **kwargs):
        return LLMResponse(
            content=candidate_response,
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


class TestLATSReasonerInit:
    """Tests for LATSReasoner initialization."""

    def test_init_with_llm_and_tools(self, mock_llm_with_candidates, sample_tools):
        """Test initialization with LLM and tools."""
        reasoner = LATSReasoner(
            llm=mock_llm_with_candidates,
            tools=sample_tools,
        )

        assert reasoner.llm is mock_llm_with_candidates
        assert len(reasoner.tools) == 3
        assert reasoner.config.n_candidates == 5

    def test_init_without_llm(self):
        """Test initialization without LLM (fallback mode)."""
        reasoner = LATSReasoner(default_tool="echo")

        assert reasoner.llm is None
        assert reasoner.default_tool == "echo"

    def test_register_tools(self, sample_tools):
        """Test registering tools after initialization."""
        reasoner = LATSReasoner()
        reasoner.register_tools(sample_tools)

        assert len(reasoner.tools) == 3

    def test_set_llm(self, mock_llm_with_candidates):
        """Test setting LLM after initialization."""
        reasoner = LATSReasoner()
        reasoner.set_llm(mock_llm_with_candidates)

        assert reasoner.llm is mock_llm_with_candidates


class TestLLMCandidateGeneration:
    """Tests for LLM-based candidate generation."""

    @pytest.mark.asyncio
    async def test_generate_candidates_with_llm(
        self, mock_llm_with_candidates, sample_tools
    ):
        """Test generating candidates using LLM."""
        reasoner = LATSReasoner(
            llm=mock_llm_with_candidates,
            tools=sample_tools,
        )

        state = AgentState(task="Analyze the codebase")
        state = state.add_message(Message(role=MessageRole.USER, content="Analyze the codebase"))

        candidates = await reasoner._generate_candidate_calls_llm(state)

        assert len(candidates) == 3
        assert all(isinstance(c, CandidatePlan) for c in candidates)

        # Verify candidate properties
        tools_used = [c.action for c in candidates]
        assert "read_file" in tools_used
        assert "run_command" in tools_used
        assert "write_file" in tools_used

    @pytest.mark.asyncio
    async def test_generate_candidates_filters_invalid_tools(
        self, mock_llm_invalid_tools, sample_tools
    ):
        """Test that invalid tool names are filtered out."""
        reasoner = LATSReasoner(
            llm=mock_llm_invalid_tools,
            tools=sample_tools,
        )

        state = AgentState(task="Test task")
        candidates = await reasoner._generate_candidate_calls_llm(state)

        # Only the valid tool should remain
        assert len(candidates) == 1
        assert candidates[0].action == "read_file"

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(
        self, mock_llm_invalid_response, sample_tools
    ):
        """Test fallback when LLM returns invalid JSON."""
        reasoner = LATSReasoner(
            llm=mock_llm_invalid_response,
            tools=sample_tools,
            default_tool="read_file",
        )

        state = AgentState(task="Test task")
        candidates = await reasoner._generate_candidate_calls_llm(state)

        # Should fall back to default tool
        assert len(candidates) == 1
        assert candidates[0].action == "read_file"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, mock_llm_error, sample_tools):
        """Test fallback when LLM raises an error."""
        reasoner = LATSReasoner(
            llm=mock_llm_error,
            tools=sample_tools,
            default_tool="read_file",
        )

        state = AgentState(task="Test task")
        candidates = await reasoner._generate_candidate_calls_llm(state)

        # Should fall back to default tool
        assert len(candidates) == 1
        assert candidates[0].action == "read_file"


class TestPlanMethod:
    """Tests for the plan() method with LLM integration."""

    @pytest.mark.asyncio
    async def test_plan_uses_llm_when_available(
        self, mock_llm_with_candidates, sample_tools
    ):
        """Test that plan() uses LLM when available."""
        reasoner = LATSReasoner(
            llm=mock_llm_with_candidates,
            tools=sample_tools,
        )

        state = AgentState(task="Analyze the code")
        new_state = await reasoner.plan(state)

        # Should have added a tool call
        assert len(new_state.pending_tool_calls) == 1
        assert new_state.pending_tool_calls[0].tool_name in [
            "read_file", "write_file", "run_command"
        ]

        # Should have added planning message
        assert len(new_state.messages) == 1
        assert "Planning to call tool" in new_state.messages[0].content

    @pytest.mark.asyncio
    async def test_plan_uses_fallback_without_llm(self):
        """Test that plan() uses fallback when LLM is not available."""
        reasoner = LATSReasoner(
            default_tool="echo_tool",
            tool_args_fn=lambda s: {"text": s.task},
        )

        state = AgentState(task="Test task")
        new_state = await reasoner.plan(state)

        assert len(new_state.pending_tool_calls) == 1
        assert new_state.pending_tool_calls[0].tool_name == "echo_tool"

    @pytest.mark.asyncio
    async def test_plan_no_tools_available(self):
        """Test plan() behavior when no tools are available."""
        reasoner = LATSReasoner()

        state = AgentState(task="Test task")
        new_state = await reasoner.plan(state)

        # Should add a message but no tool calls
        assert len(new_state.pending_tool_calls) == 0
        assert len(new_state.messages) == 1
        assert "No tools available" in new_state.messages[0].content


class TestPromptFormatting:
    """Tests for prompt formatting helpers."""

    def test_format_tools_for_prompt(self, sample_tools):
        """Test formatting tools for the prompt."""
        reasoner = LATSReasoner(tools=sample_tools)
        formatted = reasoner._format_tools_for_prompt()

        assert "read_file" in formatted
        assert "write_file" in formatted
        assert "run_command" in formatted
        assert "Read contents of a file" in formatted

    def test_format_history(self):
        """Test formatting conversation history."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")
        state = state.add_message(Message(role=MessageRole.USER, content="Hello"))
        state = state.add_message(Message(role=MessageRole.ASSISTANT, content="Hi there"))

        formatted = reasoner._format_history(state)

        assert "[USER]: Hello" in formatted
        assert "[ASSISTANT]: Hi there" in formatted

    def test_format_history_empty(self):
        """Test formatting empty history."""
        reasoner = LATSReasoner()
        state = AgentState(task="Test")

        formatted = reasoner._format_history(state)

        assert formatted == "No previous conversation."

    def test_format_context(self):
        """Test formatting memory context."""
        reasoner = LATSReasoner()
        context = {
            "fact1": "The sky is blue",
            "data": {"key": "value"},
        }

        formatted = reasoner._format_context(context)

        assert "fact1: The sky is blue" in formatted
        assert "data:" in formatted

    def test_format_context_empty(self):
        """Test formatting empty context."""
        reasoner = LATSReasoner()
        formatted = reasoner._format_context({})

        assert formatted == "No additional context."


class TestParseLLMCandidates:
    """Tests for parsing LLM responses into candidates."""

    def test_parse_valid_json(self, sample_tools):
        """Test parsing valid JSON response."""
        reasoner = LATSReasoner(tools=sample_tools)

        content = json.dumps([
            {
                "thought": "Read the file",
                "tool_name": "read_file",
                "arguments": {"path": "/test.txt"},
                "expected_outcome": "Get file contents",
            }
        ])

        candidates = reasoner._parse_llm_candidates(content)

        assert len(candidates) == 1
        assert candidates[0].action == "read_file"
        assert candidates[0].thought == "Read the file"
        assert candidates[0].context["arguments"]["path"] == "/test.txt"

    def test_parse_json_with_markdown(self, sample_tools):
        """Test parsing JSON wrapped in markdown code blocks."""
        reasoner = LATSReasoner(tools=sample_tools)

        content = """Here are the candidates:
```json
[
    {
        "thought": "Read the file",
        "tool_name": "read_file",
        "arguments": {"path": "/test.txt"},
        "expected_outcome": "Get file contents"
    }
]
```
"""
        candidates = reasoner._parse_llm_candidates(content)

        assert len(candidates) == 1
        assert candidates[0].action == "read_file"

    def test_parse_invalid_json(self, sample_tools):
        """Test parsing invalid JSON returns empty list."""
        reasoner = LATSReasoner(tools=sample_tools)
        candidates = reasoner._parse_llm_candidates("not valid json")

        assert candidates == []

    def test_parse_filters_invalid_tools(self, sample_tools):
        """Test that invalid tool names are filtered."""
        reasoner = LATSReasoner(tools=sample_tools)

        content = json.dumps([
            {
                "thought": "Invalid tool",
                "tool_name": "fake_tool",
                "arguments": {},
                "expected_outcome": "Nothing",
            },
            {
                "thought": "Valid tool",
                "tool_name": "read_file",
                "arguments": {"path": "/file.txt"},
                "expected_outcome": "Read file",
            },
        ])

        candidates = reasoner._parse_llm_candidates(content)

        assert len(candidates) == 1
        assert candidates[0].action == "read_file"

    def test_parse_handles_missing_fields(self, sample_tools):
        """Test parsing handles missing optional fields."""
        reasoner = LATSReasoner(tools=sample_tools)

        content = json.dumps([
            {
                "tool_name": "read_file",
                "arguments": {"path": "/file.txt"},
            }
        ])

        candidates = reasoner._parse_llm_candidates(content)

        assert len(candidates) == 1
        assert candidates[0].thought == "Use read_file"
        assert candidates[0].expected_outcome == "Progress on task"
