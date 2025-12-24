"""
Tests for agent resume functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from squadron.core.agent import Agent
from squadron.core.config import SquadronConfig
from squadron.core.state import AgentState, AgentPhase, Message, MessageRole


class MockCheckpointTuple:
    """Mock checkpoint tuple for testing."""

    def __init__(self, state_data: dict):
        self.checkpoint = {"channel_values": state_data}
        self.config = {}
        self.metadata = {}


@pytest.fixture
def mock_checkpointer_with_state():
    """Mock checkpointer that returns a valid state."""
    checkpointer = MagicMock()

    state = AgentState(
        task="Test task",
        phase=AgentPhase.ACTING,
        iteration=2,
    )
    state = state.add_message(
        Message(role=MessageRole.USER, content="Test task")
    )

    # Return the state directly in channel_values
    checkpoint_tuple = MockCheckpointTuple({"state": state})

    checkpointer.get_tuple = MagicMock(return_value=checkpoint_tuple)
    checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

    return checkpointer


@pytest.fixture
def mock_checkpointer_awaiting_approval():
    """Mock checkpointer with state awaiting approval."""
    checkpointer = MagicMock()

    state = AgentState(
        task="Delete important file",
        phase=AgentPhase.ACTING,
        iteration=1,
        awaiting_human_approval=True,
        approval_request={
            "tool_name": "delete_file",
            "arguments": {"path": "/important.txt"},
            "reason": "Dangerous operation",
        },
    )

    checkpoint_tuple = MockCheckpointTuple({"state": state})

    checkpointer.get_tuple = MagicMock(return_value=checkpoint_tuple)
    checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

    return checkpointer


@pytest.fixture
def mock_checkpointer_empty():
    """Mock checkpointer that returns no checkpoint."""
    checkpointer = MagicMock()
    checkpointer.get_tuple = MagicMock(return_value=None)
    checkpointer.aget_tuple = AsyncMock(return_value=None)
    return checkpointer


@pytest.fixture
def mock_graph():
    """Mock the agent graph to return completed state."""
    async def mock_graph_fn(state):
        return state.set_phase(AgentPhase.COMPLETED)

    return mock_graph_fn


class TestResumeBasic:
    """Basic tests for resume functionality."""

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(
        self, mock_checkpointer_with_state, mock_graph
    ):
        """Test resuming from a valid checkpoint."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_with_state,
        )
        agent._graph = mock_graph

        session_id = uuid4()
        result = await agent.resume(session_id)

        assert result.phase == AgentPhase.COMPLETED
        # The async method is preferred when available
        mock_checkpointer_with_state.aget_tuple.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_no_checkpoint_raises(self, mock_checkpointer_empty):
        """Test that resuming with no checkpoint raises an error."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_empty,
        )

        session_id = uuid4()

        with pytest.raises(ValueError, match="No checkpoint found"):
            await agent.resume(session_id)

    @pytest.mark.asyncio
    async def test_resume_with_user_input(
        self, mock_checkpointer_awaiting_approval, mock_graph
    ):
        """Test resuming with additional user input."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_awaiting_approval,
        )
        agent._graph = mock_graph

        session_id = uuid4()
        result = await agent.resume(
            session_id,
            approval=True,
            user_input="Yes, proceed with the operation",
        )

        # Should have added the user input as a message
        user_messages = [m for m in result.messages if m.role == MessageRole.USER]
        assert any("proceed with the operation" in m.content for m in user_messages)


class TestResumeApproval:
    """Tests for human-in-the-loop approval during resume."""

    @pytest.mark.asyncio
    async def test_resume_with_approval(
        self, mock_checkpointer_awaiting_approval, mock_graph
    ):
        """Test resuming after approving a pending action."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_awaiting_approval,
        )
        agent._graph = mock_graph

        session_id = uuid4()
        result = await agent.resume(session_id, approval=True)

        # Should have cleared approval and completed
        assert result.awaiting_human_approval is False
        assert result.phase == AgentPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_with_rejection(self, mock_checkpointer_awaiting_approval):
        """Test resuming after rejecting a pending action."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_awaiting_approval,
        )

        session_id = uuid4()
        result = await agent.resume(session_id, approval=False)

        # Should have added error and be in error state
        assert result.phase == AgentPhase.ERROR
        assert "rejected by user" in result.errors[0].lower()


class TestStateReconstruction:
    """Tests for state reconstruction from checkpoint data."""

    def test_reconstruct_from_agent_state(self):
        """Test reconstruction when data is already AgentState."""
        agent = Agent(name="test")

        state = AgentState(task="Test")
        result = agent._reconstruct_state(state, uuid4())

        assert result is state

    def test_reconstruct_from_dict_with_state_key(self):
        """Test reconstruction from dict with 'state' key."""
        agent = Agent(name="test")

        state = AgentState(task="Test")
        state_data = {"state": state}

        result = agent._reconstruct_state(state_data, uuid4())

        assert result is state

    def test_reconstruct_from_raw_dict(self):
        """Test reconstruction from raw dict data."""
        agent = Agent(name="test")
        session_id = uuid4()

        state_data = {
            "task": "Test task",
            "iteration": 5,
        }

        result = agent._reconstruct_state(state_data, session_id)

        assert result is not None
        assert result.task == "Test task"
        assert result.session_id == session_id

    def test_reconstruct_invalid_data_returns_none(self):
        """Test that invalid data returns None."""
        agent = Agent(name="test")

        result = agent._reconstruct_state("not a dict or state", uuid4())

        assert result is None


class TestGetSessionState:
    """Tests for getting session state without resuming."""

    @pytest.mark.asyncio
    async def test_get_session_state_exists(self, mock_checkpointer_with_state):
        """Test getting state for an existing session."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_with_state,
        )

        session_id = uuid4()
        state = await agent.get_session_state(session_id)

        assert state is not None
        assert state.task == "Test task"

    @pytest.mark.asyncio
    async def test_get_session_state_not_exists(self, mock_checkpointer_empty):
        """Test getting state for non-existent session."""
        agent = Agent(
            name="test",
            checkpointer=mock_checkpointer_empty,
        )

        session_id = uuid4()
        state = await agent.get_session_state(session_id)

        assert state is None


class TestIsSessionPaused:
    """Tests for checking if session is paused."""

    def test_session_paused_when_awaiting_approval(self):
        """Test that session is paused when awaiting approval."""
        agent = Agent(name="test")

        state = AgentState(task="Test", awaiting_human_approval=True)

        assert agent.is_session_paused(state) is True

    def test_session_not_paused_normally(self):
        """Test that session is not paused in normal operation."""
        agent = Agent(name="test")

        state = AgentState(task="Test", awaiting_human_approval=False)

        assert agent.is_session_paused(state) is False


class TestCheckpointRetrieval:
    """Tests for checkpoint retrieval."""

    @pytest.mark.asyncio
    async def test_async_checkpointer(self):
        """Test retrieval with async checkpointer."""
        checkpointer = MagicMock()
        checkpoint_tuple = MockCheckpointTuple({"state": AgentState(task="Test")})
        checkpointer.aget_tuple = AsyncMock(return_value=checkpoint_tuple)

        agent = Agent(name="test", checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-123"}}
        result = await agent._get_checkpoint(config)

        assert result is checkpoint_tuple
        checkpointer.aget_tuple.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_sync_checkpointer_fallback(self):
        """Test fallback to sync checkpointer."""
        checkpointer = MagicMock()
        checkpoint_tuple = MockCheckpointTuple({"state": AgentState(task="Test")})

        # Only has sync method
        del checkpointer.aget_tuple
        checkpointer.get_tuple = MagicMock(return_value=checkpoint_tuple)

        agent = Agent(name="test", checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-123"}}
        result = await agent._get_checkpoint(config)

        assert result is checkpoint_tuple
        checkpointer.get_tuple.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_checkpointer_error_returns_none(self):
        """Test that checkpointer errors return None."""
        checkpointer = MagicMock()
        checkpointer.aget_tuple = AsyncMock(side_effect=Exception("DB error"))

        agent = Agent(name="test", checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-123"}}
        result = await agent._get_checkpoint(config)

        assert result is None
