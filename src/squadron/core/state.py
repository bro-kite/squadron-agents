"""
Agent State Management

Defines the core state structures used throughout the agent lifecycle.
State is immutable and passed through the LangGraph execution graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation history."""

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    name: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class ToolCall(BaseModel):
    """A tool invocation request from the agent."""

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    arguments: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_call_id: UUID
    tool_name: str
    result: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success(self) -> bool:
        """Check if the tool execution was successful."""
        return self.error is None

    class Config:
        frozen = True


class ThoughtNode(BaseModel):
    """
    A node in the reasoning tree (LATS).
    
    Represents a single thought/action pair with associated value estimates.
    """

    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID | None = None
    thought: str
    action: ToolCall | None = None
    value: float = 0.0
    visits: int = 0
    children_ids: list[UUID] = Field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class AgentPhase(str, Enum):
    """Current phase in the agent's cognitive loop."""

    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentState(BaseModel):
    """
    Complete state of an agent at any point in execution.
    
    This is the primary state object passed through the LangGraph execution graph.
    It is designed to be immutable - all updates create new state instances.
    """

    # Identity
    agent_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)
    
    # Current phase
    phase: AgentPhase = AgentPhase.PLANNING
    
    # Conversation history
    messages: tuple[Message, ...] = Field(default_factory=tuple)
    
    # Current task
    task: str = ""
    task_id: UUID | None = None
    
    # Tool interactions
    pending_tool_calls: tuple[ToolCall, ...] = Field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = Field(default_factory=tuple)
    
    # Reasoning tree (LATS)
    thought_tree: dict[str, ThoughtNode] = Field(default_factory=dict)
    current_thought_id: UUID | None = None
    best_trajectory: list[UUID] = Field(default_factory=list)
    
    # Memory references
    memory_context: dict[str, Any] = Field(default_factory=dict)
    retrieved_facts: list[dict[str, Any]] = Field(default_factory=list)
    
    # Execution metadata
    iteration: int = 0
    max_iterations: int = 50
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_update: datetime = Field(default_factory=datetime.utcnow)
    
    # Error handling
    errors: list[str] = Field(default_factory=list)
    
    # Human-in-the-loop
    awaiting_human_approval: bool = False
    approval_request: dict[str, Any] | None = None

    class Config:
        frozen = True

    def add_message(self, message: Message) -> "AgentState":
        """Add a message to the conversation history."""
        return self.model_copy(
            update={
                "messages": (*self.messages, message),
                "last_update": datetime.utcnow(),
            }
        )

    def add_tool_call(self, tool_call: ToolCall) -> "AgentState":
        """Add a pending tool call."""
        return self.model_copy(
            update={
                "pending_tool_calls": (*self.pending_tool_calls, tool_call),
                "last_update": datetime.utcnow(),
            }
        )

    def add_tool_result(self, result: ToolResult) -> "AgentState":
        """Add a tool result and remove from pending."""
        pending = tuple(
            tc for tc in self.pending_tool_calls if tc.id != result.tool_call_id
        )
        return self.model_copy(
            update={
                "pending_tool_calls": pending,
                "tool_results": (*self.tool_results, result),
                "last_update": datetime.utcnow(),
            }
        )

    def set_phase(self, phase: AgentPhase) -> "AgentState":
        """Transition to a new phase."""
        return self.model_copy(
            update={
                "phase": phase,
                "last_update": datetime.utcnow(),
            }
        )

    def increment_iteration(self) -> "AgentState":
        """Increment the iteration counter."""
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "last_update": datetime.utcnow(),
            }
        )

    def add_error(self, error: str) -> "AgentState":
        """Record an error."""
        return self.model_copy(
            update={
                "errors": [*self.errors, error],
                "phase": AgentPhase.ERROR,
                "last_update": datetime.utcnow(),
            }
        )

    def request_approval(self, request: dict[str, Any]) -> "AgentState":
        """Request human approval for an action."""
        return self.model_copy(
            update={
                "awaiting_human_approval": True,
                "approval_request": request,
                "last_update": datetime.utcnow(),
            }
        )

    def grant_approval(self) -> "AgentState":
        """Grant approval and continue execution."""
        return self.model_copy(
            update={
                "awaiting_human_approval": False,
                "approval_request": None,
                "last_update": datetime.utcnow(),
            }
        )

    def add_thought_node(self, node: ThoughtNode) -> "AgentState":
        """Add a node to the reasoning tree."""
        tree = dict(self.thought_tree)
        tree[str(node.id)] = node
        return self.model_copy(
            update={
                "thought_tree": tree,
                "current_thought_id": node.id,
                "last_update": datetime.utcnow(),
            }
        )

    def update_memory_context(self, context: dict[str, Any]) -> "AgentState":
        """Update memory context from Graphiti."""
        return self.model_copy(
            update={
                "memory_context": {**self.memory_context, **context},
                "last_update": datetime.utcnow(),
            }
        )

    @property
    def is_complete(self) -> bool:
        """Check if the agent has completed its task."""
        return self.phase == AgentPhase.COMPLETED

    @property
    def has_error(self) -> bool:
        """Check if the agent encountered an error."""
        return self.phase == AgentPhase.ERROR or len(self.errors) > 0

    @property
    def should_continue(self) -> bool:
        """Check if the agent should continue execution."""
        return (
            not self.is_complete
            and not self.has_error
            and self.iteration < self.max_iterations
            and not self.awaiting_human_approval
        )