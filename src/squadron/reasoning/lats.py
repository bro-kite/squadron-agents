"""
Language Agent Tree Search (LATS) Reasoner

This is a lightweight, working-first implementation that sits on top of the
generic `MCTSController`. It focuses on producing executable tool calls so the
core Agent loop can make forward progress even without a domain‑specific
planner. The design keeps the surface area small while we iterate toward a
full MCTS-based rollout policy and rollout simulator.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable
from uuid import uuid4

import structlog

from squadron.core.state import AgentState, Message, MessageRole, ToolCall
from squadron.reasoning.mcts import MCTSController
from squadron.reasoning.verifier import ListWiseVerifier
from squadron.core.config import ReasoningConfig

logger = structlog.get_logger(__name__)


class LATSReasoner:
    """
    Minimal LATS reasoner.

    Responsibilities (initial version):
    - Generate at least one executable ToolCall so the agent doesn't stall.
    - Optionally rank multiple candidate actions using ListWiseVerifier.
    - Update AgentState with chosen action and trace messages.

    The full Monte‑Carlo tree search/rollout loop can be layered on later by
    expanding the `expand_fn` and `simulate_fn` passed into `MCTSController`.
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        memory: Any | None = None,
        verifier: ListWiseVerifier | None = None,
        default_tool: str | None = None,
        tool_args_fn: Callable[[AgentState], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or ReasoningConfig()
        self.memory = memory
        self.verifier = verifier or ListWiseVerifier(config=self.config)
        self.default_tool = default_tool
        # Function to derive tool arguments from state; falls back to {"text": task}
        self.tool_args_fn = tool_args_fn

        # Placeholder MCTS controller – ready for richer policies later
        self.mcts = MCTSController(
            expand_fn=self._expand_stub,
            simulate_fn=self._simulate_stub,
            exploration_constant=self.config.exploration_constant,
            max_depth=self.config.max_depth,
        )

    # ------------------------------------------------------------------
    # Public API used by Agent
    # ------------------------------------------------------------------
    async def plan(self, state: AgentState) -> AgentState:
        """Produce one or more candidate ToolCalls and pick the best."""
        logger.debug("LATS plan start", iteration=state.iteration)

        candidates = list(self._generate_candidate_calls(state))

        if not candidates:
            # Nothing to do – emit a planning message so the loop can reflect
            msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"No tools available for task: {state.task}",
                metadata={"phase": "planning"},
            )
            return state.add_message(msg)

        # If we have more than one candidate, rank them list-wise
        if len(candidates) > 1:
            ranked = await self.verifier.rank(
                task=state.task,
                candidates=candidates,
                context=state.memory_context,
            )
            chosen = ranked[0]
        else:
            chosen = candidates[0]

        tool_call = ToolCall(
            id=uuid4(),
            tool_name=chosen.action,
            arguments=chosen.context.get("arguments", {}),
        )

        plan_msg = Message(
            role=MessageRole.ASSISTANT,
            content=f"Planning to call tool '{tool_call.tool_name}'",
            metadata={"phase": "planning"},
        )

        state = state.add_message(plan_msg)
        state = state.add_tool_call(tool_call)
        return state

    async def reflect(self, state: AgentState) -> AgentState:
        """Lightweight reflection: mark completion if last tool succeeded."""
        if state.tool_results:
            last_result = state.tool_results[-1]
            status = "succeeded" if last_result.success else "failed"
            msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"Tool '{last_result.tool_name}' {status}.",
                metadata={"phase": "reflection"},
            )
            state = state.add_message(msg)
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_candidate_calls(self, state: AgentState) -> Iterable[CandidatePlan]:
        """
        Generate naive candidate tool calls. Currently uses a single default tool
        and passes the task text as `text` unless a custom arg fn is provided.
        """
        if not self.default_tool:
            return []

        args = self.tool_args_fn(state) if self.tool_args_fn else {"text": state.task}

        from squadron.reasoning.verifier import CandidatePlan  # lazy import to avoid cycle

        yield CandidatePlan(
            id=uuid4(),
            thought=f"Use tool {self.default_tool} to progress the task",
            action=self.default_tool,
            expected_outcome="Progress toward task completion",
            context={"arguments": args},
        )

    # MCTS placeholders (to be expanded with real rollout logic)
    def _expand_stub(self, state: Any):
        return []

    def _simulate_stub(self, state: Any) -> float:
        return 0.0
