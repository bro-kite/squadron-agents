"""
Language Agent Tree Search (LATS) Reasoner

This is a lightweight, working-first implementation that sits on top of the
generic `MCTSController`. It focuses on producing executable tool calls so the
core Agent loop can make forward progress even without a domain‑specific
planner. The design keeps the surface area small while we iterate toward a
full MCTS-based rollout policy and rollout simulator.

Enhanced with LLM-based action generation for intelligent candidate creation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable
from uuid import uuid4

import structlog

from squadron.core.state import AgentState, Message, MessageRole, ToolCall
from squadron.reasoning.mcts import MCTSController
from squadron.reasoning.verifier import ListWiseVerifier, CandidatePlan
from squadron.core.config import ReasoningConfig
from squadron.llm.base import LLMProvider, LLMMessage, ToolDefinition

logger = structlog.get_logger(__name__)


# Prompt template for LLM-based action generation
ACTION_GENERATION_PROMPT = '''You are an expert AI agent planning assistant. Your task is to generate candidate actions for accomplishing the given task.

## Current Task
{task}

## Available Tools
{tools}

## Conversation History
{history}

## Memory Context
{context}

## Instructions
Generate {n_candidates} diverse candidate actions that could help accomplish the task. Each candidate should use one of the available tools.

For each candidate, provide:
1. thought: Your reasoning for why this action would be helpful
2. tool_name: The name of the tool to use (must match an available tool exactly)
3. arguments: The arguments to pass to the tool as a JSON object
4. expected_outcome: What you expect to happen if this action succeeds

Consider:
- What information do we need to gather?
- What actions would make the most progress toward the goal?
- What are different approaches to solving this problem?
- What could go wrong and how might we handle it?

## Output Format
Respond with a JSON array of candidates:
```json
[
  {{
    "thought": "reasoning for this action",
    "tool_name": "tool_name_here",
    "arguments": {{"arg1": "value1"}},
    "expected_outcome": "what should happen"
  }},
  ...
]
```

Generate exactly {n_candidates} candidates:'''


class LATSReasoner:
    """
    LATS reasoner with LLM-based action generation.

    Responsibilities:
    - Generate multiple candidate ToolCalls using LLM reasoning.
    - Rank candidates using ListWiseVerifier.
    - Select the best action for execution.
    - Update AgentState with chosen action and trace messages.

    The full Monte‑Carlo tree search/rollout loop can be layered on later by
    expanding the `expand_fn` and `simulate_fn` passed into `MCTSController`.
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        llm: LLMProvider | None = None,
        tools: list[ToolDefinition] | None = None,
        memory: Any | None = None,
        verifier: ListWiseVerifier | None = None,
        default_tool: str | None = None,
        tool_args_fn: Callable[[AgentState], dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the LATS reasoner.

        Args:
            config: Reasoning configuration
            llm: LLM provider for generating candidate actions
            tools: Available tool definitions for the agent
            memory: Memory system for context retrieval
            verifier: ListWiseVerifier for ranking candidates
            default_tool: Fallback tool when LLM is unavailable
            tool_args_fn: Function to derive tool arguments from state
        """
        self.config = config or ReasoningConfig()
        self.llm = llm
        self.tools = tools or []
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

        # Try LLM-based generation first, fall back to simple generation
        if self.llm and self.tools:
            candidates = await self._generate_candidate_calls_llm(state)
        else:
            candidates = list(self._generate_candidate_calls(state))

        if not candidates:
            # Nothing to do – emit a planning message so the loop can reflect
            msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"No tools available for task: {state.task}",
                metadata={"phase": "planning"},
            )
            return state.add_message(msg)

        logger.info(
            "Generated candidate actions",
            num_candidates=len(candidates),
            tools=[c.action for c in candidates],
        )

        # If we have more than one candidate, rank them list-wise
        if len(candidates) > 1:
            ranked = await self.verifier.rank(
                task=state.task,
                candidates=candidates,
                context=state.memory_context,
            )
            chosen = ranked[0]
            logger.info(
                "Selected best action",
                tool=chosen.action,
                score=chosen.score,
                reasoning=chosen.reasoning[:100] if chosen.reasoning else None,
            )
        else:
            chosen = candidates[0]

        tool_call = ToolCall(
            id=uuid4(),
            tool_name=chosen.action,
            arguments=chosen.context.get("arguments", {}),
        )

        plan_msg = Message(
            role=MessageRole.ASSISTANT,
            content=f"Planning to call tool '{tool_call.tool_name}': {chosen.thought}",
            metadata={
                "phase": "planning",
                "thought": chosen.thought,
                "expected_outcome": chosen.expected_outcome,
            },
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
    async def _generate_candidate_calls_llm(
        self, state: AgentState
    ) -> list[CandidatePlan]:
        """
        Generate candidate tool calls using the LLM.

        Uses the ACTION_GENERATION_PROMPT to ask the LLM to propose
        multiple candidate actions based on the current state and available tools.
        """
        # Format available tools for the prompt
        tools_text = self._format_tools_for_prompt()

        # Format conversation history
        history_text = self._format_history(state)

        # Format memory context
        context_text = self._format_context(state.memory_context)

        # Build the prompt
        prompt = ACTION_GENERATION_PROMPT.format(
            task=state.task,
            tools=tools_text,
            history=history_text,
            context=context_text,
            n_candidates=self.config.n_candidates,
        )

        logger.debug(
            "Generating candidates with LLM",
            n_candidates=self.config.n_candidates,
            num_tools=len(self.tools),
        )

        try:
            # Call the LLM
            messages = [LLMMessage.user(prompt)]
            response = await self.llm.generate(messages)

            # Parse the response
            candidates = self._parse_llm_candidates(response.content)

            if candidates:
                logger.debug(
                    "LLM generated candidates",
                    num_candidates=len(candidates),
                )
                return candidates

            # Fallback to simple generation if parsing fails
            logger.warning("Failed to parse LLM candidates, using fallback")
            return list(self._generate_candidate_calls(state))

        except Exception as e:
            logger.error("LLM candidate generation failed", error=str(e))
            # Fallback to simple generation
            return list(self._generate_candidate_calls(state))

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in the prompt."""
        if not self.tools:
            return "No tools available."

        lines = []
        for tool in self.tools:
            params_str = json.dumps(tool.parameters, indent=2)
            lines.append(
                f"### {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n```json\n{params_str}\n```\n"
            )
        return "\n".join(lines)

    def _format_history(self, state: AgentState) -> str:
        """Format conversation history for the prompt."""
        if not state.messages:
            return "No previous conversation."

        lines = []
        # Include last 10 messages to avoid context overflow
        recent_messages = state.messages[-10:]
        for msg in recent_messages:
            role = msg.role.value.upper()
            content = msg.content[:500]  # Truncate long messages
            if len(msg.content) > 500:
                content += "..."
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format memory context for the prompt."""
        if not context:
            return "No additional context."

        lines = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2)[:200]
            else:
                value_str = str(value)[:200]
            lines.append(f"- {key}: {value_str}")

        return "\n".join(lines)

    def _parse_llm_candidates(self, content: str) -> list[CandidatePlan]:
        """Parse LLM response into CandidatePlan objects."""
        candidates = []

        # Extract JSON array from response
        try:
            # Find JSON array in the response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1

            if json_start < 0 or json_end <= json_start:
                logger.warning("No JSON array found in LLM response")
                return []

            json_str = content[json_start:json_end]
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                logger.warning("LLM response is not a list")
                return []

            # Build valid tool names set for validation
            valid_tools = {tool.name for tool in self.tools}

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                tool_name = item.get("tool_name", "")
                thought = item.get("thought", "")
                arguments = item.get("arguments", {})
                expected_outcome = item.get("expected_outcome", "")

                # Validate tool name
                if tool_name not in valid_tools:
                    logger.warning(
                        "LLM suggested invalid tool",
                        tool=tool_name,
                        valid_tools=list(valid_tools),
                    )
                    continue

                # Ensure arguments is a dict
                if not isinstance(arguments, dict):
                    arguments = {}

                candidates.append(
                    CandidatePlan(
                        id=uuid4(),
                        thought=thought or f"Use {tool_name}",
                        action=tool_name,
                        expected_outcome=expected_outcome or "Progress on task",
                        context={"arguments": arguments},
                    )
                )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response", error=str(e))
            return []

        return candidates

    def _generate_candidate_calls(self, state: AgentState) -> Iterable[CandidatePlan]:
        """
        Generate naive candidate tool calls. Currently uses a single default tool
        and passes the task text as `text` unless a custom arg fn is provided.

        This is the fallback when LLM-based generation is not available.
        """
        if not self.default_tool:
            return []

        args = self.tool_args_fn(state) if self.tool_args_fn else {"text": state.task}

        yield CandidatePlan(
            id=uuid4(),
            thought=f"Use tool {self.default_tool} to progress the task",
            action=self.default_tool,
            expected_outcome="Progress toward task completion",
            context={"arguments": args},
        )

    def register_tools(self, tools: list[ToolDefinition]) -> None:
        """Register available tools for action generation."""
        self.tools = tools
        logger.info("Registered tools for LATS", num_tools=len(tools))

    def set_llm(self, llm: LLMProvider) -> None:
        """Set the LLM provider for action generation."""
        self.llm = llm
        logger.info("Set LLM provider for LATS", provider=llm.provider_name)

    # MCTS placeholders (to be expanded with real rollout logic)
    def _expand_stub(self, state: Any):
        return []

    def _simulate_stub(self, state: Any) -> float:
        return 0.0
