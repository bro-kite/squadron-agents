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

# Import skills (optional dependency)
try:
    from squadron.skills.manager import SkillsManager
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    SkillsManager = None

logger = structlog.get_logger(__name__)


# Prompt template for LLM-based action generation (with skills support)
ACTION_GENERATION_PROMPT = '''You are an expert AI agent planning assistant. Your task is to generate candidate actions for accomplishing the given task.

## Current Task
{task}

## Available Tools
{tools}
{skills_section}
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
        skills_manager: Any | None = None,
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
            skills_manager: SkillsManager for Agent Skills support
        """
        self.config = config or ReasoningConfig()
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        self.verifier = verifier or ListWiseVerifier(config=self.config)
        self.default_tool = default_tool
        # Function to derive tool arguments from state; falls back to {"text": task}
        self.tool_args_fn = tool_args_fn
        # Agent Skills support
        self.skills_manager = skills_manager

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
        Includes relevant Agent Skills context when available.
        """
        # Format available tools for the prompt
        tools_text = self._format_tools_for_prompt()

        # Format conversation history
        history_text = self._format_history(state)

        # Format memory context
        context_text = self._format_context(state.memory_context)

        # Format skills context (if available)
        skills_text = self._format_skills_for_prompt(state.task)

        # Build the prompt
        prompt = ACTION_GENERATION_PROMPT.format(
            task=state.task,
            tools=tools_text,
            skills_section=skills_text,
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

    def _format_skills_for_prompt(self, task: str) -> str:
        """
        Format relevant Agent Skills for inclusion in the prompt.
        
        Uses progressive disclosure - only includes skills relevant to the task.
        """
        if not self.skills_manager:
            return ""
        
        try:
            # Find skills relevant to the task
            matches = self.skills_manager.find_skills(task, threshold=0.2)
            
            if not matches:
                return ""
            
            lines = ["\n## Relevant Skills"]
            lines.append("The following skills may help with this task:\n")
            
            for match in matches:
                skill = match.skill
                lines.append(f"### {skill.name}")
                lines.append(f"**Description**: {skill.description}")
                
                # Include instructions if skill is loaded
                if skill.is_loaded and skill.instructions:
                    # Truncate long instructions
                    instructions = skill.instructions[:1000]
                    if len(skill.instructions) > 1000:
                        instructions += "\n... (truncated)"
                    lines.append(f"\n**Instructions**:\n{instructions}")
                
                lines.append("")
            
            logger.debug(
                "Added skills context to prompt",
                num_skills=len(matches),
                skills=[m.skill.name for m in matches],
            )
            
            return "\n".join(lines) + "\n"
            
        except Exception as e:
            logger.warning("Failed to format skills for prompt", error=str(e))
            return ""

    def set_skills_manager(self, skills_manager: Any) -> None:
        """Set the Skills Manager for Agent Skills support."""
        self.skills_manager = skills_manager
        logger.info(
            "Set SkillsManager for LATS",
            skill_count=skills_manager.skill_count if skills_manager else 0,
        )

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
        # Reinitialize MCTS with real functions now that we have an LLM
        self._init_mcts_with_llm()
        logger.info("Set LLM provider for LATS", provider=llm.provider_name)

    def _init_mcts_with_llm(self) -> None:
        """Initialize MCTS controller with LLM-based expand and simulate functions."""
        self.mcts = MCTSController(
            expand_fn=self._mcts_expand,
            simulate_fn=self._mcts_simulate,
            is_terminal_fn=self._mcts_is_terminal,
            exploration_constant=self.config.exploration_constant,
            max_depth=self.config.max_depth,
        )

    # ------------------------------------------------------------------
    # MCTS Integration
    # ------------------------------------------------------------------
    async def plan_with_mcts(
        self,
        state: AgentState,
        budget: int | None = None,
    ) -> AgentState:
        """
        Plan using full MCTS tree search.

        This method uses Monte Carlo Tree Search to explore multiple
        action trajectories and select the best one.

        Args:
            state: Current agent state
            budget: Number of MCTS simulations (defaults to config.simulation_budget)

        Returns:
            Updated agent state with chosen action
        """
        if not self.llm:
            logger.warning("MCTS planning requires LLM, falling back to simple plan")
            return await self.plan(state)

        budget = budget or self.config.simulation_budget

        logger.info("Starting MCTS planning", budget=budget)

        # Run MCTS search
        best_action, trajectory = await self.mcts.search(
            initial_state=state,
            budget=budget,
        )

        if best_action is None:
            logger.warning("MCTS found no valid actions, falling back to simple plan")
            return await self.plan(state)

        # Extract the tool call from the best action
        tool_call = ToolCall(
            id=uuid4(),
            tool_name=best_action["tool_name"],
            arguments=best_action.get("arguments", {}),
        )

        # Build planning message with trajectory info
        trajectory_desc = " -> ".join(
            node.action_description for node in trajectory if node.action_description
        )

        plan_msg = Message(
            role=MessageRole.ASSISTANT,
            content=f"MCTS planning selected: {tool_call.tool_name}\nTrajectory: {trajectory_desc}",
            metadata={
                "phase": "planning",
                "mcts_budget": budget,
                "trajectory_length": len(trajectory),
                "tree_stats": self.mcts.tree_stats,
            },
        )

        state = state.add_message(plan_msg)
        state = state.add_tool_call(tool_call)

        logger.info(
            "MCTS planning complete",
            action=tool_call.tool_name,
            tree_stats=self.mcts.tree_stats,
        )

        return state

    def _mcts_expand(self, state: AgentState) -> list[tuple[Any, str, AgentState]]:
        """
        MCTS expansion function: Generate possible actions from a state.

        This is called synchronously by MCTS, so we use a synchronous
        approach to generate candidates.

        Returns:
            List of (action_dict, description, new_state) tuples
        """
        import asyncio

        if not self.llm or not self.tools:
            return []

        try:
            # Run async candidate generation synchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._generate_mcts_expansions(state),
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self._generate_mcts_expansions(state)
                )
        except Exception as e:
            logger.error("MCTS expansion failed", error=str(e))
            return []

    async def _generate_mcts_expansions(
        self, state: AgentState
    ) -> list[tuple[Any, str, AgentState]]:
        """Generate expansion candidates for MCTS."""
        candidates = await self._generate_candidate_calls_llm(state)

        expansions = []
        for candidate in candidates:
            # Create action dict
            action = {
                "tool_name": candidate.action,
                "arguments": candidate.context.get("arguments", {}),
                "thought": candidate.thought,
            }

            # Create a hypothetical next state (without actually executing)
            # This represents what we expect the state to look like
            tool_call = ToolCall(
                id=uuid4(),
                tool_name=candidate.action,
                arguments=candidate.context.get("arguments", {}),
            )

            new_state = state.add_tool_call(tool_call)
            new_state = new_state.add_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Hypothetical: {candidate.expected_outcome}",
                    metadata={"mcts_hypothetical": True},
                )
            )

            description = f"{candidate.action}: {candidate.thought[:50]}"
            expansions.append((action, description, new_state))

        return expansions

    def _mcts_simulate(self, state: AgentState) -> float:
        """
        MCTS simulation function: Evaluate the quality of a state.

        Returns a value in [0, 1] representing how promising this state is.
        """
        import asyncio

        if not self.llm:
            return self._heuristic_state_value(state)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, use heuristic
                return self._heuristic_state_value(state)
            else:
                return loop.run_until_complete(
                    self._llm_state_value(state)
                )
        except Exception as e:
            logger.error("MCTS simulation failed", error=str(e))
            return self._heuristic_state_value(state)

    async def _llm_state_value(self, state: AgentState) -> float:
        """Use LLM to evaluate state quality."""
        prompt = f"""Evaluate the following agent state on a scale of 0.0 to 1.0.

Task: {state.task}

Current Progress:
- Messages: {len(state.messages)}
- Tool calls executed: {len(state.tool_results)}
- Successful tools: {sum(1 for r in state.tool_results if r.success)}
- Current phase: {state.phase}

Recent activity:
{self._format_history(state)}

How close is this state to completing the task successfully?
Respond with ONLY a number between 0.0 and 1.0."""

        try:
            messages = [LLMMessage.user(prompt)]
            response = await self.llm.generate(messages)

            # Parse the value
            content = response.content.strip()
            # Extract first number found
            import re
            match = re.search(r"([0-9]*\.?[0-9]+)", content)
            if match:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))

            return 0.5  # Default to middle value

        except Exception as e:
            logger.error("LLM state evaluation failed", error=str(e))
            return self._heuristic_state_value(state)

    def _heuristic_state_value(self, state: AgentState) -> float:
        """
        Heuristic state evaluation when LLM is unavailable.

        Scores based on:
        - Progress (number of successful tool calls)
        - Recency of success
        - Absence of errors
        """
        value = 0.0

        # Base value for having tool results
        if state.tool_results:
            value += 0.2

        # Bonus for successful tool calls
        successful = sum(1 for r in state.tool_results if r.success)
        total = len(state.tool_results)
        if total > 0:
            success_rate = successful / total
            value += 0.3 * success_rate

        # Bonus for recent success
        if state.tool_results and state.tool_results[-1].success:
            value += 0.2

        # Penalty for errors
        if state.errors:
            value -= 0.3

        # Bonus for completion indicators in messages
        completion_keywords = ["complete", "done", "finished", "success"]
        for msg in state.messages[-3:]:
            if any(kw in msg.content.lower() for kw in completion_keywords):
                value += 0.2
                break

        return max(0.0, min(1.0, value))

    def _mcts_is_terminal(self, state: AgentState) -> bool:
        """Check if a state is terminal (task complete or failed)."""
        # Terminal if completed or errored
        if state.phase.value in ("completed", "error"):
            return True

        # Terminal if max iterations reached
        if state.iteration >= state.max_iterations:
            return True

        # Terminal if we have completion indicators
        completion_keywords = ["complete", "done", "finished", "success"]
        for msg in state.messages[-2:]:
            if any(kw in msg.content.lower() for kw in completion_keywords):
                return True

        return False

    # Legacy stub methods (kept for backward compatibility)
    def _expand_stub(self, state: Any) -> list:
        """Stub expand function for when LLM is not available."""
        return []

    def _simulate_stub(self, state: Any) -> float:
        """Stub simulate function for when LLM is not available."""
        return 0.0
