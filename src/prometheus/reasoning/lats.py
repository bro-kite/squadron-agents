"""
LATS (Language Agent Tree Search) Reasoner

The main reasoning engine that combines MCTS with list-wise verification.
Implements System 2 thinking with branching and backtracking.
"""

import asyncio
from typing import Any, Callable, Generator, list, tuple
from uuid import UUID, uuid4

import structlog

from prometheus.core.config import LLMConfig, ReasoningConfig
from prometheus.core.state import AgentState, AgentPhase, ToolCall, ToolResult
from prometheus.memory.graphiti import GraphitiMemory
from prometheus.memory.types import MemoryResult
from prometheus.reasoning.mcts import MCTSController, MCTSNode
from prometheus.reasoning.verifier import CandidatePlan, ListWiseVerifier

logger = structlog.get_logger(__name__)


class ReasoningState:
    """State for reasoning operations."""

    def __init__(self, agent_state: AgentState):
        self.agent_state = agent_state
        self.pending_actions: list[ToolCall] = []
        self.current_plan: list[CandidatePlan] = []
        self.best_trajectory: list[MCTSNode] = []

    def add_action(self, action: ToolCall) -> None:
        """Add an action to the pending list."""
        self.pending_actions.append(action)

    def get_next_action(self) -> ToolCall | None:
        """Get and remove the next action."""
        if self.pending_actions:
            return self.pending_actions.pop(0)
        return None

    def has_actions(self) -> bool:
        """Check if there are pending actions."""
        return len(self.pending_actions) > 0


class LATSReasoner:
    """
    Language Agent Tree Search (LATS) Reasoner.
    
    Combines Monte Carlo Tree Search with list-wise verification
    to implement sophisticated System 2 thinking. When the agent
    faces a decision, it generates N candidate actions, evaluates
    them using MCTS, and selects the best path through list-wise
    comparison.
    
    Example:
        ```python
        reasoner = LATSReasoner(
            config=ReasoningConfig(n_candidates=5),
            llm_client=client,
        )
        
        # Plan for a task
        new_state = await reasoner.plan(initial_state)
        
        # Reflect on results
        reflected_state = await reasoner.reflect(new_state)
        ```
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        llm_config: LLMConfig | None = None,
        memory: GraphitiMemory | None = None,
        llm_client: Any | None = None,
    ):
        """
        Initialize the LATS reasoner.
        
        Args:
            config: Reasoning configuration
            llm_config: LLM configuration
            memory: Memory system for context
            llm_client: LLM client for reasoning
        """
        self.config = config or ReasoningConfig()
        self.llm_config = llm_config or LLMConfig()
        self.memory = memory
        self.llm_client = llm_client
        
        # Initialize components
        self.mcts = MCTSController(
            expand_fn=self._generate_candidate_actions,
            simulate_fn=self._simulate_action_outcome,
            exploration_constant=self.config.exploration_constant,
            max_depth=self.config.max_depth,
        )
        
        self.verifier = ListWiseVerifier(
            config=self.config,
            llm_client=self.llm_client,
        )
        
        # Reasoning templates
        self._planning_prompt = """You are a planning assistant. Given a task and relevant context, 
generate a plan to accomplish the task.

## Task
{task}

## Context
{context}

## Instructions
Break down the task into concrete steps. For each step, specify:
1. What needs to be done
2. What tools or actions are required
3. Expected outcome

Be specific and actionable."""

        self._action_prompt = """You need to choose the next action. Given the current task and context,
generate {n_candidates} different candidate approaches to proceed.

## Current Task
{task}

## Recent Context
{recent_messages}

## Available Actions
{available_actions}

## Instructions
Generate {n_candidates} distinct approaches. Each approach should be a different strategy for
moving forward. Be creative and consider alternative approaches.

Format each approach as:
Thought: [your thinking]
Action: [specific action to take]
Expected Outcome: [what you expect to happen]"""

        logger.info(
            "LATS Reasoner initialized",
            n_candidates=self.config.n_candidates,
            max_depth=self.config.max_depth,
        )

    async def plan(self, state: AgentState) -> AgentState:
        """
        Planning phase using LATS.
        
        Generates multiple candidate plans and selects the best one
        using tree search and list-wise verification.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with selected plan
        """
        logger.debug("Starting LATS planning phase", iteration=state.iteration)
        
        # Build context for planning
        context = await self._build_planning_context(state)
        
        # Generate candidate plans
        candidates = await self._generate_candidate_plans(state.task, context)
        
        if not candidates:
            logger.warning("No candidate plans generated")
            return state
        
        # Use list-wise verification to rank candidates
        ranked = await self.verifier.rank(
            task=state.task,
            candidates=candidates,
            context=context,
        )
        
        logger.info(
            "Plan ranking complete",
            num_candidates=len(candidates),
            best_plan_score=ranked[0].score if ranked else None,
        )
        
        # Store the best plan
        if ranked:
            best_plan = ranked[0]
            state = state.model_copy(
                update={
                    "phase": AgentPhase.ACTING,
                    "pending_tool_calls": self._plan_to_tool_calls(best_plan),
                }
            )
            
            # Add planning message
            from prometheus.core.state import Message, MessageRole
            planning_msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"Plan selected: {best_plan.thought}\nAction: {best_plan.action}\nExpected: {best_plan.expected_outcome}",
            )
            state = state.add_message(planning_msg)
        
        return state

    async def reflect(self, state: AgentState) -> AgentState:
        """
        Reflection phase using LATS.
        
        Analyzes recent results and decides whether to continue,
        backtrack, or complete the task.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reflection results
        """
        logger.debug("Starting LATS reflection phase", iteration=state.iteration)
        
        # Analyze recent results
        recent_results = state.tool_results[-3:] if state.tool_results else []
        successful_results = [r for r in recent_results if r.success]
        
        if not recent_results:
            # No results to reflect on
            return state
        
        # Check if we should continue or backtrack
        if not successful_results:
            # All recent actions failed - consider backtracking
            if state.iteration > 1:
                logger.info("All recent actions failed, considering backtrack")
                # For now, just continue - real backtracking would revert to previous plan
                # This would involve maintaining a plan history
        else:
            # Some actions succeeded - evaluate progress
            progress_score = len(successful_results) / len(recent_results)
            logger.debug("Progress evaluation", score=progress_score)
        
        # Generate reflection plans
        reflection_context = await self._build_reflection_context(state)
        candidates = await self._generate_reflection_plans(state.task, reflection_context)
        
        if candidates:
            # Rank reflection plans
            ranked = await self.verifier.rank(
                task=f"Reflect on: {state.task}",
                candidates=candidates,
                context=reflection_context,
            )
            
            # Apply the best reflection plan
            best_plan = ranked[0]
            
            from prometheus.core.state import Message, MessageRole
            reflection_msg = Message(
                role=MessageRole.ASSISTANT,
                content=f"Reflection: {best_plan.thought}\nAction: {best_plan.action}",
                metadata={"phase": "reflection"},
            )
            state = state.add_message(reflection_msg)
            
            # Execute reflection action if it's a tool call
            if "continue" in best_plan.action.lower():
                return state
            elif "complete" in best_plan.action.lower():
                return state.set_phase(AgentPhase.COMPLETED)
        
        return state

    async def _build_planning_context(self, state: AgentState) -> dict[str, Any]:
        """Build context for planning."""
        context = {
            "task": state.task,
            "iteration": state.iteration,
            "recent_messages": [msg.content for msg in state.messages[-5:]],
        }
        
        # Add memory context
        if state.memory_context:
            context["memory_facts"] = state.memory_context.get("facts", [])
            context["memory_entities"] = state.memory_context.get("entities", [])
        
        return context

    async def _build_reflection_context(self, state: AgentState) -> dict[str, Any]:
        """Build context for reflection."""
        recent_results = state.tool_results[-3:] if state.tool_results else []
        
        context = {
            "task": state.task,
            "iteration": state.iteration,
            "recent_results": [
                {
                    "tool": r.tool_name,
                    "success": r.success,
                    "result": str(r.result)[:100],
                    "error": r.error,
                }
                for r in recent_results
            ],
            "pending_actions": [tc.tool_name for tc in state.pending_tool_calls],
        }
        
        return context

    async def _generate_candidate_plans(
        self,
        task: str,
        context: dict[str, Any],
    ) -> list[CandidatePlan]:
        """Generate multiple candidate plans for a task."""
        try:
            if self.llm_client:
                return await self._generate_llm_plans(task, context)
            else:
                return self._generate_heuristic_plans(task, context)
        except Exception as e:
            logger.error("Plan generation failed", error=str(e))
            return self._generate_heuristic_plans(task, context)

    async def _generate_llm_plans(
        self,
        task: str,
        context: dict[str, Any],
    ) -> list[CandidatePlan]:
        """Generate plans using LLM."""
        prompt = self._planning_prompt.format(
            task=task,
            context="\n".join(f"{k}: {v}" for k, v in context.items()),
        )
        
        try:
            response = await self.llm_client.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse the response to extract plan steps
            # This is simplified - real implementation would parse structured output
            plans = self._parse_plan_response(content, task)
            return plans
            
        except Exception as e:
            logger.error("LLM plan generation failed", error=str(e))
            return self._generate_heuristic_plans(task, context)

    def _generate_heuristic_plans(
        self,
        task: str,
        context: dict[str, Any],
    ) -> list[CandidatePlan]:
        """Generate plans using heuristics when LLM is not available."""
        task_lower = task.lower()
        plans = []
        
        # Analyze task type and generate appropriate plans
        if "refactor" in task_lower:
            plans = [
                CandidatePlan(
                    id=uuid4(),
                    thought="Analyze the current codebase structure to understand dependencies",
                    action="List files and examine code structure",
                    expected_outcome="Understand of current architecture",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Create a backup of the original code before making changes",
                    action="Create a git branch and backup files",
                    expected_outcome="Safe rollback capability",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Implement the refactoring changes systematically",
                    action="Apply async/await patterns to identified modules",
                    expected_outcome="Refactored code with improved async patterns",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Test the refactored code to ensure it works correctly",
                    action="Run tests and validate functionality",
                    expected_outcome="Verified working refactored code",
                ),
            ]
        elif "analyze" in task_lower or "review" in task_lower:
            plans = [
                CandidatePlan(
                    id=uuid4(),
                    thought="Examine the codebase structure and key files",
                    action="List directory structure and identify main files",
                    expected_outcome="Overview of project structure",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Read and analyze the core implementation files",
                    action="Examine key source files for understanding",
                    expected_outcome="Deep understanding of implementation",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Generate a comprehensive analysis report",
                    action="Create analysis document with findings",
                    expected_outcome="Complete analysis report",
                ),
            ]
        else:
            # Generic plans for any task
            plans = [
                CandidatePlan(
                    id=uuid4(),
                    thought="Break down the task into smaller, manageable steps",
                    action="Identify key components and dependencies",
                    expected_outcome="Clear step-by-step approach",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Research relevant information and gather context",
                    action="Search for documentation and examples",
                    expected_outcome="Better understanding of requirements",
                ),
                CandidatePlan(
                    id=uuid4(),
                    thought="Execute the task systematically",
                    action="Implement the solution step by step",
                    expected_outcome="Task completed successfully",
                ),
            ]
        
        return plans[:self.config.n_candidates]

    def _parse_plan_response(self, content: str, task: str) -> list[CandidatePlan]:
        """Parse LLM response to extract plan steps."""
        # Simplified parsing - real implementation would use structured output
        plans = []
        
        # Split content into sections
        sections = content.split("\n\n")
        
        for i, section in enumerate(sections[:self.config.n_candidates]):
            lines = section.strip().split("\n")
            if len(lines) >= 2:
                thought = lines[0].strip()
                action = lines[1].strip()
                expected = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
                
                plans.append(CandidatePlan(
                    id=uuid4(),
                    thought=thought,
                    action=action,
                    expected_outcome=expected or "Expected outcome",
                ))
        
        return plans or self._generate_heuristic_plans(task, {})

    async def _generate_candidate_actions(
        self,
        state: ReasoningState,
    ) -> list[tuple[ToolCall, str, ReasoningState]]:
        """Generate candidate actions for MCTS expansion."""
        candidates = []
        
        # Generate action candidates based on current state
        if state.agent_state.pending_tool_calls:
            # Continue with existing actions
            for i, action in enumerate(state.agent_state.pending_tool_calls):
                candidates.append((action, f"Execute {action.tool_name}", state))
        else:
            # Generate new actions
            action_ideas = [
                ("Search files", "Search for relevant files in the codebase"),
                ("Read file", "Examine a specific file for information"),
                ("List directory", "Get overview of project structure"),
                ("Run command", "Execute a shell command"),
                ("Analyze code", "Analyze code structure and patterns"),
            ]
            
            for action_name, description in action_ideas:
                # Create a tool call (simplified)
                tool_call = ToolCall(
                    tool_name=action_name.lower().replace(" ", "_"),
                    arguments={"description": description},
                )
                new_state = ReasoningState(state.agent_state)
                new_state.add_action(tool_call)
                candidates.append((tool_call, description, new_state))
        
        return candidates[:self.config.n_candidates]

    async def _simulate_action_outcome(self, state: ReasoningState) -> float:
        """Simulate the outcome of an action."""
        # Simplified simulation
        # In real implementation, this would estimate success probability
        
        action_count = len(state.pending_actions)
        if action_count == 0:
            return 0.5  # Neutral outcome
        
        # Prefer actions that move toward completion
        if action_count <= 2:
            return 0.8  # Likely good
        elif action_count <= 5:
            return 0.6  # Reasonable
        else:
            return 0.4  # Too many actions, may be inefficient

    async def _generate_reflection_plans(
        self,
        task: str,
        context: dict[str, Any],
    ) -> list[CandidatePlan]:
        """Generate reflection plans."""
        plans = [
            CandidatePlan(
                id=uuid4(),
                thought="Continue with the current approach as progress is being made",
                action="Continue executing the plan",
                expected_outcome="Continue making progress on the task",
            ),
            CandidatePlan(
                id=uuid4(),
                thought="The current approach isn't working, try an alternative strategy",
                action="Generate a new plan with different approach",
                expected_outcome="Improved progress with alternative strategy",
            ),
            CandidatePlan(
                id=uuid4(),
                thought="The task appears to be complete based on the results",
                action="Mark task as complete and summarize results",
                expected_outcome="Task successfully completed",
            ),
        ]
        
        return plans

    def _plan_to_tool_calls(self, plan: CandidatePlan) -> list[ToolCall]:
        """Convert a candidate plan to tool calls."""
        # Simplified conversion
        # In real implementation, this would parse the plan structure
        
        tool_call = ToolCall(
            tool_name="execute_plan",
            arguments={
                "thought": plan.thought,
                "action": plan.action,
                "expected_outcome": plan.expected_outcome,
            }
        )
        
        return [tool_call]

    @property
    def stats(self) -> dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "mcts_nodes": len(self.mcts._nodes),
            "mcts_stats": self.mcts.tree_stats,
        }
