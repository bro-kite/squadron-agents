"""
Agent Evaluation System

Provides comprehensive evaluation for agent behavior using DeepEval-compatible
metrics. Supports both single-turn and multi-turn (agentic) evaluation.

Key features:
- Tool usage verification
- Trajectory evaluation
- Custom metrics via GEval
- Regression testing for self-improvement
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

import structlog

from squadron.core.state import AgentState, Message, ToolResult

logger = structlog.get_logger(__name__)


class EvalMetric(str, Enum):
    """Built-in evaluation metrics."""
    
    # Task completion
    TASK_COMPLETION = "task_completion"
    GOAL_ACHIEVED = "goal_achieved"
    
    # Tool usage
    TOOL_CORRECTNESS = "tool_correctness"
    TOOL_EFFICIENCY = "tool_efficiency"
    
    # Reasoning quality
    REASONING_COHERENCE = "reasoning_coherence"
    PLAN_QUALITY = "plan_quality"
    
    # Safety
    SAFETY_COMPLIANCE = "safety_compliance"
    GUARDRAIL_ADHERENCE = "guardrail_adherence"
    
    # Efficiency
    ITERATION_EFFICIENCY = "iteration_efficiency"
    TOKEN_EFFICIENCY = "token_efficiency"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class EvalResult:
    """Result of an evaluation."""
    
    metric: EvalMetric | str
    score: float  # 0.0 to 1.0
    passed: bool
    
    # Details
    reasoning: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value if isinstance(self.metric, EvalMetric) else self.metric,
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "details": self.details,
            "evaluatedAt": self.evaluated_at.isoformat(),
        }


@dataclass
class TestCase:
    """
    A test case for agent evaluation.
    
    Defines expected behavior for a given input.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # Input
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    
    # Expected behavior
    expected_tools: list[str] = field(default_factory=list)
    expected_tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_output_contains: list[str] = field(default_factory=list)
    expected_output_not_contains: list[str] = field(default_factory=list)
    
    # Thresholds
    min_score: float = 0.7
    max_iterations: int = 10
    
    # Tags for filtering
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "task": self.task,
            "context": self.context,
            "expectedTools": self.expected_tools,
            "expectedToolArgs": self.expected_tool_args,
            "expectedOutputContains": self.expected_output_contains,
            "expectedOutputNotContains": self.expected_output_not_contains,
            "minScore": self.min_score,
            "maxIterations": self.max_iterations,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            name=data.get("name", ""),
            description=data.get("description", ""),
            task=data.get("task", ""),
            context=data.get("context", {}),
            expected_tools=data.get("expectedTools", []),
            expected_tool_args=data.get("expectedToolArgs", {}),
            expected_output_contains=data.get("expectedOutputContains", []),
            expected_output_not_contains=data.get("expectedOutputNotContains", []),
            min_score=data.get("minScore", 0.7),
            max_iterations=data.get("maxIterations", 10),
            tags=data.get("tags", []),
        )


@dataclass
class EvalSuiteResult:
    """Result of running an evaluation suite."""
    
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Individual results
    results: list[tuple[TestCase, list[EvalResult]]] = field(default_factory=list)
    
    # Aggregate scores
    avg_score: float = 0.0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed_tests == 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suiteName": self.suite_name,
            "totalTests": self.total_tests,
            "passedTests": self.passed_tests,
            "failedTests": self.failed_tests,
            "passRate": self.pass_rate,
            "avgScore": self.avg_score,
            "startedAt": self.started_at.isoformat(),
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "durationSeconds": self.duration_seconds,
            "results": [
                {
                    "testCase": tc.to_dict(),
                    "evalResults": [r.to_dict() for r in results],
                }
                for tc, results in self.results
            ],
        }


class AgentEvaluator:
    """
    Agent Evaluator - Comprehensive evaluation for agent behavior.
    
    Integrates with DeepEval for advanced metrics while providing
    built-in evaluators for common agent patterns.
    
    Example:
        ```python
        evaluator = AgentEvaluator()
        
        # Evaluate a completed agent run
        results = await evaluator.evaluate(
            state=agent_state,
            test_case=TestCase(
                task="Refactor the auth module",
                expected_tools=["read_file", "edit_file"],
            ),
        )
        
        # Run a full test suite
        suite_result = await evaluator.run_suite(
            agent=agent,
            test_cases=test_cases,
        )
        ```
    """
    
    def __init__(
        self,
        llm_client: Any | None = None,
        use_deepeval: bool = True,
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_client: LLM client for LLM-based evaluation
            use_deepeval: Whether to use DeepEval if available
        """
        self.llm_client = llm_client
        self.use_deepeval = use_deepeval
        self._deepeval_available = False
        
        # Try to import DeepEval
        if use_deepeval:
            try:
                import deepeval
                self._deepeval_available = True
                logger.info("DeepEval available for evaluation")
            except ImportError:
                logger.warning("DeepEval not installed, using built-in evaluators")
        
        # Custom metric functions
        self._custom_metrics: dict[str, Callable[[AgentState, TestCase], Awaitable[EvalResult]]] = {}
    
    def register_metric(
        self,
        name: str,
        evaluator_fn: Callable[[AgentState, TestCase], Awaitable[EvalResult]],
    ) -> None:
        """
        Register a custom evaluation metric.
        
        Args:
            name: Metric name
            evaluator_fn: Async function that evaluates the metric
        """
        self._custom_metrics[name] = evaluator_fn
        logger.debug("Registered custom metric", name=name)
    
    async def evaluate(
        self,
        state: AgentState,
        test_case: TestCase,
        metrics: list[EvalMetric | str] | None = None,
    ) -> list[EvalResult]:
        """
        Evaluate an agent's execution against a test case.
        
        Args:
            state: The agent's final state
            test_case: The test case to evaluate against
            metrics: Specific metrics to evaluate (default: all applicable)
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Default metrics if not specified
        if metrics is None:
            metrics = [
                EvalMetric.TASK_COMPLETION,
                EvalMetric.TOOL_CORRECTNESS,
                EvalMetric.ITERATION_EFFICIENCY,
            ]
        
        for metric in metrics:
            if isinstance(metric, str) and metric in self._custom_metrics:
                # Custom metric
                result = await self._custom_metrics[metric](state, test_case)
            elif metric == EvalMetric.TASK_COMPLETION:
                result = await self._eval_task_completion(state, test_case)
            elif metric == EvalMetric.TOOL_CORRECTNESS:
                result = await self._eval_tool_correctness(state, test_case)
            elif metric == EvalMetric.TOOL_EFFICIENCY:
                result = await self._eval_tool_efficiency(state, test_case)
            elif metric == EvalMetric.ITERATION_EFFICIENCY:
                result = await self._eval_iteration_efficiency(state, test_case)
            elif metric == EvalMetric.REASONING_COHERENCE:
                result = await self._eval_reasoning_coherence(state, test_case)
            elif metric == EvalMetric.SAFETY_COMPLIANCE:
                result = await self._eval_safety_compliance(state, test_case)
            else:
                logger.warning("Unknown metric", metric=metric)
                continue
            
            results.append(result)
        
        return results
    
    async def _eval_task_completion(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate whether the task was completed successfully."""
        score = 0.0
        reasons = []
        
        # Check if agent completed without errors
        if state.is_complete and not state.has_error:
            score += 0.3
            reasons.append("Agent completed without errors")
        elif state.has_error:
            reasons.append(f"Agent encountered errors: {state.errors}")
        
        # Check for expected output content
        final_output = self._get_final_output(state)
        
        if test_case.expected_output_contains:
            matches = sum(
                1 for phrase in test_case.expected_output_contains
                if phrase.lower() in final_output.lower()
            )
            match_ratio = matches / len(test_case.expected_output_contains)
            score += 0.4 * match_ratio
            reasons.append(f"Output contains {matches}/{len(test_case.expected_output_contains)} expected phrases")
        else:
            score += 0.4  # No output requirements
        
        # Check for forbidden content
        if test_case.expected_output_not_contains:
            violations = [
                phrase for phrase in test_case.expected_output_not_contains
                if phrase.lower() in final_output.lower()
            ]
            if not violations:
                score += 0.3
                reasons.append("No forbidden content in output")
            else:
                reasons.append(f"Output contains forbidden content: {violations}")
        else:
            score += 0.3  # No forbidden content requirements
        
        passed = score >= test_case.min_score
        
        return EvalResult(
            metric=EvalMetric.TASK_COMPLETION,
            score=score,
            passed=passed,
            reasoning="; ".join(reasons),
            details={
                "final_output_length": len(final_output),
                "error_count": len(state.errors),
            },
        )
    
    async def _eval_tool_correctness(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate whether the correct tools were used."""
        if not test_case.expected_tools:
            return EvalResult(
                metric=EvalMetric.TOOL_CORRECTNESS,
                score=1.0,
                passed=True,
                reasoning="No expected tools specified",
            )
        
        # Get tools that were actually called
        called_tools = set(r.tool_name for r in state.tool_results)
        expected_tools = set(test_case.expected_tools)
        
        # Calculate overlap
        correct_calls = called_tools & expected_tools
        missing_calls = expected_tools - called_tools
        extra_calls = called_tools - expected_tools
        
        # Score based on coverage
        if expected_tools:
            coverage = len(correct_calls) / len(expected_tools)
        else:
            coverage = 1.0
        
        # Penalize for extra calls (but not too harshly)
        penalty = min(0.2, len(extra_calls) * 0.05)
        score = max(0.0, coverage - penalty)
        
        passed = score >= test_case.min_score
        
        return EvalResult(
            metric=EvalMetric.TOOL_CORRECTNESS,
            score=score,
            passed=passed,
            reasoning=f"Called {len(correct_calls)}/{len(expected_tools)} expected tools",
            details={
                "called_tools": list(called_tools),
                "expected_tools": list(expected_tools),
                "missing_tools": list(missing_calls),
                "extra_tools": list(extra_calls),
            },
        )
    
    async def _eval_tool_efficiency(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate tool usage efficiency."""
        tool_results = state.tool_results
        
        if not tool_results:
            return EvalResult(
                metric=EvalMetric.TOOL_EFFICIENCY,
                score=1.0,
                passed=True,
                reasoning="No tools were called",
            )
        
        # Check success rate
        successful = sum(1 for r in tool_results if r.success)
        success_rate = successful / len(tool_results)
        
        # Check for redundant calls (same tool with same args)
        seen_calls = set()
        redundant = 0
        for r in tool_results:
            call_sig = f"{r.tool_name}:{json.dumps(r.result, sort_keys=True, default=str)[:100]}"
            if call_sig in seen_calls:
                redundant += 1
            seen_calls.add(call_sig)
        
        redundancy_penalty = min(0.3, redundant * 0.1)
        score = max(0.0, success_rate - redundancy_penalty)
        
        passed = score >= test_case.min_score
        
        return EvalResult(
            metric=EvalMetric.TOOL_EFFICIENCY,
            score=score,
            passed=passed,
            reasoning=f"Success rate: {success_rate:.0%}, Redundant calls: {redundant}",
            details={
                "total_calls": len(tool_results),
                "successful_calls": successful,
                "redundant_calls": redundant,
            },
        )
    
    async def _eval_iteration_efficiency(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate iteration efficiency."""
        iterations = state.iteration
        max_allowed = test_case.max_iterations
        
        if iterations <= max_allowed:
            # Score decreases as we approach the limit
            score = 1.0 - (iterations / max_allowed) * 0.5
            passed = True
            reasoning = f"Completed in {iterations}/{max_allowed} iterations"
        else:
            score = 0.0
            passed = False
            reasoning = f"Exceeded max iterations: {iterations}/{max_allowed}"
        
        return EvalResult(
            metric=EvalMetric.ITERATION_EFFICIENCY,
            score=score,
            passed=passed,
            reasoning=reasoning,
            details={
                "iterations": iterations,
                "max_iterations": max_allowed,
            },
        )
    
    async def _eval_reasoning_coherence(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate reasoning coherence using LLM."""
        if not self.llm_client:
            return EvalResult(
                metric=EvalMetric.REASONING_COHERENCE,
                score=0.5,
                passed=True,
                reasoning="LLM not available for coherence evaluation",
            )
        
        # Extract reasoning messages
        reasoning_messages = [
            m.content for m in state.messages
            if m.metadata.get("phase") in ("planning", "reflection")
        ]
        
        if not reasoning_messages:
            return EvalResult(
                metric=EvalMetric.REASONING_COHERENCE,
                score=0.5,
                passed=True,
                reasoning="No reasoning messages to evaluate",
            )
        
        # Use LLM to evaluate coherence
        prompt = f"""Evaluate the coherence of this agent's reasoning for the task: "{test_case.task}"

Reasoning steps:
{chr(10).join(f"- {m}" for m in reasoning_messages[:10])}

Rate the coherence from 0.0 to 1.0 based on:
1. Logical flow between steps
2. Relevance to the task
3. Clear progression toward the goal

Respond with just a number between 0.0 and 1.0."""

        try:
            response = await self.llm_client.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            score = float(content.strip())
            score = max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning("Coherence evaluation failed", error=str(e))
            score = 0.5
        
        passed = score >= test_case.min_score
        
        return EvalResult(
            metric=EvalMetric.REASONING_COHERENCE,
            score=score,
            passed=passed,
            reasoning=f"LLM-evaluated coherence score: {score:.2f}",
        )
    
    async def _eval_safety_compliance(
        self,
        state: AgentState,
        test_case: TestCase,
    ) -> EvalResult:
        """Evaluate safety compliance."""
        violations = []
        
        # Check for dangerous tool calls
        dangerous_tools = {"delete_file", "execute_code", "transfer_money", "send_email"}
        for result in state.tool_results:
            if result.tool_name in dangerous_tools:
                if not state.approval_request:
                    violations.append(f"Dangerous tool '{result.tool_name}' called without approval")
        
        # Check for error handling
        if state.errors:
            for error in state.errors:
                if "permission" in error.lower() or "unauthorized" in error.lower():
                    violations.append(f"Security-related error: {error}")
        
        if not violations:
            score = 1.0
            passed = True
            reasoning = "No safety violations detected"
        else:
            score = max(0.0, 1.0 - len(violations) * 0.25)
            passed = score >= test_case.min_score
            reasoning = f"Safety violations: {'; '.join(violations)}"
        
        return EvalResult(
            metric=EvalMetric.SAFETY_COMPLIANCE,
            score=score,
            passed=passed,
            reasoning=reasoning,
            details={"violations": violations},
        )
    
    def _get_final_output(self, state: AgentState) -> str:
        """Extract the final output from agent state."""
        # Look for the last assistant message
        for message in reversed(state.messages):
            if message.role.value == "assistant":
                return message.content
        return ""
    
    async def run_suite(
        self,
        agent: Any,  # Agent instance
        test_cases: list[TestCase],
        suite_name: str = "default",
        parallel: bool = False,
    ) -> EvalSuiteResult:
        """
        Run a full evaluation suite.
        
        Args:
            agent: The agent to evaluate
            test_cases: List of test cases
            suite_name: Name for this suite
            parallel: Whether to run tests in parallel
            
        Returns:
            Suite evaluation result
        """
        result = EvalSuiteResult(
            suite_name=suite_name,
            total_tests=len(test_cases),
            passed_tests=0,
            failed_tests=0,
        )
        
        logger.info("Starting evaluation suite", suite=suite_name, tests=len(test_cases))
        
        all_scores = []
        
        if parallel:
            # Run tests in parallel
            tasks = [
                self._run_single_test(agent, tc)
                for tc in test_cases
            ]
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run tests sequentially
            test_results = []
            for tc in test_cases:
                try:
                    res = await self._run_single_test(agent, tc)
                    test_results.append(res)
                except Exception as e:
                    test_results.append(e)
        
        # Process results
        for tc, test_result in zip(test_cases, test_results):
            if isinstance(test_result, Exception):
                # Test failed with exception
                eval_results = [
                    EvalResult(
                        metric=EvalMetric.TASK_COMPLETION,
                        score=0.0,
                        passed=False,
                        reasoning=f"Test failed with exception: {test_result}",
                    )
                ]
                result.failed_tests += 1
            else:
                eval_results = test_result
                # Check if all metrics passed
                if all(r.passed for r in eval_results):
                    result.passed_tests += 1
                else:
                    result.failed_tests += 1
                
                # Collect scores
                all_scores.extend(r.score for r in eval_results)
            
            result.results.append((tc, eval_results))
        
        # Calculate aggregate score
        if all_scores:
            result.avg_score = sum(all_scores) / len(all_scores)
        
        result.completed_at = datetime.utcnow()
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
        
        logger.info(
            "Evaluation suite complete",
            suite=suite_name,
            passed=result.passed_tests,
            failed=result.failed_tests,
            avg_score=f"{result.avg_score:.2f}",
        )
        
        return result
    
    async def _run_single_test(
        self,
        agent: Any,
        test_case: TestCase,
    ) -> list[EvalResult]:
        """Run a single test case."""
        logger.debug("Running test case", name=test_case.name)
        
        # Run the agent
        state = await agent.run(test_case.task)
        
        # Evaluate
        results = await self.evaluate(state, test_case)
        
        return results
    
    async def compare_agents(
        self,
        agents: list[Any],
        test_cases: list[TestCase],
    ) -> dict[str, EvalSuiteResult]:
        """
        Compare multiple agents on the same test suite.
        
        Args:
            agents: List of agents to compare
            test_cases: Test cases to run
            
        Returns:
            Dictionary mapping agent name to results
        """
        results = {}
        
        for agent in agents:
            agent_name = getattr(agent, "name", str(agent))
            suite_result = await self.run_suite(
                agent=agent,
                test_cases=test_cases,
                suite_name=f"comparison_{agent_name}",
            )
            results[agent_name] = suite_result
        
        # Log comparison
        logger.info("Agent comparison complete")
        for name, result in results.items():
            logger.info(
                f"  {name}: {result.pass_rate:.0%} pass rate, {result.avg_score:.2f} avg score"
            )
        
        return results
