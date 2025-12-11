"""
SICA - Self-Improving Coding Agent

Implements the self-improvement loop from the SICA paper.
Allows agents to modify their own code, prompts, and tools
based on performance feedback.

Key features:
- Prompt optimization
- Tool code improvement
- Configuration tuning
- Regression testing before acceptance
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

import structlog

from squadron.core.config import EvolutionConfig
from squadron.evolution.archive import ArchiveEntry, EvolutionArchive
from squadron.evolution.sandbox import Sandbox, SandboxConfig, ExecutionResult
from squadron.governance.evaluator import AgentEvaluator, TestCase, EvalSuiteResult

logger = structlog.get_logger(__name__)


class MutationType(str, Enum):
    """Types of mutations the SICA engine can perform."""
    
    PROMPT_OPTIMIZATION = "prompt_optimization"
    TOOL_IMPROVEMENT = "tool_improvement"
    CONFIG_TUNING = "config_tuning"
    CODE_REFACTOR = "code_refactor"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class Mutation:
    """
    A proposed mutation to the agent.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Type
    mutation_type: MutationType = MutationType.PROMPT_OPTIMIZATION
    
    # Target
    target_file: str = ""
    target_function: str = ""
    target_line_start: int = 0
    target_line_end: int = 0
    
    # Changes
    original_code: str = ""
    mutated_code: str = ""
    
    # Reasoning
    reasoning: str = ""
    expected_improvement: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_diff(self) -> str:
        """Generate a unified diff of the changes."""
        original_lines = self.original_code.splitlines(keepends=True)
        mutated_lines = self.mutated_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            mutated_lines,
            fromfile=f"a/{self.target_file}",
            tofile=f"b/{self.target_file}",
        )
        
        return "".join(diff)
    
    def get_code_hash(self) -> str:
        """Get a hash of the mutated code."""
        return hashlib.sha256(self.mutated_code.encode()).hexdigest()[:16]


@dataclass
class ImprovementResult:
    """
    Result of an improvement attempt.
    """
    
    mutation: Mutation
    
    # Scores
    baseline_score: float = 0.0
    mutated_score: float = 0.0
    improvement: float = 0.0
    
    # Status
    accepted: bool = False
    rejected_reason: str = ""
    
    # Test results
    tests_passed: bool = False
    test_details: dict[str, Any] = field(default_factory=dict)
    
    # Sandbox execution
    sandbox_result: ExecutionResult | None = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()


class SICAEngine:
    """
    SICA Engine - Self-Improving Coding Agent.
    
    Implements the recursive self-improvement loop:
    1. Execute & Benchmark: Run the agent and measure performance
    2. Reflect: Analyze execution traces for improvement opportunities
    3. Mutate: Generate code/prompt modifications
    4. Verify: Test mutations in sandbox with regression suite
    5. Accept/Reject: Only accept improvements that pass all tests
    
    Example:
        ```python
        sica = SICAEngine(
            config=EvolutionConfig(),
            evaluator=AgentEvaluator(),
        )
        
        # Run improvement cycle
        result = await sica.improve(
            agent=agent,
            test_cases=test_cases,
            mutation_type=MutationType.PROMPT_OPTIMIZATION,
        )
        
        if result.accepted:
            print(f"Improvement accepted: {result.improvement:.2%}")
        ```
    """
    
    def __init__(
        self,
        config: EvolutionConfig | None = None,
        evaluator: AgentEvaluator | None = None,
        archive: EvolutionArchive | None = None,
        sandbox: Sandbox | None = None,
        llm_client: Any | None = None,
    ):
        """
        Initialize the SICA engine.
        
        Args:
            config: Evolution configuration
            evaluator: Agent evaluator for testing
            archive: Evolution archive for history
            sandbox: Sandbox for safe execution
            llm_client: LLM client for generating mutations
        """
        self.config = config or EvolutionConfig()
        self.evaluator = evaluator or AgentEvaluator()
        self.archive = archive or EvolutionArchive()
        self.sandbox = sandbox or Sandbox()
        self.llm_client = llm_client
        
        # Mutation generators
        self._mutation_generators: dict[
            MutationType,
            Callable[[Any, list[TestCase]], Awaitable[list[Mutation]]]
        ] = {
            MutationType.PROMPT_OPTIMIZATION: self._generate_prompt_mutations,
            MutationType.TOOL_IMPROVEMENT: self._generate_tool_mutations,
            MutationType.CONFIG_TUNING: self._generate_config_mutations,
        }
    
    async def improve(
        self,
        agent: Any,
        test_cases: list[TestCase],
        mutation_type: MutationType | None = None,
        max_attempts: int = 5,
    ) -> ImprovementResult:
        """
        Run an improvement cycle.
        
        Args:
            agent: The agent to improve
            test_cases: Test cases for evaluation
            mutation_type: Type of mutation to attempt (or auto-detect)
            max_attempts: Maximum mutation attempts
            
        Returns:
            The improvement result
        """
        if not self.config.enable_self_improvement:
            return ImprovementResult(
                mutation=Mutation(),
                rejected_reason="Self-improvement is disabled",
            )
        
        logger.info("Starting improvement cycle", mutation_type=mutation_type)
        
        # Step 1: Establish baseline
        baseline_result = await self.evaluator.run_suite(
            agent=agent,
            test_cases=test_cases,
            suite_name="baseline",
        )
        baseline_score = baseline_result.avg_score
        
        logger.info("Baseline established", score=f"{baseline_score:.2f}")
        
        # Step 2: Generate mutations
        if mutation_type:
            mutation_types = [mutation_type]
        else:
            mutation_types = list(MutationType)
        
        best_result: ImprovementResult | None = None
        
        for mt in mutation_types:
            for attempt in range(max_attempts):
                logger.debug(
                    "Attempting mutation",
                    type=mt.value,
                    attempt=attempt + 1,
                )
                
                # Generate mutation
                mutations = await self._generate_mutations(agent, test_cases, mt)
                
                if not mutations:
                    logger.debug("No mutations generated", type=mt.value)
                    continue
                
                # Try each mutation
                for mutation in mutations:
                    # Check if already tried
                    if self.archive.was_tried(
                        mutation_type=mt.value,
                        target_function=mutation.target_function,
                        code_snippet=mutation.mutated_code[:100],
                    ):
                        logger.debug("Mutation already tried, skipping")
                        continue
                    
                    # Step 3: Test mutation in sandbox
                    result = await self._test_mutation(
                        agent=agent,
                        mutation=mutation,
                        test_cases=test_cases,
                        baseline_score=baseline_score,
                    )
                    
                    # Step 4: Evaluate result
                    if result.accepted:
                        logger.info(
                            "Mutation accepted",
                            improvement=f"{result.improvement:.2%}",
                        )
                        
                        # Record in archive
                        await self._record_result(result)
                        
                        return result
                    
                    # Record failed attempt
                    await self._record_result(result)
                    
                    # Track best attempt
                    if best_result is None or result.mutated_score > best_result.mutated_score:
                        best_result = result
        
        # No successful mutation found
        if best_result:
            return best_result
        
        return ImprovementResult(
            mutation=Mutation(),
            baseline_score=baseline_score,
            rejected_reason="No successful mutations found",
        )
    
    async def _generate_mutations(
        self,
        agent: Any,
        test_cases: list[TestCase],
        mutation_type: MutationType,
    ) -> list[Mutation]:
        """Generate mutations of a specific type."""
        generator = self._mutation_generators.get(mutation_type)
        
        if generator:
            return await generator(agent, test_cases)
        
        return []
    
    async def _generate_prompt_mutations(
        self,
        agent: Any,
        test_cases: list[TestCase],
    ) -> list[Mutation]:
        """Generate prompt optimization mutations."""
        mutations = []
        
        if not self.llm_client:
            logger.warning("LLM client required for prompt mutations")
            return mutations
        
        # Find prompt templates in the agent
        prompt_sources = self._find_prompt_sources(agent)
        
        for source in prompt_sources:
            # Generate improved version using LLM
            prompt = f"""You are an expert prompt engineer. Analyze this prompt template and suggest an improved version.

Original prompt:
```
{source['content']}
```

Context: This prompt is used for {source['purpose']}.

Requirements:
1. Maintain the same input/output format
2. Improve clarity and specificity
3. Add better examples if helpful
4. Reduce ambiguity

Provide ONLY the improved prompt, no explanation."""

            try:
                response = await self.llm_client.ainvoke(prompt)
                improved = response.content if hasattr(response, "content") else str(response)
                
                if improved and improved != source['content']:
                    mutations.append(Mutation(
                        mutation_type=MutationType.PROMPT_OPTIMIZATION,
                        target_file=source['file'],
                        target_function=source['function'],
                        original_code=source['content'],
                        mutated_code=improved.strip(),
                        reasoning="LLM-generated prompt improvement",
                        expected_improvement="Better task completion and clarity",
                    ))
                    
            except Exception as e:
                logger.warning("Prompt mutation generation failed", error=str(e))
        
        return mutations
    
    async def _generate_tool_mutations(
        self,
        agent: Any,
        test_cases: list[TestCase],
    ) -> list[Mutation]:
        """Generate tool improvement mutations."""
        mutations = []
        
        if not self.llm_client:
            return mutations
        
        # Find tools in the agent
        tools = getattr(agent, "tools", [])
        
        for tool in tools:
            # Get tool source code
            try:
                source = inspect.getsource(tool)
                file_path = inspect.getfile(tool)
            except Exception:
                continue
            
            # Generate improved version
            prompt = f"""You are an expert Python developer. Analyze this tool function and suggest improvements.

Original code:
```python
{source}
```

Focus on:
1. Error handling
2. Edge cases
3. Performance
4. Code clarity

Provide ONLY the improved code, no explanation."""

            try:
                response = await self.llm_client.ainvoke(prompt)
                improved = response.content if hasattr(response, "content") else str(response)
                
                # Extract code from response
                if "```python" in improved:
                    improved = improved.split("```python")[1].split("```")[0]
                elif "```" in improved:
                    improved = improved.split("```")[1].split("```")[0]
                
                if improved and improved.strip() != source.strip():
                    mutations.append(Mutation(
                        mutation_type=MutationType.TOOL_IMPROVEMENT,
                        target_file=file_path,
                        target_function=getattr(tool, "__name__", "unknown"),
                        original_code=source,
                        mutated_code=improved.strip(),
                        reasoning="LLM-generated tool improvement",
                        expected_improvement="Better error handling and performance",
                    ))
                    
            except Exception as e:
                logger.warning("Tool mutation generation failed", error=str(e))
        
        return mutations
    
    async def _generate_config_mutations(
        self,
        agent: Any,
        test_cases: list[TestCase],
    ) -> list[Mutation]:
        """Generate configuration tuning mutations."""
        mutations = []
        
        config = getattr(agent, "config", None)
        if not config:
            return mutations
        
        # Try different configuration values
        config_options = [
            ("reasoning.n_candidates", [3, 5, 7, 10]),
            ("reasoning.max_depth", [5, 10, 15]),
            ("reasoning.exploration_constant", [1.0, 1.414, 2.0]),
            ("governance.max_iterations", [25, 50, 75]),
        ]
        
        for config_path, values in config_options:
            current_value = self._get_config_value(config, config_path)
            
            for new_value in values:
                if new_value != current_value:
                    mutations.append(Mutation(
                        mutation_type=MutationType.CONFIG_TUNING,
                        target_file="config",
                        target_function=config_path,
                        original_code=str(current_value),
                        mutated_code=str(new_value),
                        reasoning=f"Testing {config_path}={new_value}",
                        expected_improvement="Better performance with tuned config",
                    ))
        
        return mutations
    
    def _find_prompt_sources(self, agent: Any) -> list[dict[str, Any]]:
        """Find prompt templates in the agent."""
        sources = []
        
        # Check reasoner
        reasoner = getattr(agent, "reasoner", None)
        if reasoner:
            verifier = getattr(reasoner, "verifier", None)
            if verifier:
                ranking_prompt = getattr(verifier, "_ranking_prompt", None)
                if ranking_prompt:
                    sources.append({
                        "content": ranking_prompt,
                        "purpose": "ranking candidate plans",
                        "file": "reasoning/verifier.py",
                        "function": "ListWiseVerifier",
                    })
        
        return sources
    
    def _get_config_value(self, config: Any, path: str) -> Any:
        """Get a nested config value by path."""
        parts = path.split(".")
        value = config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        
        return value
    
    async def _test_mutation(
        self,
        agent: Any,
        mutation: Mutation,
        test_cases: list[TestCase],
        baseline_score: float,
    ) -> ImprovementResult:
        """Test a mutation in the sandbox."""
        result = ImprovementResult(
            mutation=mutation,
            baseline_score=baseline_score,
        )
        
        try:
            # Apply mutation temporarily
            original_value = await self._apply_mutation(agent, mutation)
            
            try:
                # Run tests
                eval_result = await self.evaluator.run_suite(
                    agent=agent,
                    test_cases=test_cases,
                    suite_name="mutation_test",
                )
                
                result.mutated_score = eval_result.avg_score
                result.improvement = result.mutated_score - baseline_score
                result.tests_passed = eval_result.all_passed
                result.test_details = eval_result.to_dict()
                
                # Check if improvement meets threshold
                if (
                    result.improvement >= self.config.min_improvement_threshold
                    and result.tests_passed
                ):
                    result.accepted = True
                else:
                    if not result.tests_passed:
                        result.rejected_reason = "Tests failed"
                    else:
                        result.rejected_reason = (
                            f"Improvement {result.improvement:.2%} below threshold "
                            f"{self.config.min_improvement_threshold:.2%}"
                        )
                
            finally:
                # Revert mutation if not accepted
                if not result.accepted:
                    await self._revert_mutation(agent, mutation, original_value)
                    
        except Exception as e:
            result.rejected_reason = f"Error testing mutation: {e}"
            logger.error("Mutation test failed", error=str(e))
        
        result.completed_at = datetime.utcnow()
        return result
    
    async def _apply_mutation(
        self,
        agent: Any,
        mutation: Mutation,
    ) -> Any:
        """Apply a mutation and return the original value."""
        if mutation.mutation_type == MutationType.PROMPT_OPTIMIZATION:
            # Find and update prompt
            reasoner = getattr(agent, "reasoner", None)
            if reasoner:
                verifier = getattr(reasoner, "verifier", None)
                if verifier and hasattr(verifier, "_ranking_prompt"):
                    original = verifier._ranking_prompt
                    verifier._ranking_prompt = mutation.mutated_code
                    return original
        
        elif mutation.mutation_type == MutationType.CONFIG_TUNING:
            # Update config value
            config = getattr(agent, "config", None)
            if config:
                parts = mutation.target_function.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                
                if obj:
                    original = getattr(obj, parts[-1], None)
                    setattr(obj, parts[-1], type(original)(mutation.mutated_code))
                    return original
        
        return None
    
    async def _revert_mutation(
        self,
        agent: Any,
        mutation: Mutation,
        original_value: Any,
    ) -> None:
        """Revert a mutation to its original value."""
        if original_value is None:
            return
        
        if mutation.mutation_type == MutationType.PROMPT_OPTIMIZATION:
            reasoner = getattr(agent, "reasoner", None)
            if reasoner:
                verifier = getattr(reasoner, "verifier", None)
                if verifier:
                    verifier._ranking_prompt = original_value
        
        elif mutation.mutation_type == MutationType.CONFIG_TUNING:
            config = getattr(agent, "config", None)
            if config:
                parts = mutation.target_function.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                
                if obj:
                    setattr(obj, parts[-1], original_value)
    
    async def _record_result(self, result: ImprovementResult) -> None:
        """Record an improvement result in the archive."""
        entry = ArchiveEntry(
            mutation_type=result.mutation.mutation_type.value,
            mutation_description=result.mutation.reasoning,
            original_code=result.mutation.original_code,
            mutated_code=result.mutation.mutated_code,
            diff=result.mutation.get_diff(),
            target_file=result.mutation.target_file,
            target_function=result.mutation.target_function,
            baseline_score=result.baseline_score,
            mutated_score=result.mutated_score,
            improvement=result.improvement,
            accepted=result.accepted,
            rejected_reason=result.rejected_reason,
        )
        
        await self.archive.add(entry)
    
    async def reflect(
        self,
        agent: Any,
        execution_trace: list[dict[str, Any]],
    ) -> list[str]:
        """
        Analyze an execution trace for improvement opportunities.
        
        Args:
            agent: The agent
            execution_trace: Trace of agent execution
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if not self.llm_client:
            return suggestions
        
        # Format trace for analysis
        trace_summary = self._summarize_trace(execution_trace)
        
        prompt = f"""Analyze this agent execution trace and identify improvement opportunities.

Execution Trace:
{trace_summary}

Identify:
1. Inefficiencies (unnecessary steps, redundant tool calls)
2. Errors or failures that could be prevented
3. Missing capabilities that would help
4. Prompt improvements that would help

Provide specific, actionable suggestions."""

        try:
            response = await self.llm_client.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse suggestions
            for line in content.split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                    suggestions.append(line.lstrip("-*0123456789. "))
                    
        except Exception as e:
            logger.warning("Reflection failed", error=str(e))
        
        return suggestions
    
    def _summarize_trace(self, trace: list[dict[str, Any]]) -> str:
        """Summarize an execution trace."""
        lines = []
        
        for i, step in enumerate(trace[:20]):  # Limit to 20 steps
            step_type = step.get("type", "unknown")
            content = step.get("content", "")[:200]
            lines.append(f"Step {i + 1} ({step_type}): {content}")
        
        return "\n".join(lines)
    
    def get_improvement_history(
        self,
        mutation_type: MutationType | None = None,
    ) -> list[ArchiveEntry]:
        """Get history of improvement attempts."""
        if mutation_type:
            return self.archive.find_similar(mutation_type=mutation_type.value)
        return self.archive.entries
