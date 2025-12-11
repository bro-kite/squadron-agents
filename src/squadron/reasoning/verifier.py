"""
List-Wise Verification System

Implements list-wise ranking for selecting optimal trajectories.
Research shows list-wise comparison significantly outperforms
point-wise scoring for plan selection.
"""

from typing import Any
from uuid import UUID

import structlog

from squadron.core.config import ReasoningConfig

logger = structlog.get_logger(__name__)


class CandidatePlan:
    """A candidate plan to be ranked."""

    def __init__(
        self,
        id: UUID,
        thought: str,
        action: str,
        expected_outcome: str,
        context: dict[str, Any] | None = None,
    ):
        self.id = id
        self.thought = thought
        self.action = action
        self.expected_outcome = expected_outcome
        self.context = context or {}
        self.rank: int | None = None
        self.score: float | None = None
        self.reasoning: str | None = None

    def to_prompt_format(self, index: int) -> str:
        """Format the plan for inclusion in a ranking prompt."""
        return f"""Plan {index + 1}:
- Thought: {self.thought}
- Action: {self.action}
- Expected Outcome: {self.expected_outcome}
"""


class ListWiseVerifier:
    """
    List-wise verification for plan selection.
    
    Instead of scoring each plan independently (point-wise),
    this verifier compares all plans together and produces
    a ranked list. This approach is more robust and produces
    better selections.
    
    Example:
        ```python
        verifier = ListWiseVerifier(config=ReasoningConfig())
        
        ranked_plans = await verifier.rank(
            task="Refactor the authentication module",
            candidates=candidate_plans,
            context={"codebase": "Python", "framework": "FastAPI"},
        )
        
        best_plan = ranked_plans[0]
        ```
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        llm_client: Any | None = None,
    ):
        """
        Initialize the verifier.
        
        Args:
            config: Reasoning configuration
            llm_client: LLM client for ranking (optional)
        """
        self.config = config or ReasoningConfig()
        self.llm_client = llm_client
        
        # Ranking prompt template
        self._ranking_prompt = """You are an expert plan evaluator. Your task is to rank the following candidate plans for accomplishing a task.

## Task
{task}

## Context
{context}

## Candidate Plans
{plans}

## Instructions
Analyze each plan carefully and rank them from BEST to WORST. Consider:
1. Feasibility: Can this plan actually be executed?
2. Effectiveness: Will this plan accomplish the task?
3. Efficiency: Is this the most direct path to the goal?
4. Risk: What could go wrong? How recoverable are failures?
5. Completeness: Does the plan address all aspects of the task?

## Output Format
Provide your ranking as a JSON array of plan numbers, from best to worst.
Then provide brief reasoning for your top choice.

Example:
```json
{{
    "ranking": [3, 1, 5, 2, 4],
    "reasoning": "Plan 3 is best because..."
}}
```

Your ranking:"""

    async def rank(
        self,
        task: str,
        candidates: list[CandidatePlan],
        context: dict[str, Any] | None = None,
    ) -> list[CandidatePlan]:
        """
        Rank candidate plans using list-wise comparison.
        
        Args:
            task: The task description
            candidates: List of candidate plans to rank
            context: Additional context for ranking
            
        Returns:
            Sorted list of plans (best first)
        """
        if not candidates:
            return []
        
        if len(candidates) == 1:
            candidates[0].rank = 1
            candidates[0].score = 1.0
            return candidates
        
        logger.debug(
            "Ranking candidates",
            task=task[:50],
            num_candidates=len(candidates),
        )
        
        # Format plans for the prompt
        plans_text = "\n".join(
            plan.to_prompt_format(i)
            for i, plan in enumerate(candidates)
        )
        
        # Format context
        context_text = ""
        if context:
            context_text = "\n".join(f"- {k}: {v}" for k, v in context.items())
        else:
            context_text = "No additional context provided."
        
        # Build the ranking prompt
        prompt = self._ranking_prompt.format(
            task=task,
            context=context_text,
            plans=plans_text,
        )
        
        # Get ranking from LLM
        if self.llm_client:
            ranking_result = await self._get_llm_ranking(prompt, len(candidates))
        else:
            # Fallback to heuristic ranking
            ranking_result = await self._heuristic_ranking(candidates, task)
        
        # Apply rankings to candidates
        ranking = ranking_result.get("ranking", list(range(1, len(candidates) + 1)))
        reasoning = ranking_result.get("reasoning", "")
        
        # Validate ranking
        if len(ranking) != len(candidates):
            logger.warning(
                "Invalid ranking length, using original order",
                expected=len(candidates),
                got=len(ranking),
            )
            ranking = list(range(1, len(candidates) + 1))
        
        # Assign ranks and scores
        for rank_position, plan_index in enumerate(ranking):
            # Convert 1-indexed to 0-indexed
            idx = plan_index - 1
            if 0 <= idx < len(candidates):
                candidates[idx].rank = rank_position + 1
                # Score decreases linearly with rank
                candidates[idx].score = 1.0 - (rank_position / len(candidates))
                if rank_position == 0:
                    candidates[idx].reasoning = reasoning
        
        # Sort by rank
        ranked = sorted(candidates, key=lambda p: p.rank or float("inf"))
        
        logger.info(
            "Ranking complete",
            best_plan=ranked[0].thought[:50] if ranked else None,
            best_score=ranked[0].score if ranked else None,
        )
        
        return ranked

    async def _get_llm_ranking(
        self,
        prompt: str,
        num_candidates: int,
    ) -> dict[str, Any]:
        """Get ranking from LLM."""
        import json
        
        try:
            # Call LLM
            response = await self.llm_client.ainvoke(prompt)
            
            # Parse response
            content = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                return result
            
            logger.warning("Could not parse LLM ranking response")
            return {"ranking": list(range(1, num_candidates + 1))}
            
        except Exception as e:
            logger.error("LLM ranking failed", error=str(e))
            return {"ranking": list(range(1, num_candidates + 1))}

    async def _heuristic_ranking(
        self,
        candidates: list[CandidatePlan],
        task: str,
    ) -> dict[str, Any]:
        """
        Fallback heuristic ranking when LLM is not available.
        
        Uses simple keyword matching and length heuristics.
        """
        task_words = set(task.lower().split())
        
        scores = []
        for i, plan in enumerate(candidates):
            score = 0.0
            
            # Keyword overlap with task
            thought_words = set(plan.thought.lower().split())
            action_words = set(plan.action.lower().split())
            
            overlap = len(task_words & (thought_words | action_words))
            score += overlap * 0.1
            
            # Prefer more detailed plans (but not too long)
            thought_len = len(plan.thought)
            if 50 <= thought_len <= 500:
                score += 0.2
            
            # Prefer plans with clear expected outcomes
            if plan.expected_outcome and len(plan.expected_outcome) > 20:
                score += 0.2
            
            # Penalize vague language
            vague_words = {"maybe", "might", "could", "possibly", "try"}
            if vague_words & thought_words:
                score -= 0.1
            
            scores.append((i + 1, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [idx for idx, _ in scores]
        
        return {
            "ranking": ranking,
            "reasoning": "Ranked using heuristic scoring (LLM not available)",
        }

    async def compare_pair(
        self,
        plan_a: CandidatePlan,
        plan_b: CandidatePlan,
        task: str,
    ) -> CandidatePlan:
        """
        Compare two plans and return the better one.
        
        Useful for tournament-style selection.
        """
        ranked = await self.rank(task, [plan_a, plan_b])
        return ranked[0]

    async def filter_viable(
        self,
        candidates: list[CandidatePlan],
        task: str,
        min_score: float = 0.5,
    ) -> list[CandidatePlan]:
        """
        Filter candidates to only include viable plans.
        
        Args:
            candidates: Plans to filter
            task: Task description
            min_score: Minimum score threshold
            
        Returns:
            List of viable plans
        """
        ranked = await self.rank(task, candidates)
        return [p for p in ranked if (p.score or 0) >= min_score]