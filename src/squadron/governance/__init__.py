"""
L4: Governance Layer

Provides evaluation, safety, and guardrails for agent behavior:
- DeepEval integration for agentic evaluation
- Safety guardrails for high-risk actions
- Regression testing for self-improvement
"""

from squadron.governance.evaluator import (
    AgentEvaluator,
    EvalResult,
    EvalMetric,
    TestCase,
)
from squadron.governance.guardrails import (
    Guardrail,
    GuardrailResult,
    SafetyGuardrails,
)

__all__ = [
    # Evaluation
    "AgentEvaluator",
    "EvalResult",
    "EvalMetric",
    "TestCase",
    # Safety
    "Guardrail",
    "GuardrailResult",
    "SafetyGuardrails",
]
