"""
Tests for L4 Governance Layer.

Tests evaluation system and safety guardrails.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import re

from squadron.governance.evaluator import (
    AgentEvaluator,
    EvalMetric,
    EvalResult,
    TestCase,
)
from squadron.governance.guardrails import (
    SafetyGuardrails,
    Guardrail,
    GuardrailAction,
    GuardrailResult,
)
from squadron.core.config import GovernanceConfig


class TestEvalMetric:
    """Tests for EvalMetric enum."""
    
    def test_all_metrics_exist(self):
        assert EvalMetric.TASK_COMPLETION
        assert EvalMetric.TOOL_CORRECTNESS
        assert EvalMetric.TOOL_EFFICIENCY
        assert EvalMetric.REASONING_COHERENCE
        assert EvalMetric.SAFETY_COMPLIANCE


class TestEvalResult:
    """Tests for EvalResult."""
    
    def test_create(self):
        result = EvalResult(
            metric=EvalMetric.TASK_COMPLETION,
            score=0.85,
            passed=True,
        )
        assert result.score == 0.85
        assert result.passed is True
    
    def test_with_details(self):
        result = EvalResult(
            metric=EvalMetric.TOOL_CORRECTNESS,
            score=1.0,
            passed=True,
            details={"expected": ["tool1"], "actual": ["tool1"]},
        )
        assert result.details["expected"] == ["tool1"]
    
    def test_to_dict(self):
        result = EvalResult(
            metric=EvalMetric.TOOL_EFFICIENCY,
            score=0.75,
            passed=True,
        )
        data = result.to_dict()
        assert data["metric"] == "tool_efficiency"
        assert data["score"] == 0.75


class TestTestCase:
    """Tests for TestCase."""
    
    def test_create(self):
        tc = TestCase(
            name="test_file_read",
            task="Read the README file",
            expected_tools=["read_file"],
        )
        assert tc.name == "test_file_read"
        assert tc.task == "Read the README file"
        assert "read_file" in tc.expected_tools
    
    def test_with_expected_output(self):
        tc = TestCase(
            name="test_greeting",
            task="Say hello",
            expected_output_contains=["hello", "hi"],
        )
        assert "hello" in tc.expected_output_contains


class TestAgentEvaluator:
    """Tests for AgentEvaluator."""
    
    def test_init(self):
        evaluator = AgentEvaluator()
        assert evaluator is not None


class TestGuardrailResult:
    """Tests for GuardrailResult."""
    
    def test_passed(self):
        result = GuardrailResult(
            guardrail_name="test",
            action=GuardrailAction.ALLOW,
            passed=True,
        )
        assert result.passed is True
    
    def test_blocked(self):
        result = GuardrailResult(
            guardrail_name="shell_injection",
            action=GuardrailAction.BLOCK,
            passed=False,
            reason="Blocked pattern detected",
        )
        assert result.passed is False
        assert result.action == GuardrailAction.BLOCK


class TestGuardrail:
    """Tests for Guardrail."""
    
    def test_create(self):
        guardrail = Guardrail(
            name="no_delete",
            description="Prevent file deletion",
            action=GuardrailAction.REQUIRE_APPROVAL,
        )
        assert guardrail.name == "no_delete"
        assert guardrail.action == GuardrailAction.REQUIRE_APPROVAL
    
    def test_with_patterns(self):
        guardrail = Guardrail(
            name="sql_injection",
            description="Block SQL injection",
            blocked_patterns=[re.compile(r"DROP TABLE", re.IGNORECASE)],
            action=GuardrailAction.BLOCK,
        )
        assert len(guardrail.blocked_patterns) == 1
    
    def test_applies_to_tools(self):
        guardrail = Guardrail(
            name="dangerous_tools",
            description="Require approval for dangerous tools",
            applies_to_tools=["delete_file", "run_command"],
            action=GuardrailAction.REQUIRE_APPROVAL,
        )
        assert "delete_file" in guardrail.applies_to_tools


class TestSafetyGuardrails:
    """Tests for SafetyGuardrails."""
    
    def test_init(self):
        guardrails = SafetyGuardrails()
        assert guardrails is not None
    
    def test_add_guardrail(self):
        guardrails = SafetyGuardrails()
        initial_count = len(guardrails.get_guardrails())
        
        guardrails.add_guardrail(Guardrail(
            name="custom",
            description="Custom guardrail",
        ))
        
        assert len(guardrails.get_guardrails()) == initial_count + 1
    
    def test_remove_guardrail(self):
        guardrails = SafetyGuardrails()
        guardrails.add_guardrail(Guardrail(
            name="to_remove",
            description="Will be removed",
        ))
        
        removed = guardrails.remove_guardrail("to_remove")
        assert removed is True
        
        removed_again = guardrails.remove_guardrail("to_remove")
        assert removed_again is False
    
    def test_enable_disable_guardrail(self):
        guardrails = SafetyGuardrails()
        guardrails.add_guardrail(Guardrail(
            name="toggleable",
            description="Can be toggled",
            enabled=True,
        ))
        
        guardrails.disable_guardrail("toggleable")
        g = guardrails.get_guardrail("toggleable")
        assert g.enabled is False
        
        guardrails.enable_guardrail("toggleable")
        g = guardrails.get_guardrail("toggleable")
        assert g.enabled is True
