"""
Tests for L5 Evolution Layer.

Tests SICA engine, sandbox execution, and evolution archive.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4

from squadron.evolution.sica import (
    SICAEngine,
    Mutation,
    MutationType,
    ImprovementResult,
)
from squadron.evolution.sandbox import (
    Sandbox,
    SandboxConfig,
    SandboxType,
    ExecutionResult,
)
from squadron.evolution.archive import (
    ArchiveEntry,
    EvolutionArchive,
)
from squadron.core.config import EvolutionConfig


class TestMutationType:
    """Tests for MutationType enum."""
    
    def test_all_types_exist(self):
        assert MutationType.PROMPT_OPTIMIZATION
        assert MutationType.TOOL_IMPROVEMENT
        assert MutationType.CONFIG_TUNING
        assert MutationType.CODE_REFACTOR


class TestMutation:
    """Tests for Mutation."""
    
    def test_create(self):
        mutation = Mutation(
            mutation_type=MutationType.PROMPT_OPTIMIZATION,
            target_file="reasoning/verifier.py",
            target_function="ListWiseVerifier",
            original_code="old prompt",
            mutated_code="new prompt",
        )
        assert mutation.mutation_type == MutationType.PROMPT_OPTIMIZATION
        assert mutation.target_file == "reasoning/verifier.py"
    
    def test_get_diff(self):
        mutation = Mutation(
            mutation_type=MutationType.CODE_REFACTOR,
            original_code="def foo():\n    pass",
            mutated_code="def foo():\n    return 42",
        )
        diff = mutation.get_diff()
        assert "---" in diff or "-    pass" in diff
    
    def test_get_code_hash(self):
        mutation = Mutation(mutated_code="some code")
        hash1 = mutation.get_code_hash()
        
        mutation2 = Mutation(mutated_code="different code")
        hash2 = mutation2.get_code_hash()
        
        assert hash1 != hash2
        assert len(hash1) == 16


class TestImprovementResult:
    """Tests for ImprovementResult."""
    
    def test_create(self):
        mutation = Mutation(mutation_type=MutationType.CONFIG_TUNING)
        result = ImprovementResult(
            mutation=mutation,
            baseline_score=0.75,
            mutated_score=0.82,
            improvement=0.07,
            accepted=True,
        )
        assert result.improvement == 0.07
        assert result.accepted is True
    
    def test_rejected(self):
        mutation = Mutation(mutation_type=MutationType.PROMPT_OPTIMIZATION)
        result = ImprovementResult(
            mutation=mutation,
            baseline_score=0.75,
            mutated_score=0.70,
            improvement=-0.05,
            accepted=False,
            rejected_reason="Score decreased",
        )
        assert result.accepted is False
        assert "decreased" in result.rejected_reason


class TestSandboxConfig:
    """Tests for SandboxConfig."""
    
    def test_defaults(self):
        config = SandboxConfig()
        assert config.sandbox_type == SandboxType.SUBPROCESS
        assert config.max_memory_mb == 512
        assert config.max_execution_seconds == 60.0
    
    def test_docker_config(self):
        config = SandboxConfig(
            sandbox_type=SandboxType.DOCKER,
            docker_image="python:3.11-slim",
            allow_network=False,
        )
        assert config.sandbox_type == SandboxType.DOCKER
        assert config.docker_image == "python:3.11-slim"
    
    def test_from_evolution_config(self):
        evo_config = EvolutionConfig()
        sandbox_config = SandboxConfig.from_evolution_config(evo_config)
        assert sandbox_config is not None


class TestExecutionResult:
    """Tests for ExecutionResult."""
    
    def test_success(self):
        result = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
        )
        assert result.success is True
        assert result.exit_code == 0
    
    def test_failure(self):
        result = ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error occurred",
            error="Command failed",
        )
        assert result.success is False
        assert result.error == "Command failed"
    
    def test_to_dict(self):
        result = ExecutionResult(success=True, exit_code=0)
        data = result.to_dict()
        assert data["success"] is True
        assert data["exitCode"] == 0


class TestSandbox:
    """Tests for Sandbox."""
    
    def test_init(self):
        sandbox = Sandbox()
        assert sandbox.config is not None
    
    def test_init_with_config(self):
        config = SandboxConfig(max_execution_seconds=30.0)
        sandbox = Sandbox(config=config)
        assert sandbox.config.max_execution_seconds == 30.0
    
    @pytest.mark.asyncio
    async def test_execute_python(self):
        sandbox = Sandbox()
        result = await sandbox.execute(
            code="print('Hello from sandbox')",
            language="python",
            timeout=10.0,
        )
        assert result.success is True
        assert "Hello from sandbox" in result.stdout
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        sandbox = Sandbox()
        result = await sandbox.execute(
            code="import time; time.sleep(10)",
            language="python",
            timeout=1.0,
        )
        assert result.success is False
        assert "timed out" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_error(self):
        sandbox = Sandbox()
        result = await sandbox.execute(
            code="raise ValueError('test error')",
            language="python",
            timeout=10.0,
        )
        assert result.success is False
        assert result.exit_code != 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        sandbox = Sandbox()
        await sandbox.execute(code="print('test')", language="python")
        await sandbox.cleanup()
        assert len(sandbox._temp_dirs) == 0


class TestArchiveEntry:
    """Tests for ArchiveEntry."""
    
    def test_create(self):
        entry = ArchiveEntry(
            mutation_type="prompt_optimization",
            mutation_description="Improved planning prompt",
            baseline_score=0.75,
            mutated_score=0.82,
            improvement=0.07,
            accepted=True,
        )
        assert entry.mutation_type == "prompt_optimization"
        assert entry.accepted is True
    
    def test_to_dict(self):
        entry = ArchiveEntry(
            mutation_type="config_tuning",
            baseline_score=0.80,
            mutated_score=0.85,
            improvement=0.05,
            accepted=True,
        )
        data = entry.to_dict()
        assert data["mutationType"] == "config_tuning"
        assert data["improvement"] == 0.05
    
    def test_from_dict(self):
        data = {
            "mutationType": "tool_improvement",
            "baselineScore": 0.70,
            "mutatedScore": 0.75,
            "improvement": 0.05,
            "accepted": True,
        }
        entry = ArchiveEntry.from_dict(data)
        assert entry.mutation_type == "tool_improvement"
        assert entry.improvement == 0.05


class TestEvolutionArchive:
    """Tests for EvolutionArchive."""
    
    def test_init(self):
        archive = EvolutionArchive()
        assert archive.count == 0
    
    def test_init_with_max_entries(self):
        archive = EvolutionArchive(max_entries=100)
        assert archive.max_entries == 100
    
    @pytest.mark.asyncio
    async def test_add_entry(self):
        archive = EvolutionArchive()
        
        entry = ArchiveEntry(
            mutation_type="test",
            accepted=True,
        )
        await archive.add(entry)
        
        assert archive.count == 1
    
    @pytest.mark.asyncio
    async def test_find_similar(self):
        archive = EvolutionArchive()
        
        entry1 = ArchiveEntry(
            mutation_type="prompt_optimization",
            target_function="plan",
            accepted=True,
        )
        entry2 = ArchiveEntry(
            mutation_type="config_tuning",
            target_function="other",
            accepted=False,
        )
        
        await archive.add(entry1)
        await archive.add(entry2)
        
        similar = archive.find_similar(mutation_type="prompt_optimization")
        assert len(similar) == 1
        assert similar[0].mutation_type == "prompt_optimization"
    
    def test_was_tried(self):
        archive = EvolutionArchive()
        archive._entries[uuid4()] = ArchiveEntry(
            mutation_type="prompt_optimization",
            target_function="plan",
            mutated_code="new prompt content here",
        )
        
        # Should find similar
        result = archive.was_tried(
            mutation_type="prompt_optimization",
            target_function="plan",
            code_snippet="new prompt",
        )
        assert result is True
        
        # Should not find
        result = archive.was_tried(
            mutation_type="prompt_optimization",
            target_function="plan",
            code_snippet="completely different",
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_successful_mutations(self):
        archive = EvolutionArchive()
        
        await archive.add(ArchiveEntry(
            mutation_type="test",
            improvement=0.10,
            accepted=True,
        ))
        await archive.add(ArchiveEntry(
            mutation_type="test",
            improvement=0.05,
            accepted=True,
        ))
        await archive.add(ArchiveEntry(
            mutation_type="test",
            improvement=-0.02,
            accepted=False,
        ))
        
        successful = archive.get_successful_mutations(min_improvement=0.05)
        assert len(successful) == 2
    
    def test_get_stats(self):
        archive = EvolutionArchive()
        archive._entries[uuid4()] = ArchiveEntry(accepted=True, improvement=0.1)
        archive._entries[uuid4()] = ArchiveEntry(accepted=True, improvement=0.2)
        archive._entries[uuid4()] = ArchiveEntry(accepted=False)
        
        stats = archive.get_stats()
        assert stats["total_entries"] == 3
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["acceptance_rate"] == 2/3


class TestSICAEngine:
    """Tests for SICAEngine."""
    
    def test_init(self):
        engine = SICAEngine()
        assert engine is not None
        assert engine.archive is not None
        assert engine.sandbox is not None
    
    def test_init_disabled(self):
        config = EvolutionConfig()
        config.enable_self_improvement = False
        engine = SICAEngine(config=config)
        assert engine.config.enable_self_improvement is False
    
    @pytest.mark.asyncio
    async def test_improve_disabled(self):
        config = EvolutionConfig()
        config.enable_self_improvement = False
        engine = SICAEngine(config=config)
        
        result = await engine.improve(
            agent=MagicMock(),
            test_cases=[],
        )
        
        assert result.accepted is False
        assert "disabled" in result.rejected_reason.lower()
