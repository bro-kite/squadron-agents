#!/usr/bin/env python3
"""
Self-Improvement (SICA) Example

Demonstrates the Self-Improving Coding Agent (SICA) system:
- Creating test cases for evaluation
- Running the improvement loop
- Sandbox execution for safe testing
- Evolution archive for tracking mutations
"""

import asyncio
from pathlib import Path
import tempfile

from squadron import Agent, SquadronConfig, SICAEngine, Sandbox, EvolutionArchive
from squadron.governance.evaluator import AgentEvaluator, TestCase, EvalMetric
from squadron.evolution.sica import Mutation, MutationType, ImprovementResult
from squadron.evolution.sandbox import SandboxConfig, SandboxType
from squadron.memory import GraphitiMemory
from squadron.reasoning import LATSReasoner


# =============================================================================
# Example: Creating Test Cases
# =============================================================================

async def example_test_cases():
    """Example creating test cases for agent evaluation."""
    print("\n" + "=" * 50)
    print("📋 Test Cases Example")
    print("=" * 50)
    
    # Define test cases for evaluating an agent
    test_cases = [
        TestCase(
            name="file_read",
            task="Read the contents of README.md",
            expected_tools=["read_file"],
            expected_output_contains=["README", "project"],
            timeout_seconds=30.0,
        ),
        TestCase(
            name="code_search",
            task="Find all functions that start with 'test_'",
            expected_tools=["grep", "find_files"],
            timeout_seconds=60.0,
        ),
        TestCase(
            name="file_write",
            task="Create a new file called hello.py with a hello world function",
            expected_tools=["write_file"],
            expected_output_contains=["hello", "def"],
            timeout_seconds=30.0,
        ),
        TestCase(
            name="analysis",
            task="Analyze the project structure and list all Python files",
            expected_tools=["list_directory", "find_files"],
            timeout_seconds=45.0,
        ),
        TestCase(
            name="multi_step",
            task="Read the main.py file, find any TODO comments, and create a summary",
            expected_tools=["read_file", "grep"],
            timeout_seconds=90.0,
        ),
    ]
    
    print(f"\nCreated {len(test_cases)} test cases:")
    for tc in test_cases:
        print(f"\n  📝 {tc.name}")
        print(f"     Task: {tc.task[:50]}...")
        print(f"     Expected tools: {tc.expected_tools}")
        print(f"     Timeout: {tc.timeout_seconds}s")
    
    return test_cases


# =============================================================================
# Example: Agent Evaluation
# =============================================================================

async def example_evaluation():
    """Example running agent evaluation."""
    print("\n" + "=" * 50)
    print("📊 Agent Evaluation Example")
    print("=" * 50)
    
    # Create evaluator
    evaluator = AgentEvaluator()
    
    print("\nAvailable metrics:")
    for metric in EvalMetric:
        print(f"  - {metric.value}")
    
    # Simulate evaluation results
    print("\nExample evaluation results:")
    print("""
    Test Case: file_read
    ├── Task Completion: 0.95 ✅
    ├── Tool Correctness: 1.00 ✅
    ├── Efficiency: 0.80 ✅
    └── Overall: 0.92 ✅
    
    Test Case: code_search
    ├── Task Completion: 0.85 ✅
    ├── Tool Correctness: 0.50 ⚠️ (used wrong tool)
    ├── Efficiency: 0.70 ✅
    └── Overall: 0.68 ⚠️
    
    Suite Summary:
    ├── Total Tests: 5
    ├── Passed: 4
    ├── Failed: 1
    └── Average Score: 0.82
    """)


# =============================================================================
# Example: Sandbox Execution
# =============================================================================

async def example_sandbox():
    """Example using sandbox for safe code execution."""
    print("\n" + "=" * 50)
    print("🔒 Sandbox Execution Example")
    print("=" * 50)
    
    # Create sandbox with subprocess isolation
    sandbox = Sandbox(config=SandboxConfig(
        sandbox_type=SandboxType.SUBPROCESS,
        max_execution_seconds=10.0,
        max_memory_mb=256,
    ))
    
    print("\nSandbox configuration:")
    print(f"  Type: {sandbox.config.sandbox_type.value}")
    print(f"  Timeout: {sandbox.config.max_execution_seconds}s")
    print(f"  Memory limit: {sandbox.config.max_memory_mb}MB")
    
    # Execute safe code
    print("\n🟢 Executing safe code:")
    result = await sandbox.execute(
        code="print('Hello from sandbox!')\nprint(2 + 2)",
        language="python",
    )
    print(f"  Success: {result.success}")
    print(f"  Output: {result.stdout.strip()}")
    
    # Execute code with error
    print("\n🔴 Executing code with error:")
    result = await sandbox.execute(
        code="raise ValueError('Test error')",
        language="python",
    )
    print(f"  Success: {result.success}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Error: {result.stderr[:100]}...")
    
    # Execute code with timeout
    print("\n⏱️ Executing code with timeout:")
    result = await sandbox.execute(
        code="import time; time.sleep(100)",
        language="python",
        timeout=2.0,
    )
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    
    # Cleanup
    await sandbox.cleanup()
    print("\n✅ Sandbox cleaned up")


# =============================================================================
# Example: Evolution Archive
# =============================================================================

async def example_archive():
    """Example using evolution archive to track mutations."""
    print("\n" + "=" * 50)
    print("📚 Evolution Archive Example")
    print("=" * 50)
    
    # Create archive
    archive = EvolutionArchive(max_entries=1000)
    
    print(f"\nArchive created (max entries: {archive.max_entries})")
    
    # Add some mutation entries
    from squadron.evolution.archive import ArchiveEntry
    
    entries = [
        ArchiveEntry(
            mutation_type="prompt_optimization",
            mutation_description="Improved planning prompt clarity",
            target_function="plan",
            baseline_score=0.75,
            mutated_score=0.82,
            improvement=0.07,
            accepted=True,
        ),
        ArchiveEntry(
            mutation_type="config_tuning",
            mutation_description="Increased exploration constant",
            target_function="mcts_search",
            baseline_score=0.80,
            mutated_score=0.78,
            improvement=-0.02,
            accepted=False,
        ),
        ArchiveEntry(
            mutation_type="tool_improvement",
            mutation_description="Added retry logic to file operations",
            target_function="read_file",
            baseline_score=0.85,
            mutated_score=0.92,
            improvement=0.07,
            accepted=True,
        ),
    ]
    
    for entry in entries:
        await archive.add(entry)
    
    print(f"\nAdded {len(entries)} entries to archive")
    
    # Get statistics
    stats = archive.get_stats()
    print(f"\nArchive statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.0%}")
    
    # Find successful mutations
    successful = archive.get_successful_mutations(min_improvement=0.05)
    print(f"\nSuccessful mutations (>5% improvement): {len(successful)}")
    for entry in successful:
        print(f"  - {entry.mutation_type}: +{entry.improvement:.0%}")
    
    # Check if mutation was tried before
    was_tried = archive.was_tried(
        mutation_type="prompt_optimization",
        target_function="plan",
        code_snippet="planning prompt",
    )
    print(f"\nSimilar mutation tried before: {was_tried}")


# =============================================================================
# Example: SICA Engine
# =============================================================================

async def example_sica_engine():
    """Example using the SICA self-improvement engine."""
    print("\n" + "=" * 50)
    print("🧬 SICA Engine Example")
    print("=" * 50)
    
    # Create SICA engine
    from squadron.core.config import EvolutionConfig
    
    config = EvolutionConfig()
    config.enable_self_improvement = True
    config.min_improvement_threshold = 0.05
    config.max_mutations_per_cycle = 3
    
    engine = SICAEngine(
        config=config,
        evaluator=AgentEvaluator(),
    )
    
    print("\nSICA Engine configuration:")
    print(f"  Self-improvement enabled: {config.enable_self_improvement}")
    print(f"  Min improvement threshold: {config.min_improvement_threshold:.0%}")
    print(f"  Max mutations per cycle: {config.max_mutations_per_cycle}")
    
    print("\nMutation types:")
    for mt in MutationType:
        print(f"  - {mt.value}")
    
    print("""
SICA Improvement Loop:

1. 📊 Evaluate baseline performance
   └── Run test suite, calculate scores

2. 🧬 Generate mutations
   ├── Prompt optimization
   ├── Tool improvement
   ├── Config tuning
   └── Code refactoring

3. 🔒 Test in sandbox
   └── Safe execution environment

4. 📈 Evaluate mutated version
   └── Compare against baseline

5. ✅ Accept or reject
   ├── Accept if improvement > threshold
   └── Reject if regression detected

6. 📚 Update archive
   └── Track mutation history

7. 🔄 Repeat
   └── Continue until no improvements
""")


# =============================================================================
# Example: Full Improvement Cycle
# =============================================================================

async def example_full_cycle():
    """Example of a full self-improvement cycle."""
    print("\n" + "=" * 50)
    print("🔄 Full Improvement Cycle Example")
    print("=" * 50)
    
    print("""
To run a full improvement cycle:

```python
from squadron import Agent, SICAEngine, AgentEvaluator
from squadron.governance.evaluator import TestCase

# Create agent
agent = Agent(name="my-agent", ...)

# Define test cases
test_cases = [
    TestCase(
        name="task1",
        task="Do something",
        expected_tools=["tool1"],
    ),
    # ... more test cases
]

# Create SICA engine
sica = SICAEngine(evaluator=AgentEvaluator())

# Run improvement
result = await sica.improve(
    agent=agent,
    test_cases=test_cases,
    max_iterations=10,
)

# Check results
if result.accepted:
    print(f"✅ Improvement accepted!")
    print(f"   Baseline: {result.baseline_score:.2%}")
    print(f"   Mutated: {result.mutated_score:.2%}")
    print(f"   Improvement: {result.improvement:.2%}")
    print(f"   Mutation: {result.mutation.mutation_type.value}")
else:
    print(f"❌ No improvement found")
    print(f"   Reason: {result.rejected_reason}")
```

The SICA engine will:
1. Evaluate the agent's current performance
2. Generate candidate mutations
3. Test each mutation in a sandbox
4. Accept mutations that improve performance
5. Track all attempts in the evolution archive
""")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all self-improvement examples."""
    print("🚀 Squadron Self-Improvement (SICA) Examples")
    print("=" * 50)
    
    try:
        await example_test_cases()
        await example_evaluation()
        await example_sandbox()
        await example_archive()
        await example_sica_engine()
        await example_full_cycle()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Self-improvement examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
