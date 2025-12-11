"""
L5: Evolution Layer

Provides self-improvement capabilities for agents:
- SICA (Self-Improving Coding Agent): Code and prompt optimization
- ADAS (Automated Design of Agentic Systems): Architecture search
- Sandboxed execution for safe experimentation
"""

from squadron.evolution.sica import (
    SICAEngine,
    Mutation,
    MutationType,
    ImprovementResult,
)
from squadron.evolution.sandbox import (
    Sandbox,
    SandboxConfig,
    ExecutionResult,
)
from squadron.evolution.archive import (
    ArchiveEntry,
    EvolutionArchive,
)

__all__ = [
    # SICA
    "SICAEngine",
    "Mutation",
    "MutationType",
    "ImprovementResult",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "ExecutionResult",
    # Archive
    "ArchiveEntry",
    "EvolutionArchive",
]
