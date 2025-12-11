"""
Evolution Archive

Maintains a history of agent mutations and their performance.
Prevents cyclical regression by tracking what has been tried.

Based on the ADAS (Automated Design of Agentic Systems) archive mechanism.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ArchiveEntry:
    """
    An entry in the evolution archive.
    
    Records a mutation attempt and its results.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Mutation details
    mutation_type: str = ""
    mutation_description: str = ""
    
    # Code changes
    original_code: str = ""
    mutated_code: str = ""
    diff: str = ""
    
    # Target
    target_file: str = ""
    target_function: str = ""
    
    # Performance
    baseline_score: float = 0.0
    mutated_score: float = 0.0
    improvement: float = 0.0
    
    # Evaluation details
    eval_metrics: dict[str, float] = field(default_factory=dict)
    test_results: dict[str, bool] = field(default_factory=dict)
    
    # Status
    accepted: bool = False
    rejected_reason: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    agent_version: str = ""
    
    # Tags for filtering
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "mutationType": self.mutation_type,
            "mutationDescription": self.mutation_description,
            "originalCode": self.original_code,
            "mutatedCode": self.mutated_code,
            "diff": self.diff,
            "targetFile": self.target_file,
            "targetFunction": self.target_function,
            "baselineScore": self.baseline_score,
            "mutatedScore": self.mutated_score,
            "improvement": self.improvement,
            "evalMetrics": self.eval_metrics,
            "testResults": self.test_results,
            "accepted": self.accepted,
            "rejectedReason": self.rejected_reason,
            "createdAt": self.created_at.isoformat(),
            "agentVersion": self.agent_version,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchiveEntry:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            mutation_type=data.get("mutationType", ""),
            mutation_description=data.get("mutationDescription", ""),
            original_code=data.get("originalCode", ""),
            mutated_code=data.get("mutatedCode", ""),
            diff=data.get("diff", ""),
            target_file=data.get("targetFile", ""),
            target_function=data.get("targetFunction", ""),
            baseline_score=data.get("baselineScore", 0.0),
            mutated_score=data.get("mutatedScore", 0.0),
            improvement=data.get("improvement", 0.0),
            eval_metrics=data.get("evalMetrics", {}),
            test_results=data.get("testResults", {}),
            accepted=data.get("accepted", False),
            rejected_reason=data.get("rejectedReason", ""),
            agent_version=data.get("agentVersion", ""),
            tags=data.get("tags", []),
        )


class EvolutionArchive:
    """
    Evolution Archive - Tracks mutation history.
    
    Provides:
    - Persistent storage of mutation attempts
    - Duplicate detection to prevent retrying failed mutations
    - Performance tracking over time
    - Rollback capabilities
    
    Example:
        ```python
        archive = EvolutionArchive(storage_path="evolution_archive.json")
        await archive.load()
        
        # Record a mutation
        entry = ArchiveEntry(
            mutation_type="prompt_optimization",
            mutation_description="Improved planning prompt",
            baseline_score=0.75,
            mutated_score=0.82,
            improvement=0.07,
            accepted=True,
        )
        await archive.add(entry)
        
        # Check if similar mutation was tried
        similar = archive.find_similar(
            mutation_type="prompt_optimization",
            target_function="plan",
        )
        ```
    """
    
    def __init__(
        self,
        storage_path: str | Path | None = None,
        max_entries: int = 1000,
    ):
        """
        Initialize the archive.
        
        Args:
            storage_path: Path to persist the archive
            max_entries: Maximum entries to keep
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_entries = max_entries
        
        self._entries: dict[UUID, ArchiveEntry] = {}
        self._loaded = False
    
    async def load(self) -> None:
        """Load the archive from storage."""
        if not self.storage_path or not self.storage_path.exists():
            self._loaded = True
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for entry_data in data.get("entries", []):
                entry = ArchiveEntry.from_dict(entry_data)
                self._entries[entry.id] = entry
            
            logger.info("Archive loaded", entries=len(self._entries))
            
        except Exception as e:
            logger.error("Failed to load archive", error=str(e))
        
        self._loaded = True
    
    async def save(self) -> None:
        """Save the archive to storage."""
        if not self.storage_path:
            return
        
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "savedAt": datetime.utcnow().isoformat(),
                "entries": [e.to_dict() for e in self._entries.values()],
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Archive saved", entries=len(self._entries))
            
        except Exception as e:
            logger.error("Failed to save archive", error=str(e))
    
    async def add(self, entry: ArchiveEntry) -> None:
        """
        Add an entry to the archive.
        
        Args:
            entry: The entry to add
        """
        self._entries[entry.id] = entry
        
        # Prune if over limit
        if len(self._entries) > self.max_entries:
            await self._prune()
        
        # Auto-save
        await self.save()
        
        logger.debug(
            "Added archive entry",
            id=str(entry.id),
            type=entry.mutation_type,
            accepted=entry.accepted,
        )
    
    async def _prune(self) -> None:
        """Remove oldest entries to stay under limit."""
        if len(self._entries) <= self.max_entries:
            return
        
        # Sort by creation time
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
        )
        
        # Remove oldest (keep accepted ones longer)
        to_remove = len(self._entries) - self.max_entries
        removed = 0
        
        for entry in sorted_entries:
            if removed >= to_remove:
                break
            
            # Prefer removing rejected entries
            if not entry.accepted:
                del self._entries[entry.id]
                removed += 1
        
        # If still over, remove oldest regardless
        for entry in sorted_entries:
            if removed >= to_remove:
                break
            if entry.id in self._entries:
                del self._entries[entry.id]
                removed += 1
        
        logger.debug("Pruned archive", removed=removed)
    
    def find_similar(
        self,
        mutation_type: str | None = None,
        target_file: str | None = None,
        target_function: str | None = None,
        code_hash: str | None = None,
    ) -> list[ArchiveEntry]:
        """
        Find similar mutations in the archive.
        
        Args:
            mutation_type: Type of mutation
            target_file: Target file
            target_function: Target function
            code_hash: Hash of the mutated code
            
        Returns:
            List of similar entries
        """
        results = []
        
        for entry in self._entries.values():
            match = True
            
            if mutation_type and entry.mutation_type != mutation_type:
                match = False
            if target_file and entry.target_file != target_file:
                match = False
            if target_function and entry.target_function != target_function:
                match = False
            
            if match:
                results.append(entry)
        
        return results
    
    def was_tried(
        self,
        mutation_type: str,
        target_function: str,
        code_snippet: str,
    ) -> bool:
        """
        Check if a similar mutation was already tried.
        
        Args:
            mutation_type: Type of mutation
            target_function: Target function
            code_snippet: Key code snippet
            
        Returns:
            True if a similar mutation was tried
        """
        for entry in self._entries.values():
            if entry.mutation_type != mutation_type:
                continue
            if entry.target_function != target_function:
                continue
            
            # Check if code is similar (simple substring check)
            if code_snippet in entry.mutated_code:
                return True
        
        return False
    
    def get_successful_mutations(
        self,
        mutation_type: str | None = None,
        min_improvement: float = 0.0,
    ) -> list[ArchiveEntry]:
        """
        Get successful mutations.
        
        Args:
            mutation_type: Filter by type
            min_improvement: Minimum improvement threshold
            
        Returns:
            List of successful entries
        """
        results = []
        
        for entry in self._entries.values():
            if not entry.accepted:
                continue
            if entry.improvement < min_improvement:
                continue
            if mutation_type and entry.mutation_type != mutation_type:
                continue
            
            results.append(entry)
        
        # Sort by improvement
        results.sort(key=lambda e: e.improvement, reverse=True)
        
        return results
    
    def get_failed_mutations(
        self,
        mutation_type: str | None = None,
    ) -> list[ArchiveEntry]:
        """
        Get failed mutations.
        
        Args:
            mutation_type: Filter by type
            
        Returns:
            List of failed entries
        """
        results = []
        
        for entry in self._entries.values():
            if entry.accepted:
                continue
            if mutation_type and entry.mutation_type != mutation_type:
                continue
            
            results.append(entry)
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get archive statistics."""
        total = len(self._entries)
        accepted = sum(1 for e in self._entries.values() if e.accepted)
        rejected = total - accepted
        
        improvements = [
            e.improvement for e in self._entries.values()
            if e.accepted and e.improvement > 0
        ]
        
        return {
            "total_entries": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "max_improvement": max(improvements) if improvements else 0,
            "mutation_types": list(set(e.mutation_type for e in self._entries.values())),
        }
    
    def get_entry(self, entry_id: UUID) -> ArchiveEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)
    
    def get_latest(self, n: int = 10) -> list[ArchiveEntry]:
        """Get the N most recent entries."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return sorted_entries[:n]
    
    @property
    def entries(self) -> list[ArchiveEntry]:
        """Get all entries."""
        return list(self._entries.values())
    
    @property
    def count(self) -> int:
        """Get entry count."""
        return len(self._entries)
