"""
Agent Skills Data Models

Defines the core data structures for Agent Skills following the
agentskills.io specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillMetadata:
    """
    Level 1: Skill metadata from YAML frontmatter.
    
    Always loaded at startup for skill discovery.
    """
    name: str
    description: str
    short_description: str | None = None
    version: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> float:
        """
        Score how well this skill matches a query.
        
        Returns a relevance score between 0.0 and 1.0.
        """
        query_lower = query.lower()
        score = 0.0
        
        # Check name match
        if self.name.lower() in query_lower:
            score += 0.4
        
        # Check description match
        desc_words = self.description.lower().split()
        query_words = set(query_lower.split())
        matching_words = sum(1 for w in desc_words if w in query_words)
        if desc_words:
            score += 0.4 * (matching_words / len(desc_words))
        
        # Check tags match
        for tag in self.tags:
            if tag.lower() in query_lower:
                score += 0.2
                break
        
        return min(1.0, score)


@dataclass
class Skill:
    """
    Full Skill representation with all levels of content.
    
    Levels:
    - Level 1: Metadata (always loaded)
    - Level 2: Instructions (loaded when triggered)
    - Level 3: Resources (loaded as needed)
    """
    metadata: SkillMetadata
    path: Path
    instructions: str | None = None  # Level 2: SKILL.md body
    resources: dict[str, str] = field(default_factory=dict)  # Level 3: additional files
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def description(self) -> str:
        return self.metadata.description
    
    @property
    def is_loaded(self) -> bool:
        """Check if Level 2 instructions have been loaded."""
        return self.instructions is not None
    
    def get_resource_paths(self) -> list[Path]:
        """Get paths to all resources in the skill directory."""
        if not self.path.exists():
            return []
        
        resources = []
        for item in self.path.iterdir():
            if item.name != "SKILL.md" and item.is_file():
                resources.append(item)
            elif item.is_dir():
                # Include files from subdirectories
                for subitem in item.rglob("*"):
                    if subitem.is_file():
                        resources.append(subitem)
        return resources
    
    def to_context_string(self, include_resources: bool = False) -> str:
        """
        Format skill for inclusion in LLM context.
        
        Args:
            include_resources: Whether to include Level 3 resources
        """
        parts = [
            f"## Skill: {self.name}",
            f"**Description**: {self.description}",
        ]
        
        if self.instructions:
            parts.append(f"\n### Instructions\n{self.instructions}")
        
        if include_resources and self.resources:
            parts.append("\n### Resources")
            for name, content in self.resources.items():
                parts.append(f"\n#### {name}\n```\n{content}\n```")
        
        return "\n".join(parts)


@dataclass
class SkillMatch:
    """A skill that matched a query with its relevance score."""
    skill: Skill
    score: float
    
    def __lt__(self, other: SkillMatch) -> bool:
        return self.score < other.score
