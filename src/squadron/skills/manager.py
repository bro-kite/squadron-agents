"""
Skills Manager

Manages Agent Skills discovery, loading, and retrieval following
the agentskills.io specification with progressive disclosure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from squadron.skills.models import Skill, SkillMetadata, SkillMatch
from squadron.skills.parser import parse_skill_md, load_skill_metadata_only

logger = structlog.get_logger(__name__)

# Default skills directories following the standard
DEFAULT_SKILLS_DIRS = [
    ".github/skills",    # GitHub Copilot standard
    ".claude/skills",    # Claude/Anthropic standard
    ".squadron/skills",  # Squadron-specific
    "skills",            # Generic fallback
]


class SkillsManager:
    """
    Manages Agent Skills with progressive disclosure.
    
    Level 1 (Discovery): Loads name/description from all skills at startup
    Level 2 (Instructions): Loads full SKILL.md when skill is triggered
    Level 3 (Resources): Loads additional files as needed
    
    Example:
        ```python
        manager = SkillsManager(workspace_path="/path/to/project")
        await manager.discover_skills()
        
        # Find relevant skills for a task
        matches = manager.find_skills("create a PowerPoint presentation")
        
        # Load full skill content
        skill = await manager.load_skill("powerpoint-creation")
        ```
    """
    
    def __init__(
        self,
        workspace_path: str | Path | None = None,
        skills_dirs: list[str] | None = None,
        max_skills_in_context: int = 5,
    ):
        """
        Initialize the Skills Manager.
        
        Args:
            workspace_path: Root workspace directory to search for skills
            skills_dirs: Custom skills directories to search (relative to workspace)
            max_skills_in_context: Maximum skills to include in LLM context
        """
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.skills_dirs = skills_dirs or DEFAULT_SKILLS_DIRS
        self.max_skills_in_context = max_skills_in_context
        
        # Level 1: Metadata index (always loaded)
        self._metadata_index: dict[str, SkillMetadata] = {}
        self._skill_paths: dict[str, Path] = {}
        
        # Level 2: Loaded skills cache
        self._loaded_skills: dict[str, Skill] = {}
        
        logger.info(
            "SkillsManager initialized",
            workspace=str(self.workspace_path),
            skills_dirs=self.skills_dirs,
        )
    
    async def discover_skills(self) -> int:
        """
        Discover all available skills (Level 1 loading).
        
        Scans configured directories for SKILL.md files and loads
        only their metadata (name, description).
        
        Returns:
            Number of skills discovered
        """
        self._metadata_index.clear()
        self._skill_paths.clear()
        
        for skills_dir in self.skills_dirs:
            dir_path = self.workspace_path / skills_dir
            if not dir_path.exists():
                continue
            
            logger.debug("Scanning skills directory", path=str(dir_path))
            
            # Each subdirectory could be a skill
            for item in dir_path.iterdir():
                if not item.is_dir():
                    continue
                
                skill_md_path = item / "SKILL.md"
                if not skill_md_path.exists():
                    continue
                
                metadata = load_skill_metadata_only(skill_md_path)
                if metadata:
                    self._metadata_index[metadata.name] = metadata
                    self._skill_paths[metadata.name] = skill_md_path
                    logger.debug(
                        "Discovered skill",
                        name=metadata.name,
                        description=metadata.description[:50],
                    )
        
        logger.info("Skills discovery complete", count=len(self._metadata_index))
        return len(self._metadata_index)
    
    def find_skills(
        self,
        query: str,
        threshold: float = 0.1,
        max_results: int | None = None,
    ) -> list[SkillMatch]:
        """
        Find skills relevant to a query.
        
        Uses semantic matching against skill names and descriptions.
        
        Args:
            query: Search query or task description
            threshold: Minimum relevance score (0.0-1.0)
            max_results: Maximum number of results (defaults to max_skills_in_context)
            
        Returns:
            List of SkillMatch objects sorted by relevance
        """
        max_results = max_results or self.max_skills_in_context
        matches = []
        
        for name, metadata in self._metadata_index.items():
            score = metadata.matches_query(query)
            if score >= threshold:
                # Use cached loaded skill if available, otherwise create lightweight object
                if name in self._loaded_skills:
                    skill = self._loaded_skills[name]
                else:
                    skill = Skill(
                        metadata=metadata,
                        path=self._skill_paths[name].parent,
                    )
                matches.append(SkillMatch(skill=skill, score=score))
        
        # Sort by score descending
        matches.sort(reverse=True, key=lambda m: m.score)
        
        return matches[:max_results]
    
    async def load_skill(self, skill_name: str) -> Skill | None:
        """
        Load a skill's full content (Level 2 loading).
        
        Args:
            skill_name: Name of the skill to load
            
        Returns:
            Fully loaded Skill object or None if not found
        """
        # Check cache first
        if skill_name in self._loaded_skills:
            return self._loaded_skills[skill_name]
        
        # Check if skill exists
        if skill_name not in self._skill_paths:
            logger.warning("Skill not found", name=skill_name)
            return None
        
        skill_md_path = self._skill_paths[skill_name]
        skill = parse_skill_md(skill_md_path)
        
        if skill:
            self._loaded_skills[skill_name] = skill
            logger.info("Loaded skill", name=skill_name)
        
        return skill
    
    async def load_skill_resource(
        self,
        skill_name: str,
        resource_path: str,
    ) -> str | None:
        """
        Load a specific resource from a skill (Level 3 loading).
        
        Args:
            skill_name: Name of the skill
            resource_path: Relative path to the resource within the skill
            
        Returns:
            Resource content as string or None if not found
        """
        skill = await self.load_skill(skill_name)
        if not skill:
            return None
        
        # Check cache
        if resource_path in skill.resources:
            return skill.resources[resource_path]
        
        # Security: Validate path to prevent traversal attacks
        full_path = (skill.path / resource_path).resolve()
        skill_path_resolved = skill.path.resolve()
        
        if not full_path.is_relative_to(skill_path_resolved):
            logger.warning(
                "Path traversal attempt blocked",
                skill=skill_name,
                resource=resource_path,
            )
            return None
        
        if not full_path.exists():
            logger.warning(
                "Skill resource not found",
                skill=skill_name,
                resource=resource_path,
            )
            return None
        
        try:
            content = full_path.read_text(encoding="utf-8")
            skill.resources[resource_path] = content
            logger.debug(
                "Loaded skill resource",
                skill=skill_name,
                resource=resource_path,
            )
            return content
        except Exception as e:
            logger.error(
                "Failed to load skill resource",
                skill=skill_name,
                resource=resource_path,
                error=str(e),
            )
            return None
    
    def get_skills_summary(self) -> str:
        """
        Get a summary of all available skills for LLM context.
        
        Returns Level 1 information only (name and description).
        """
        if not self._metadata_index:
            return "No skills available."
        
        lines = ["## Available Skills"]
        for name, metadata in sorted(self._metadata_index.items()):
            lines.append(f"- **{name}**: {metadata.description}")
        
        return "\n".join(lines)
    
    def get_skills_for_context(
        self,
        query: str,
        include_instructions: bool = True,
    ) -> str:
        """
        Get formatted skills context for LLM prompt.
        
        Finds relevant skills and formats them for inclusion in
        the action generation prompt.
        
        Args:
            query: Task or query to match skills against
            include_instructions: Whether to include Level 2 content
            
        Returns:
            Formatted string for LLM context
        """
        matches = self.find_skills(query)
        
        if not matches:
            return ""
        
        lines = ["## Relevant Skills"]
        
        for match in matches:
            skill = match.skill
            lines.append(f"\n### {skill.name}")
            lines.append(f"**Description**: {skill.description}")
            lines.append(f"**Relevance**: {match.score:.2f}")
            
            if include_instructions and skill.is_loaded:
                lines.append(f"\n**Instructions**:\n{skill.instructions}")
        
        return "\n".join(lines)
    
    @property
    def available_skills(self) -> list[str]:
        """List names of all discovered skills."""
        return list(self._metadata_index.keys())
    
    @property
    def skill_count(self) -> int:
        """Number of discovered skills."""
        return len(self._metadata_index)
    
    def get_skill_metadata(self, skill_name: str) -> SkillMetadata | None:
        """Get metadata for a specific skill."""
        return self._metadata_index.get(skill_name)
    
    def clear_cache(self) -> None:
        """Clear the loaded skills cache."""
        self._loaded_skills.clear()
        logger.debug("Skills cache cleared")
