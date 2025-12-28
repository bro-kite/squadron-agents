"""
Agent Skills Module

Implements the Agent Skills standard (agentskills.io) for Squadron Agents.
Skills are modular capabilities packaged as directories with SKILL.md files.
"""

from squadron.skills.manager import SkillsManager
from squadron.skills.models import Skill, SkillMetadata

__all__ = ["SkillsManager", "Skill", "SkillMetadata"]
