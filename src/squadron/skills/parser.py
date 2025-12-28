"""
SKILL.md Parser

Parses SKILL.md files following the agentskills.io specification.
Handles YAML frontmatter and markdown content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from squadron.skills.models import Skill, SkillMetadata

logger = structlog.get_logger(__name__)

# Regex to match YAML frontmatter
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?(.*)$",
    re.DOTALL
)


def parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from SKILL.md content.
    
    Returns:
        Tuple of (metadata_dict, body_content)
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        logger.warning("No YAML frontmatter found in SKILL.md")
        return {}, content
    
    yaml_content = match.group(1)
    body_content = match.group(2).strip()
    
    # Simple YAML parsing (avoiding external dependency)
    metadata = {}
    current_key = None
    current_value_lines = []
    
    for line in yaml_content.split("\n"):
        # Skip empty lines
        if not line.strip():
            continue
            
        # Check for key: value pattern
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            # Save previous key if exists
            if current_key:
                value = "\n".join(current_value_lines).strip()
                metadata[current_key] = _parse_yaml_value(value)
            
            # Start new key
            key, _, value = line.partition(":")
            current_key = key.strip()
            current_value_lines = [value.strip()] if value.strip() else []
        elif current_key:
            # Continue multi-line value
            current_value_lines.append(line)
    
    # Save last key
    if current_key:
        value = "\n".join(current_value_lines).strip()
        metadata[current_key] = _parse_yaml_value(value)
    
    return metadata, body_content


def _parse_yaml_value(value: str) -> Any:
    """Parse a YAML value into appropriate Python type."""
    if not value:
        return None
    
    # Handle lists (simple inline format)
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip().strip("'\"") for item in items if item.strip()]
    
    # Handle booleans
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    
    # Handle numbers
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # Handle quoted strings
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    return value


def parse_skill_md(path: Path) -> Skill | None:
    """
    Parse a SKILL.md file into a Skill object.
    
    Args:
        path: Path to the SKILL.md file
        
    Returns:
        Skill object with metadata and instructions, or None if invalid
    """
    if not path.exists():
        logger.error("SKILL.md not found", path=str(path))
        return None
    
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read SKILL.md", path=str(path), error=str(e))
        return None
    
    metadata_dict, body = parse_yaml_frontmatter(content)
    
    # Validate required fields
    name = metadata_dict.get("name")
    description = metadata_dict.get("description")
    
    if not name:
        logger.error("SKILL.md missing required 'name' field", path=str(path))
        return None
    
    if not description:
        logger.error("SKILL.md missing required 'description' field", path=str(path))
        return None
    
    # Validate name format (lowercase, hyphens, no reserved words)
    if not _validate_skill_name(name):
        logger.error("Invalid skill name", name=name, path=str(path))
        return None
    
    # Build metadata
    metadata = SkillMetadata(
        name=name,
        description=description,
        short_description=metadata_dict.get("short-description") or metadata_dict.get("short_description"),
        version=metadata_dict.get("version"),
        author=metadata_dict.get("author"),
        tags=metadata_dict.get("tags", []),
    )
    
    # Create skill with Level 1 (metadata) and Level 2 (instructions)
    return Skill(
        metadata=metadata,
        path=path.parent,
        instructions=body if body else None,
    )


def _validate_skill_name(name: str) -> bool:
    """
    Validate skill name per agentskills.io spec.
    
    - Maximum 64 characters
    - Only lowercase letters, numbers, and hyphens
    - Cannot contain reserved words
    """
    if len(name) > 64:
        return False
    
    if not re.match(r"^[a-z0-9-]+$", name):
        return False
    
    reserved_words = ["anthropic", "claude"]
    for word in reserved_words:
        if word in name.lower():
            return False
    
    return True


def load_skill_metadata_only(path: Path) -> SkillMetadata | None:
    """
    Load only Level 1 (metadata) from a SKILL.md file.
    
    This is more efficient for initial discovery when we don't
    need the full instructions yet.
    """
    if not path.exists():
        return None
    
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    
    metadata_dict, _ = parse_yaml_frontmatter(content)
    
    name = metadata_dict.get("name")
    description = metadata_dict.get("description")
    
    if not name or not description:
        return None
    
    if not _validate_skill_name(name):
        return None
    
    return SkillMetadata(
        name=name,
        description=description,
        short_description=metadata_dict.get("short-description") or metadata_dict.get("short_description"),
        version=metadata_dict.get("version"),
        author=metadata_dict.get("author"),
        tags=metadata_dict.get("tags", []),
    )
