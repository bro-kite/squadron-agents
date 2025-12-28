"""
Tests for Agent Skills functionality.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from squadron.skills.models import Skill, SkillMetadata, SkillMatch
from squadron.skills.parser import (
    parse_yaml_frontmatter,
    parse_skill_md,
    load_skill_metadata_only,
    _validate_skill_name,
)
from squadron.skills.manager import SkillsManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_skills_dir():
    """Create a temporary directory with sample skills."""
    with TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / ".github" / "skills"
        skills_dir.mkdir(parents=True)
        
        # Create a valid skill
        valid_skill = skills_dir / "python-testing"
        valid_skill.mkdir()
        (valid_skill / "SKILL.md").write_text("""---
name: python-testing
description: A skill for writing Python tests with pytest
tags: [python, testing, pytest]
version: 1.0.0
---
# Python Testing Skill

## Instructions
When writing Python tests:
1. Use pytest as the testing framework
2. Follow the AAA pattern (Arrange, Act, Assert)
3. Use descriptive test names

## Examples
```python
def test_example():
    # Arrange
    data = [1, 2, 3]
    
    # Act
    result = sum(data)
    
    # Assert
    assert result == 6
```
""")
        
        # Create another skill
        web_skill = skills_dir / "web-scraping"
        web_skill.mkdir()
        (web_skill / "SKILL.md").write_text("""---
name: web-scraping
description: Skill for web scraping with BeautifulSoup and requests
tags: [web, scraping, beautifulsoup]
---
# Web Scraping Skill

Use BeautifulSoup and requests for web scraping tasks.
""")
        (web_skill / "selectors.md").write_text("# CSS Selector Reference\n...")
        
        # Create an invalid skill (missing required fields)
        invalid_skill = skills_dir / "invalid-skill"
        invalid_skill.mkdir()
        (invalid_skill / "SKILL.md").write_text("""---
name: invalid-skill
---
Missing description field.
""")
        
        yield Path(tmpdir)


@pytest.fixture
def sample_skill_md_content():
    """Sample SKILL.md content for parser tests."""
    return """---
name: sample-skill
description: A sample skill for testing
short-description: Sample skill
version: 1.0.0
author: Test Author
tags: [test, sample]
---
# Sample Skill

These are the instructions for the sample skill.

## Usage
Follow these steps to use the skill.
"""


# ============================================================================
# Test SkillMetadata
# ============================================================================

class TestSkillMetadata:
    """Tests for SkillMetadata class."""
    
    def test_matches_query_exact_name(self):
        """Test matching when query contains exact skill name."""
        metadata = SkillMetadata(
            name="python-testing",
            description="Write Python tests",
        )
        
        score = metadata.matches_query("help me with python-testing")
        assert score >= 0.4
    
    def test_matches_query_description_words(self):
        """Test matching based on description words."""
        metadata = SkillMetadata(
            name="my-skill",
            description="Create PowerPoint presentations with charts",
        )
        
        score = metadata.matches_query("create a PowerPoint presentation")
        assert score > 0.0
    
    def test_matches_query_tags(self):
        """Test matching based on tags."""
        metadata = SkillMetadata(
            name="my-skill",
            description="Some description",
            tags=["pytest", "testing"],
        )
        
        score = metadata.matches_query("I need pytest")
        assert score >= 0.2
    
    def test_matches_query_no_match(self):
        """Test low score for unrelated query."""
        metadata = SkillMetadata(
            name="python-testing",
            description="Write Python tests",
        )
        
        score = metadata.matches_query("cook a delicious meal")
        assert score < 0.2


# ============================================================================
# Test Parser Functions
# ============================================================================

class TestParser:
    """Tests for SKILL.md parser functions."""
    
    def test_parse_yaml_frontmatter_valid(self, sample_skill_md_content):
        """Test parsing valid YAML frontmatter."""
        metadata, body = parse_yaml_frontmatter(sample_skill_md_content)
        
        assert metadata["name"] == "sample-skill"
        assert metadata["description"] == "A sample skill for testing"
        assert metadata["version"] == "1.0.0"
        assert "# Sample Skill" in body
    
    def test_parse_yaml_frontmatter_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "# Just Markdown\n\nNo frontmatter here."
        metadata, body = parse_yaml_frontmatter(content)
        
        assert metadata == {}
        assert "# Just Markdown" in body
    
    def test_parse_skill_md_valid(self, temp_skills_dir):
        """Test parsing a valid SKILL.md file."""
        skill_path = temp_skills_dir / ".github" / "skills" / "python-testing" / "SKILL.md"
        skill = parse_skill_md(skill_path)
        
        assert skill is not None
        assert skill.name == "python-testing"
        assert "pytest" in skill.description
        assert skill.instructions is not None
        assert "AAA pattern" in skill.instructions
    
    def test_parse_skill_md_missing_description(self, temp_skills_dir):
        """Test parsing SKILL.md with missing description returns None."""
        skill_path = temp_skills_dir / ".github" / "skills" / "invalid-skill" / "SKILL.md"
        skill = parse_skill_md(skill_path)
        
        assert skill is None
    
    def test_parse_skill_md_nonexistent(self):
        """Test parsing nonexistent file returns None."""
        skill = parse_skill_md(Path("/nonexistent/SKILL.md"))
        assert skill is None
    
    def test_validate_skill_name_valid(self):
        """Test valid skill names."""
        assert _validate_skill_name("python-testing") is True
        assert _validate_skill_name("web-scraping") is True
        assert _validate_skill_name("skill123") is True
    
    def test_validate_skill_name_invalid(self):
        """Test invalid skill names."""
        assert _validate_skill_name("Python-Testing") is False  # uppercase
        assert _validate_skill_name("skill_name") is False  # underscore
        assert _validate_skill_name("my-anthropic-skill") is False  # reserved word
        assert _validate_skill_name("a" * 65) is False  # too long


# ============================================================================
# Test SkillsManager
# ============================================================================

class TestSkillsManager:
    """Tests for SkillsManager class."""
    
    @pytest.mark.asyncio
    async def test_discover_skills(self, temp_skills_dir):
        """Test skill discovery."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        count = await manager.discover_skills()
        
        # Should find 2 valid skills (invalid-skill is filtered out)
        assert count == 2
        assert "python-testing" in manager.available_skills
        assert "web-scraping" in manager.available_skills
    
    @pytest.mark.asyncio
    async def test_find_skills(self, temp_skills_dir):
        """Test finding relevant skills."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        matches = manager.find_skills("write python tests")
        
        assert len(matches) > 0
        # Python testing skill should be highly relevant
        skill_names = [m.skill.name for m in matches]
        assert "python-testing" in skill_names
    
    @pytest.mark.asyncio
    async def test_load_skill(self, temp_skills_dir):
        """Test loading a full skill."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        skill = await manager.load_skill("python-testing")
        
        assert skill is not None
        assert skill.is_loaded
        assert skill.instructions is not None
        assert "pytest" in skill.instructions
    
    @pytest.mark.asyncio
    async def test_load_skill_resource(self, temp_skills_dir):
        """Test loading a skill resource."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        content = await manager.load_skill_resource("web-scraping", "selectors.md")
        
        assert content is not None
        assert "CSS Selector" in content
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_skill(self, temp_skills_dir):
        """Test loading a skill that doesn't exist."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        skill = await manager.load_skill("nonexistent-skill")
        assert skill is None
    
    @pytest.mark.asyncio
    async def test_get_skills_summary(self, temp_skills_dir):
        """Test getting skills summary."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        summary = manager.get_skills_summary()
        
        assert "Available Skills" in summary
        assert "python-testing" in summary
        assert "web-scraping" in summary
    
    @pytest.mark.asyncio
    async def test_empty_workspace(self):
        """Test manager with no skills."""
        with TemporaryDirectory() as tmpdir:
            manager = SkillsManager(workspace_path=tmpdir)
            count = await manager.discover_skills()
            
            assert count == 0
            assert manager.get_skills_summary() == "No skills available."
    
    @pytest.mark.asyncio
    async def test_find_skills_uses_loaded_cache(self, temp_skills_dir):
        """Test that find_skills returns loaded skills with instructions."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        # Load a skill first
        loaded_skill = await manager.load_skill("python-testing")
        assert loaded_skill is not None
        assert loaded_skill.is_loaded
        
        # Now find_skills should return the loaded version
        matches = manager.find_skills("python testing")
        python_match = next((m for m in matches if m.skill.name == "python-testing"), None)
        
        assert python_match is not None
        assert python_match.skill.is_loaded  # Should have instructions
        assert python_match.skill.instructions is not None
    
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, temp_skills_dir):
        """Test that path traversal attempts are blocked."""
        manager = SkillsManager(workspace_path=temp_skills_dir)
        await manager.discover_skills()
        
        # Try to load a resource with path traversal
        content = await manager.load_skill_resource("web-scraping", "../../../etc/passwd")
        assert content is None
        
        # Try another traversal pattern
        content = await manager.load_skill_resource("web-scraping", "../../python-testing/SKILL.md")
        assert content is None


# ============================================================================
# Test Skill Model
# ============================================================================

class TestSkillModel:
    """Tests for Skill model class."""
    
    def test_to_context_string(self):
        """Test formatting skill for LLM context."""
        metadata = SkillMetadata(
            name="test-skill",
            description="A test skill",
        )
        skill = Skill(
            metadata=metadata,
            path=Path("/tmp/test-skill"),
            instructions="Follow these instructions.",
        )
        
        context = skill.to_context_string()
        
        assert "## Skill: test-skill" in context
        assert "A test skill" in context
        assert "Follow these instructions" in context
    
    def test_is_loaded(self):
        """Test is_loaded property."""
        metadata = SkillMetadata(name="test", description="Test")
        
        # Not loaded
        skill = Skill(metadata=metadata, path=Path("/tmp"))
        assert skill.is_loaded is False
        
        # Loaded
        skill.instructions = "Some instructions"
        assert skill.is_loaded is True


# ============================================================================
# Test SkillMatch
# ============================================================================

class TestSkillMatch:
    """Tests for SkillMatch class."""
    
    def test_sorting(self):
        """Test that SkillMatch sorts by score."""
        metadata = SkillMetadata(name="test", description="Test")
        skill = Skill(metadata=metadata, path=Path("/tmp"))
        
        match1 = SkillMatch(skill=skill, score=0.5)
        match2 = SkillMatch(skill=skill, score=0.8)
        match3 = SkillMatch(skill=skill, score=0.3)
        
        sorted_matches = sorted([match1, match2, match3], reverse=True)
        scores = [m.score for m in sorted_matches]
        
        assert scores == [0.8, 0.5, 0.3]
