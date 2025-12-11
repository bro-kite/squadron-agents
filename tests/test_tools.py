"""
Tests for Pre-built Tool Packs.

Tests CodingTools, ResearchTools, and OpsTools.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from squadron.tools.coding import CodingTools, FileMatch, EditResult
from squadron.tools.research import ResearchTools, SearchResult, WebPage
from squadron.tools.ops import OpsTools, CommandResult, ProcessInfo


class TestCodingTools:
    """Tests for CodingTools."""
    
    def test_init(self):
        tools = CodingTools(workspace_root="/tmp/test")
        assert tools.workspace_root == Path("/tmp/test")
    
    def test_init_default_workspace(self):
        tools = CodingTools()
        assert tools.workspace_root == Path.cwd()
    
    @pytest.mark.asyncio
    async def test_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name
        
        try:
            tools = CodingTools(workspace_root=os.path.dirname(temp_path))
            content = await tools.read_file(os.path.basename(temp_path))
            assert "Hello, World!" in content
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_write_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = CodingTools(workspace_root=tmpdir)
            result = await tools.write_file("test.txt", "Test content")
            
            assert "success" in result.lower() or "wrote" in result.lower()
            assert os.path.exists(os.path.join(tmpdir, "test.txt"))
    
    @pytest.mark.asyncio
    async def test_grep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("def hello():\n    print('hello')\n")
            
            tools = CodingTools(workspace_root=tmpdir)
            results = await tools.grep("hello")
            
            assert len(results) > 0
            assert any("hello" in str(r) for r in results)
    
    @pytest.mark.asyncio
    async def test_find_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "subdir"))
            Path(os.path.join(tmpdir, "file1.py")).touch()
            Path(os.path.join(tmpdir, "file2.py")).touch()
            Path(os.path.join(tmpdir, "subdir", "file3.py")).touch()
            
            tools = CodingTools(workspace_root=tmpdir)
            files = await tools.find_files("*.py")
            
            assert len(files) >= 3


class TestFileMatch:
    """Tests for FileMatch dataclass."""
    
    def test_create(self):
        match = FileMatch(
            file_path="src/main.py",
            line_number=42,
            line_content="def main():",
            context_before=["# Main entry point"],
            context_after=["    pass"],
        )
        assert match.file_path == "src/main.py"
        assert match.line_number == 42


class TestEditResult:
    """Tests for EditResult dataclass."""
    
    def test_success(self):
        result = EditResult(
            success=True,
            file_path="test.py",
            changes_made=1,
        )
        assert result.success is True
    
    def test_failure(self):
        result = EditResult(
            success=False,
            file_path="test.py",
            error="File not found",
        )
        assert result.success is False
        assert result.error == "File not found"


class TestResearchTools:
    """Tests for ResearchTools."""
    
    def test_init(self):
        tools = ResearchTools()
        assert tools is not None
    
    def test_init_with_api_key(self):
        tools = ResearchTools(search_api_key="test-key")
        assert tools.search_api_key == "test-key"


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_create(self):
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="This is a test snippet",
        )
        assert result.title == "Test Result"
        assert result.url == "https://example.com"


class TestWebPage:
    """Tests for WebPage dataclass."""
    
    def test_create(self):
        page = WebPage(
            url="https://example.com",
            title="Example",
            content="Page content here",
            links=["https://example.com/page1"],
        )
        assert page.url == "https://example.com"
        assert len(page.links) == 1


class TestOpsTools:
    """Tests for OpsTools."""
    
    def test_init(self):
        tools = OpsTools()
        assert tools is not None
    
    def test_init_with_allowed_commands(self):
        tools = OpsTools(allowed_commands=["ls", "cat", "echo"])
        assert "ls" in tools.allowed_commands
    
    @pytest.mark.asyncio
    async def test_run_command_echo(self):
        tools = OpsTools()
        result = await tools.run_command("echo 'hello'")
        assert result.success is True
        assert "hello" in result.stdout
    
    @pytest.mark.asyncio
    async def test_run_command_blocked(self):
        tools = OpsTools(blocked_commands=["rm -rf /"])
        result = await tools.run_command("rm -rf /")
        assert result.success is False
    
    @pytest.mark.asyncio
    async def test_system_info(self):
        tools = OpsTools()
        info = await tools.system_info()
        # system_info returns a dict
        assert isinstance(info, dict)
        assert "platform" in info or "os" in info or "system" in info


class TestCommandResult:
    """Tests for CommandResult dataclass."""
    
    def test_success(self):
        result = CommandResult(
            command="echo test",
            exit_code=0,
            stdout="output",
            stderr="",
            success=True,
        )
        assert result.success is True
        assert result.exit_code == 0
    
    def test_failure(self):
        result = CommandResult(
            command="bad_command",
            exit_code=1,
            stdout="",
            stderr="error message",
            success=False,
        )
        assert result.success is False


class TestProcessInfo:
    """Tests for ProcessInfo dataclass."""
    
    def test_create(self):
        info = ProcessInfo(
            pid=1234,
            name="python",
            cpu_percent=5.0,
            memory_mb=256.0,
            status="running",
        )
        assert info.pid == 1234
        assert info.name == "python"
