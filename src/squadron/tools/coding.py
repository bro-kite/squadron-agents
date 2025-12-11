"""
Coding Tools Pack

A comprehensive set of tools for software development tasks:
- File operations (read, write, edit)
- Code search (grep, find)
- Git integration
- Code analysis
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

import structlog

from squadron.connectivity.mcp_host import mcp_tool

logger = structlog.get_logger(__name__)


@dataclass
class FileMatch:
    """A match found in a file."""
    file_path: str
    line_number: int
    line_content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class EditResult:
    """Result of a file edit operation."""
    success: bool
    file_path: str
    changes_made: int = 0
    error: str | None = None


class CodingTools:
    """
    Coding Tools Pack.
    
    Provides essential tools for software development:
    - File reading and writing
    - Code search with ripgrep
    - Git operations
    - Code analysis
    
    Example:
        ```python
        tools = CodingTools(workspace_root="/path/to/project")
        
        # Read a file
        content = await tools.read_file("src/main.py")
        
        # Search for patterns
        matches = await tools.grep("def main", include="*.py")
        
        # Edit a file
        result = await tools.edit_file(
            "src/main.py",
            old_string="print('hello')",
            new_string="print('Hello, World!')",
        )
        ```
    """
    
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        allowed_extensions: list[str] | None = None,
        max_file_size_mb: float = 10.0,
    ):
        """
        Initialize coding tools.
        
        Args:
            workspace_root: Root directory for file operations
            allowed_extensions: Allowed file extensions (None = all)
            max_file_size_mb: Maximum file size to read
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.allowed_extensions = allowed_extensions
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        
        # Check for ripgrep
        self._has_ripgrep = self._check_command("rg")
        self._has_git = self._check_command("git")
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False
    
    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to workspace root."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.workspace_root / path
    
    def _validate_path(self, path: Path) -> bool:
        """Validate that a path is safe to access."""
        try:
            resolved = path.resolve()
            workspace_resolved = self.workspace_root.resolve()
            
            # Must be within workspace
            if not str(resolved).startswith(str(workspace_resolved)):
                return False
            
            # Check extension if restricted
            if self.allowed_extensions:
                if path.suffix.lstrip(".") not in self.allowed_extensions:
                    return False
            
            return True
        except Exception:
            return False
    
    @mcp_tool(description="Read the contents of a file")
    async def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Read the contents of a file.
        
        Args:
            path: Path to the file (relative to workspace)
            start_line: Starting line number (1-indexed, optional)
            end_line: Ending line number (inclusive, optional)
            
        Returns:
            File contents with line numbers
        """
        file_path = self._resolve_path(path)
        
        if not self._validate_path(file_path):
            raise ValueError(f"Access denied: {path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if file_path.stat().st_size > self.max_file_size_bytes:
            raise ValueError(f"File too large: {path}")
        
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        
        # Apply line range
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]
            line_offset = start_idx
        else:
            line_offset = 0
        
        # Format with line numbers
        numbered_lines = [
            f"{i + line_offset + 1:6d}\t{line}"
            for i, line in enumerate(lines)
        ]
        
        return "\n".join(numbered_lines)
    
    @mcp_tool(description="Write content to a file")
    async def write_file(
        self,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> str:
        """
        Write content to a file.
        
        Args:
            path: Path to the file
            content: Content to write
            create_dirs: Create parent directories if needed
            
        Returns:
            Success message
        """
        file_path = self._resolve_path(path)
        
        if not self._validate_path(file_path):
            raise ValueError(f"Access denied: {path}")
        
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content, encoding="utf-8")
        
        return f"Successfully wrote {len(content)} bytes to {path}"
    
    @mcp_tool(description="Edit a file by replacing text")
    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """
        Edit a file by replacing text.
        
        Args:
            path: Path to the file
            old_string: Text to find
            new_string: Text to replace with
            replace_all: Replace all occurrences
            
        Returns:
            Edit result
        """
        file_path = self._resolve_path(path)
        
        if not self._validate_path(file_path):
            return EditResult(
                success=False,
                file_path=path,
                error=f"Access denied: {path}",
            )
        
        if not file_path.exists():
            return EditResult(
                success=False,
                file_path=path,
                error=f"File not found: {path}",
            )
        
        content = file_path.read_text(encoding="utf-8")
        
        if old_string not in content:
            return EditResult(
                success=False,
                file_path=path,
                error="old_string not found in file",
            )
        
        if not replace_all:
            # Check uniqueness
            count = content.count(old_string)
            if count > 1:
                return EditResult(
                    success=False,
                    file_path=path,
                    error=f"old_string found {count} times, use replace_all=True or provide more context",
                )
        
        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            changes = content.count(old_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            changes = 1
        
        file_path.write_text(new_content, encoding="utf-8")
        
        return EditResult(
            success=True,
            file_path=path,
            changes_made=changes,
        )
    
    @mcp_tool(description="Search for patterns in files using ripgrep")
    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
        exclude: str | None = None,
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 50,
    ) -> list[FileMatch]:
        """
        Search for patterns in files.
        
        Args:
            pattern: Search pattern (regex supported)
            path: Directory to search (default: workspace root)
            include: Glob pattern for files to include
            exclude: Glob pattern for files to exclude
            case_sensitive: Case-sensitive search
            context_lines: Lines of context around matches
            max_results: Maximum number of results
            
        Returns:
            List of matches
        """
        search_path = self._resolve_path(path) if path else self.workspace_root
        
        if not self._validate_path(search_path):
            raise ValueError(f"Access denied: {path}")
        
        if self._has_ripgrep:
            return await self._grep_ripgrep(
                pattern, search_path, include, exclude,
                case_sensitive, context_lines, max_results
            )
        else:
            return await self._grep_python(
                pattern, search_path, include, exclude,
                case_sensitive, context_lines, max_results
            )
    
    async def _grep_ripgrep(
        self,
        pattern: str,
        search_path: Path,
        include: str | None,
        exclude: str | None,
        case_sensitive: bool,
        context_lines: int,
        max_results: int,
    ) -> list[FileMatch]:
        """Search using ripgrep."""
        cmd = ["rg", "--json", "-m", str(max_results)]
        
        if not case_sensitive:
            cmd.append("-i")
        
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        
        if include:
            cmd.extend(["-g", include])
        
        if exclude:
            cmd.extend(["-g", f"!{exclude}"])
        
        cmd.extend([pattern, str(search_path)])
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            
            matches = []
            import json
            
            for line in stdout.decode().splitlines():
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data["data"]
                        matches.append(FileMatch(
                            file_path=match_data["path"]["text"],
                            line_number=match_data["line_number"],
                            line_content=match_data["lines"]["text"].rstrip(),
                        ))
                except json.JSONDecodeError:
                    continue
            
            return matches[:max_results]
            
        except Exception as e:
            logger.warning("ripgrep failed, falling back to Python", error=str(e))
            return await self._grep_python(
                pattern, search_path, include, exclude,
                case_sensitive, context_lines, max_results
            )
    
    async def _grep_python(
        self,
        pattern: str,
        search_path: Path,
        include: str | None,
        exclude: str | None,
        case_sensitive: bool,
        context_lines: int,
        max_results: int,
    ) -> list[FileMatch]:
        """Search using Python (fallback)."""
        import fnmatch
        
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        matches = []
        
        for root, _, files in os.walk(search_path):
            for filename in files:
                if len(matches) >= max_results:
                    break
                
                # Check include/exclude patterns
                if include and not fnmatch.fnmatch(filename, include):
                    continue
                if exclude and fnmatch.fnmatch(filename, exclude):
                    continue
                
                file_path = Path(root) / filename
                
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            matches.append(FileMatch(
                                file_path=str(file_path.relative_to(self.workspace_root)),
                                line_number=i + 1,
                                line_content=line,
                            ))
                            
                            if len(matches) >= max_results:
                                break
                                
                except Exception:
                    continue
        
        return matches
    
    @mcp_tool(description="Find files by name pattern")
    async def find_files(
        self,
        pattern: str,
        path: str | None = None,
        file_type: str = "file",
        max_depth: int | None = None,
        max_results: int = 50,
    ) -> list[str]:
        """
        Find files by name pattern.
        
        Args:
            pattern: Glob pattern to match
            path: Directory to search
            file_type: "file", "directory", or "any"
            max_depth: Maximum directory depth
            max_results: Maximum results
            
        Returns:
            List of matching paths
        """
        search_path = self._resolve_path(path) if path else self.workspace_root
        
        if not self._validate_path(search_path):
            raise ValueError(f"Access denied: {path}")
        
        results = []
        
        for item in search_path.rglob(pattern):
            if len(results) >= max_results:
                break
            
            # Check depth
            if max_depth is not None:
                relative = item.relative_to(search_path)
                if len(relative.parts) > max_depth:
                    continue
            
            # Check type
            if file_type == "file" and not item.is_file():
                continue
            if file_type == "directory" and not item.is_dir():
                continue
            
            results.append(str(item.relative_to(self.workspace_root)))
        
        return results
    
    @mcp_tool(description="Get git status")
    async def git_status(self) -> dict[str, Any]:
        """
        Get git repository status.
        
        Returns:
            Git status information
        """
        if not self._has_git:
            raise RuntimeError("Git not available")
        
        result = {
            "branch": "",
            "staged": [],
            "modified": [],
            "untracked": [],
        }
        
        # Get current branch
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "--show-current",
                cwd=self.workspace_root,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            result["branch"] = stdout.decode().strip()
        except Exception:
            pass
        
        # Get status
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain",
                cwd=self.workspace_root,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            
            for line in stdout.decode().splitlines():
                if not line:
                    continue
                
                status = line[:2]
                file_path = line[3:]
                
                if status[0] in "MADRC":
                    result["staged"].append(file_path)
                if status[1] == "M":
                    result["modified"].append(file_path)
                if status == "??":
                    result["untracked"].append(file_path)
                    
        except Exception:
            pass
        
        return result
    
    @mcp_tool(description="Get git diff")
    async def git_diff(
        self,
        path: str | None = None,
        staged: bool = False,
    ) -> str:
        """
        Get git diff.
        
        Args:
            path: Specific file to diff (optional)
            staged: Show staged changes
            
        Returns:
            Diff output
        """
        if not self._has_git:
            raise RuntimeError("Git not available")
        
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if path:
            cmd.append(path)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace_root,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        
        return stdout.decode()
    
    @mcp_tool(description="List directory contents")
    async def list_dir(
        self,
        path: str | None = None,
        show_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            show_hidden: Include hidden files
            
        Returns:
            List of file/directory info
        """
        dir_path = self._resolve_path(path) if path else self.workspace_root
        
        if not self._validate_path(dir_path):
            raise ValueError(f"Access denied: {path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        items = []
        
        for item in sorted(dir_path.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            
            info = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.relative_to(self.workspace_root)),
            }
            
            if item.is_file():
                info["size"] = item.stat().st_size
            elif item.is_dir():
                try:
                    info["items"] = len(list(item.iterdir()))
                except PermissionError:
                    info["items"] = 0
            
            items.append(info)
        
        return items
    
    def get_tools(self) -> list[Callable]:
        """Get all tools as a list of callables."""
        return [
            self.read_file,
            self.write_file,
            self.edit_file,
            self.grep,
            self.find_files,
            self.git_status,
            self.git_diff,
            self.list_dir,
        ]
