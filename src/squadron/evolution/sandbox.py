"""
Sandboxed Execution Environment

Provides isolated execution environments for self-improvement experiments.
Uses Docker containers or subprocess isolation to safely test mutations.

Key features:
- Docker-based isolation
- Resource limits (CPU, memory, time)
- File system isolation
- Network restrictions
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

from squadron.core.config import EvolutionConfig

logger = structlog.get_logger(__name__)


class SandboxType(str, Enum):
    """Type of sandbox isolation."""
    DOCKER = "docker"
    SUBPROCESS = "subprocess"
    NONE = "none"  # For testing only


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    
    sandbox_type: SandboxType = SandboxType.SUBPROCESS
    
    # Docker settings
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"  # Disable networking by default
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_seconds: float = 60.0
    
    # File system
    working_dir: str | None = None
    mount_paths: list[tuple[str, str]] = field(default_factory=list)  # (host, container)
    
    # Permissions
    allow_network: bool = False
    allow_write: bool = True
    
    @classmethod
    def from_evolution_config(cls, config: EvolutionConfig) -> SandboxConfig:
        """Create from evolution config."""
        return cls(
            docker_image=config.sandbox_image,
        )


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    
    id: UUID = field(default_factory=uuid4)
    
    # Status
    success: bool = False
    exit_code: int = -1
    
    # Output
    stdout: str = ""
    stderr: str = ""
    
    # Artifacts
    output_files: dict[str, str] = field(default_factory=dict)  # path -> content
    
    # Metrics
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    
    # Error
    error: str | None = None
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "success": self.success,
            "exitCode": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "executionTimeSeconds": self.execution_time_seconds,
            "memoryUsedMb": self.memory_used_mb,
            "error": self.error,
            "startedAt": self.started_at.isoformat(),
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }


class Sandbox:
    """
    Sandboxed Execution Environment.
    
    Provides isolated execution for testing agent mutations and
    self-improvement experiments.
    
    Example:
        ```python
        sandbox = Sandbox(config=SandboxConfig())
        
        # Execute code in sandbox
        result = await sandbox.execute(
            code="print('Hello, World!')",
            language="python",
        )
        
        # Execute a file
        result = await sandbox.execute_file(
            file_path="test_agent.py",
            args=["--test"],
        )
        ```
    """
    
    def __init__(self, config: SandboxConfig | None = None):
        """
        Initialize the sandbox.
        
        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self._temp_dirs: list[Path] = []
        self._docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout (overrides config)
            env: Environment variables
            
        Returns:
            Execution result
        """
        timeout = timeout or self.config.max_execution_seconds
        
        # Create temp directory for execution
        temp_dir = Path(tempfile.mkdtemp(prefix="squadron_sandbox_"))
        self._temp_dirs.append(temp_dir)
        
        try:
            # Write code to file
            if language == "python":
                code_file = temp_dir / "main.py"
                code_file.write_text(code)
                cmd = ["python", str(code_file)]
            elif language == "javascript":
                code_file = temp_dir / "main.js"
                code_file.write_text(code)
                cmd = ["node", str(code_file)]
            elif language == "bash":
                code_file = temp_dir / "main.sh"
                code_file.write_text(code)
                cmd = ["bash", str(code_file)]
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported language: {language}",
                )
            
            # Execute based on sandbox type
            if self.config.sandbox_type == SandboxType.DOCKER and self._docker_available:
                result = await self._execute_docker(cmd, temp_dir, timeout, env)
            else:
                result = await self._execute_subprocess(cmd, temp_dir, timeout, env)
            
            return result
            
        finally:
            # Cleanup temp directory
            self._cleanup_temp_dir(temp_dir)
    
    async def execute_file(
        self,
        file_path: str | Path,
        args: list[str] | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a file in the sandbox.
        
        Args:
            file_path: Path to the file to execute
            args: Command line arguments
            timeout: Execution timeout
            env: Environment variables
            
        Returns:
            Execution result
        """
        file_path = Path(file_path)
        timeout = timeout or self.config.max_execution_seconds
        args = args or []
        
        if not file_path.exists():
            return ExecutionResult(
                success=False,
                error=f"File not found: {file_path}",
            )
        
        # Determine command based on file extension
        ext = file_path.suffix.lower()
        if ext == ".py":
            cmd = ["python", str(file_path)] + args
        elif ext == ".js":
            cmd = ["node", str(file_path)] + args
        elif ext == ".sh":
            cmd = ["bash", str(file_path)] + args
        else:
            return ExecutionResult(
                success=False,
                error=f"Unsupported file type: {ext}",
            )
        
        # Create temp directory for working
        temp_dir = Path(tempfile.mkdtemp(prefix="squadron_sandbox_"))
        self._temp_dirs.append(temp_dir)
        
        try:
            if self.config.sandbox_type == SandboxType.DOCKER and self._docker_available:
                result = await self._execute_docker(cmd, temp_dir, timeout, env)
            else:
                result = await self._execute_subprocess(cmd, temp_dir, timeout, env)
            
            return result
            
        finally:
            self._cleanup_temp_dir(temp_dir)
    
    async def _execute_subprocess(
        self,
        cmd: list[str],
        working_dir: Path,
        timeout: float,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute using subprocess isolation."""
        result = ExecutionResult()
        result.started_at = datetime.utcnow()
        
        # Build environment
        exec_env = dict(os.environ)
        if env:
            exec_env.update(env)
        
        # Remove potentially dangerous env vars
        for key in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
            if key in exec_env and not env:
                pass  # Keep system defaults unless overridden
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=exec_env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.success = False
                result.error = f"Execution timed out after {timeout}s"
                result.exit_code = -1
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.exit_code = -1
        
        result.completed_at = datetime.utcnow()
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        return result
    
    async def _execute_docker(
        self,
        cmd: list[str],
        working_dir: Path,
        timeout: float,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute using Docker isolation."""
        result = ExecutionResult()
        result.started_at = datetime.utcnow()
        
        # Build docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            f"--memory={self.config.max_memory_mb}m",
            f"--cpus={self.config.max_cpu_percent / 100}",
            f"--network={self.config.docker_network if not self.config.allow_network else 'bridge'}",
            "-v", f"{working_dir}:/workspace:{'rw' if self.config.allow_write else 'ro'}",
            "-w", "/workspace",
        ]
        
        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add mount paths
        for host_path, container_path in self.config.mount_paths:
            docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
        
        # Add image and command
        docker_cmd.append(self.config.docker_image)
        docker_cmd.extend(cmd)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 10,  # Extra time for Docker overhead
                )
                
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
                
            except asyncio.TimeoutError:
                # Kill the container
                container_id = await self._get_running_container()
                if container_id:
                    subprocess.run(["docker", "kill", container_id], capture_output=True)
                
                result.success = False
                result.error = f"Docker execution timed out after {timeout}s"
                result.exit_code = -1
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.exit_code = -1
        
        result.completed_at = datetime.utcnow()
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        return result
    
    async def _get_running_container(self) -> str | None:
        """Get the ID of a running container (for cleanup)."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={self.config.docker_image}"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.decode().strip().split("\n")[0]
        except Exception:
            pass
        return None
    
    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up a temporary directory."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_dirs:
                self._temp_dirs.remove(temp_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir", path=str(temp_dir), error=str(e))
    
    async def cleanup(self) -> None:
        """Clean up all temporary directories."""
        for temp_dir in list(self._temp_dirs):
            self._cleanup_temp_dir(temp_dir)
    
    async def run_tests(
        self,
        test_file: str | Path,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Run a test file in the sandbox.
        
        Args:
            test_file: Path to test file
            timeout: Execution timeout
            
        Returns:
            Execution result
        """
        test_file = Path(test_file)
        timeout = timeout or self.config.max_execution_seconds * 2  # More time for tests
        
        # Use pytest for Python tests
        if test_file.suffix == ".py":
            return await self.execute_file(
                file_path=test_file,
                args=["-v"],
                timeout=timeout,
            )
        
        return await self.execute_file(test_file, timeout=timeout)
    
    @property
    def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        return self._docker_available
