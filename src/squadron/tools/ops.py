"""
Operations Tools Pack

Tools for system operations and DevOps tasks:
- Shell command execution
- Docker management
- Process monitoring
- System information
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import structlog

from squadron.connectivity.mcp_host import mcp_tool

logger = structlog.get_logger(__name__)


@dataclass
class CommandResult:
    """Result of a shell command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    success: bool = False


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = ""
    created: datetime | None = None


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    id: str
    name: str
    image: str
    status: str
    ports: list[str] = field(default_factory=list)
    created: str = ""


class OpsTools:
    """
    Operations Tools Pack.
    
    Provides tools for system operations:
    - Shell command execution (with safety checks)
    - Docker container management
    - Process monitoring
    - System information
    
    Example:
        ```python
        tools = OpsTools(
            allowed_commands=["ls", "cat", "grep", "docker"],
            working_dir="/app",
        )
        
        # Run a command
        result = await tools.run_command("ls -la")
        
        # List Docker containers
        containers = await tools.docker_ps()
        
        # Get system info
        info = await tools.system_info()
        ```
    """
    
    def __init__(
        self,
        working_dir: str | None = None,
        allowed_commands: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        max_output_length: int = 50000,
        command_timeout: float = 60.0,
    ):
        """
        Initialize ops tools.
        
        Args:
            working_dir: Default working directory
            allowed_commands: Whitelist of allowed commands (None = all)
            blocked_commands: Blacklist of blocked commands
            max_output_length: Maximum output length
            command_timeout: Default command timeout
        """
        self.working_dir = working_dir or os.getcwd()
        self.allowed_commands = allowed_commands
        self.blocked_commands = blocked_commands or [
            "rm -rf /",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # Fork bomb
            "chmod -R 777 /",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
        ]
        self.max_output_length = max_output_length
        self.command_timeout = command_timeout
        
        # Check for Docker
        self._has_docker = self._check_command("docker")
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        return shutil.which(cmd) is not None
    
    def _validate_command(self, command: str) -> tuple[bool, str]:
        """
        Validate a command for safety.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check blocked patterns
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Blocked command pattern: {blocked}"
        
        # Check allowed commands
        if self.allowed_commands:
            cmd_name = command.split()[0] if command else ""
            if cmd_name not in self.allowed_commands:
                return False, f"Command not in allowlist: {cmd_name}"
        
        return True, ""
    
    @mcp_tool(description="Execute a shell command", requires_approval=True)
    async def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Command timeout
            env: Environment variables
            
        Returns:
            Command result
        """
        # Validate command
        is_valid, error = self._validate_command(command)
        if not is_valid:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=error,
                success=False,
            )
        
        cwd = cwd or self.working_dir
        timeout = timeout or self.command_timeout
        
        # Build environment
        exec_env = dict(os.environ)
        if env:
            exec_env.update(env)
        
        start_time = datetime.utcnow()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=exec_env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                
                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")
                
                # Truncate if too long
                if len(stdout_str) > self.max_output_length:
                    stdout_str = stdout_str[:self.max_output_length] + "\n[Output truncated...]"
                if len(stderr_str) > self.max_output_length:
                    stderr_str = stderr_str[:self.max_output_length] + "\n[Output truncated...]"
                
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                return CommandResult(
                    command=command,
                    exit_code=process.returncode or 0,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    duration_seconds=duration,
                    success=process.returncode == 0,
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return CommandResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    success=False,
                )
                
        except Exception as e:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                success=False,
            )
    
    @mcp_tool(description="List running Docker containers")
    async def docker_ps(
        self,
        all_containers: bool = False,
    ) -> list[ContainerInfo]:
        """
        List Docker containers.
        
        Args:
            all_containers: Include stopped containers
            
        Returns:
            List of container info
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "ps", "--format", "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}"]
        if all_containers:
            cmd.insert(2, "-a")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        
        containers = []
        for line in stdout.decode().splitlines():
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                containers.append(ContainerInfo(
                    id=parts[0],
                    name=parts[1],
                    image=parts[2],
                    status=parts[3],
                    ports=parts[4].split(",") if len(parts) > 4 and parts[4] else [],
                ))
        
        return containers
    
    @mcp_tool(description="Get Docker container logs")
    async def docker_logs(
        self,
        container: str,
        tail: int = 100,
        follow: bool = False,
    ) -> str:
        """
        Get Docker container logs.
        
        Args:
            container: Container name or ID
            tail: Number of lines to show
            follow: Follow log output (not recommended for tools)
            
        Returns:
            Container logs
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "logs", "--tail", str(tail), container]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        # Docker logs go to both stdout and stderr
        output = stdout.decode("utf-8", errors="replace")
        output += stderr.decode("utf-8", errors="replace")
        
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "\n[Logs truncated...]"
        
        return output
    
    @mcp_tool(description="Execute a command in a Docker container")
    async def docker_exec(
        self,
        container: str,
        command: str,
        workdir: str | None = None,
    ) -> CommandResult:
        """
        Execute a command in a Docker container.
        
        Args:
            container: Container name or ID
            command: Command to execute
            workdir: Working directory in container
            
        Returns:
            Command result
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.extend([container, "sh", "-c", command])
        
        start_time = datetime.utcnow()
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.command_timeout,
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return CommandResult(
                command=f"docker exec {container}: {command}",
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_seconds=duration,
                success=process.returncode == 0,
            )
            
        except asyncio.TimeoutError:
            return CommandResult(
                command=f"docker exec {container}: {command}",
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {self.command_timeout}s",
                success=False,
            )
    
    @mcp_tool(description="Get system information")
    async def system_info(self) -> dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information dictionary
        """
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }
        
        # Try to get more detailed info
        try:
            import psutil
            
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            memory = psutil.virtual_memory()
            info["memory_total_gb"] = round(memory.total / (1024**3), 2)
            info["memory_available_gb"] = round(memory.available / (1024**3), 2)
            info["memory_percent"] = memory.percent
            
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / (1024**3), 2)
            info["disk_free_gb"] = round(disk.free / (1024**3), 2)
            info["disk_percent"] = disk.percent
            
        except ImportError:
            pass
        
        return info
    
    @mcp_tool(description="List running processes")
    async def list_processes(
        self,
        filter_name: str | None = None,
        limit: int = 20,
    ) -> list[ProcessInfo]:
        """
        List running processes.
        
        Args:
            filter_name: Filter by process name
            limit: Maximum number of processes
            
        Returns:
            List of process info
        """
        processes = []
        
        try:
            import psutil
            
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status", "create_time"]):
                try:
                    info = proc.info
                    
                    # Apply filter
                    if filter_name and filter_name.lower() not in info["name"].lower():
                        continue
                    
                    memory_mb = info["memory_info"].rss / (1024 * 1024) if info["memory_info"] else 0
                    
                    processes.append(ProcessInfo(
                        pid=info["pid"],
                        name=info["name"],
                        cpu_percent=info["cpu_percent"] or 0,
                        memory_mb=round(memory_mb, 2),
                        status=info["status"],
                        created=datetime.fromtimestamp(info["create_time"]) if info["create_time"] else None,
                    ))
                    
                    if len(processes) >= limit:
                        break
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except ImportError:
            # Fallback to ps command
            result = await self.run_command("ps aux | head -20")
            if result.success:
                logger.info("psutil not available, using ps command")
        
        # Sort by CPU usage
        processes.sort(key=lambda p: p.cpu_percent, reverse=True)
        
        return processes[:limit]
    
    @mcp_tool(description="Check if a port is in use")
    async def check_port(
        self,
        port: int,
        host: str = "localhost",
    ) -> dict[str, Any]:
        """
        Check if a port is in use.
        
        Args:
            port: Port number
            host: Host to check
            
        Returns:
            Port status information
        """
        import socket
        
        result = {
            "port": port,
            "host": host,
            "in_use": False,
            "process": None,
        }
        
        # Check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        
        try:
            sock.connect((host, port))
            result["in_use"] = True
        except (socket.timeout, ConnectionRefusedError):
            result["in_use"] = False
        finally:
            sock.close()
        
        # Try to find process using the port
        if result["in_use"]:
            try:
                import psutil
                
                for conn in psutil.net_connections():
                    if conn.laddr.port == port:
                        try:
                            proc = psutil.Process(conn.pid)
                            result["process"] = {
                                "pid": conn.pid,
                                "name": proc.name(),
                            }
                            break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except ImportError:
                pass
        
        return result
    
    @mcp_tool(description="Get environment variables")
    async def get_env(
        self,
        prefix: str | None = None,
    ) -> dict[str, str]:
        """
        Get environment variables.
        
        Args:
            prefix: Filter by prefix
            
        Returns:
            Environment variables
        """
        env = dict(os.environ)
        
        if prefix:
            env = {k: v for k, v in env.items() if k.startswith(prefix)}
        
        # Redact sensitive values
        sensitive_keys = ["KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL"]
        for key in env:
            for sensitive in sensitive_keys:
                if sensitive in key.upper():
                    env[key] = "[REDACTED]"
                    break
        
        return env
    
    @mcp_tool(description="Watch a file for changes")
    async def watch_file(
        self,
        path: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Watch a file for changes.
        
        Args:
            path: File path to watch
            timeout: Maximum time to watch
            
        Returns:
            Change information
        """
        from pathlib import Path
        
        file_path = Path(path)
        
        if not file_path.exists():
            return {"error": f"File not found: {path}"}
        
        initial_stat = file_path.stat()
        initial_mtime = initial_stat.st_mtime
        initial_size = initial_stat.st_size
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                return {
                    "changed": False,
                    "path": path,
                    "watched_seconds": timeout,
                }
            
            await asyncio.sleep(0.5)
            
            try:
                current_stat = file_path.stat()
                if current_stat.st_mtime != initial_mtime or current_stat.st_size != initial_size:
                    return {
                        "changed": True,
                        "path": path,
                        "watched_seconds": elapsed,
                        "old_size": initial_size,
                        "new_size": current_stat.st_size,
                    }
            except FileNotFoundError:
                return {
                    "changed": True,
                    "path": path,
                    "deleted": True,
                    "watched_seconds": elapsed,
                }
    
    def get_tools(self) -> list[Callable]:
        """Get all tools as a list of callables."""
        return [
            self.run_command,
            self.docker_ps,
            self.docker_logs,
            self.docker_exec,
            self.system_info,
            self.list_processes,
            self.check_port,
            self.get_env,
            self.watch_file,
        ]
