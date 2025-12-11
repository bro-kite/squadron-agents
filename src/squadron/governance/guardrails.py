"""
Safety Guardrails System

Provides runtime safety checks and guardrails for agent behavior.
Prevents dangerous actions and enforces policy compliance.

Key features:
- Pre-execution checks for tool calls
- Content filtering
- Rate limiting
- Policy enforcement
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable, Pattern
from uuid import UUID, uuid4

import structlog

from squadron.core.state import AgentState, ToolCall
from squadron.core.config import GovernanceConfig

logger = structlog.get_logger(__name__)


class GuardrailAction(str, Enum):
    """Action to take when a guardrail is triggered."""
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    WARN = "warn"
    MODIFY = "modify"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    
    guardrail_name: str
    action: GuardrailAction
    passed: bool
    
    # Details
    reason: str = ""
    modified_value: Any = None
    
    # Metadata
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "guardrailName": self.guardrail_name,
            "action": self.action.value,
            "passed": self.passed,
            "reason": self.reason,
            "checkedAt": self.checked_at.isoformat(),
        }


@dataclass
class Guardrail:
    """
    A single guardrail rule.
    
    Guardrails can check:
    - Tool calls before execution
    - Agent output before returning
    - State transitions
    """
    
    name: str
    description: str
    
    # Check function
    check_fn: Callable[[Any], Awaitable[GuardrailResult]] | None = None
    
    # Simple pattern matching (alternative to check_fn)
    blocked_patterns: list[Pattern] = field(default_factory=list)
    
    # Action when triggered
    action: GuardrailAction = GuardrailAction.BLOCK
    
    # Scope
    applies_to_tools: list[str] = field(default_factory=list)  # Empty = all tools
    applies_to_content: bool = False
    
    # Priority (higher = checked first)
    priority: int = 0
    
    # Enabled
    enabled: bool = True


class SafetyGuardrails:
    """
    Safety Guardrails Manager.
    
    Manages a collection of guardrails and provides methods to
    check tool calls and content against them.
    
    Example:
        ```python
        guardrails = SafetyGuardrails(config=GovernanceConfig())
        
        # Add a custom guardrail
        guardrails.add_guardrail(Guardrail(
            name="no_delete",
            description="Prevent file deletion",
            applies_to_tools=["delete_file"],
            action=GuardrailAction.REQUIRE_APPROVAL,
        ))
        
        # Check a tool call
        result = await guardrails.check_tool_call(tool_call)
        if not result.passed:
            print(f"Blocked: {result.reason}")
        ```
    """
    
    def __init__(self, config: GovernanceConfig | None = None):
        """
        Initialize the guardrails manager.
        
        Args:
            config: Governance configuration
        """
        self.config = config or GovernanceConfig()
        self._guardrails: list[Guardrail] = []
        self._rate_limits: dict[str, list[datetime]] = {}
        
        # Initialize default guardrails
        if self.config.enable_guardrails:
            self._init_default_guardrails()
    
    def _init_default_guardrails(self) -> None:
        """Initialize default safety guardrails."""
        
        # Dangerous tools require approval
        for tool_name in self.config.require_human_approval:
            self.add_guardrail(Guardrail(
                name=f"approval_{tool_name}",
                description=f"Require approval for {tool_name}",
                applies_to_tools=[tool_name],
                action=GuardrailAction.REQUIRE_APPROVAL,
                priority=100,
            ))
        
        # Block shell injection patterns
        self.add_guardrail(Guardrail(
            name="shell_injection",
            description="Block potential shell injection",
            blocked_patterns=[
                re.compile(r";\s*rm\s+-rf", re.IGNORECASE),
                re.compile(r"\|\s*bash", re.IGNORECASE),
                re.compile(r">\s*/dev/", re.IGNORECASE),
                re.compile(r"\$\(.*\)", re.IGNORECASE),
                re.compile(r"`.*`", re.IGNORECASE),
            ],
            action=GuardrailAction.BLOCK,
            applies_to_content=True,
            priority=90,
        ))
        
        # Block sensitive data patterns
        self.add_guardrail(Guardrail(
            name="sensitive_data",
            description="Block exposure of sensitive data",
            blocked_patterns=[
                re.compile(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
                re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", re.IGNORECASE),
                re.compile(r"(?:sk-|pk_live_|sk_live_)[a-zA-Z0-9]{20,}", re.IGNORECASE),
            ],
            action=GuardrailAction.BLOCK,
            applies_to_content=True,
            priority=85,
        ))
        
        # Block SQL injection patterns
        self.add_guardrail(Guardrail(
            name="sql_injection",
            description="Block potential SQL injection",
            blocked_patterns=[
                re.compile(r";\s*DROP\s+TABLE", re.IGNORECASE),
                re.compile(r";\s*DELETE\s+FROM", re.IGNORECASE),
                re.compile(r"'\s*OR\s+'1'\s*=\s*'1", re.IGNORECASE),
                re.compile(r"--\s*$", re.MULTILINE),
            ],
            action=GuardrailAction.BLOCK,
            applies_to_content=True,
            priority=80,
        ))
        
        logger.info("Default guardrails initialized", count=len(self._guardrails))
    
    def add_guardrail(self, guardrail: Guardrail) -> None:
        """
        Add a guardrail.
        
        Args:
            guardrail: The guardrail to add
        """
        self._guardrails.append(guardrail)
        # Sort by priority (descending)
        self._guardrails.sort(key=lambda g: g.priority, reverse=True)
        logger.debug("Added guardrail", name=guardrail.name)
    
    def remove_guardrail(self, name: str) -> bool:
        """
        Remove a guardrail by name.
        
        Args:
            name: Guardrail name
            
        Returns:
            True if removed, False if not found
        """
        original_count = len(self._guardrails)
        self._guardrails = [g for g in self._guardrails if g.name != name]
        return len(self._guardrails) < original_count
    
    def enable_guardrail(self, name: str) -> bool:
        """Enable a guardrail by name."""
        for g in self._guardrails:
            if g.name == name:
                g.enabled = True
                return True
        return False
    
    def disable_guardrail(self, name: str) -> bool:
        """Disable a guardrail by name."""
        for g in self._guardrails:
            if g.name == name:
                g.enabled = False
                return True
        return False
    
    async def check_tool_call(
        self,
        tool_call: ToolCall,
        state: AgentState | None = None,
    ) -> GuardrailResult:
        """
        Check a tool call against all applicable guardrails.
        
        Args:
            tool_call: The tool call to check
            state: Current agent state (optional)
            
        Returns:
            The result of the check
        """
        tool_name = tool_call.tool_name
        args_str = str(tool_call.arguments)
        
        for guardrail in self._guardrails:
            if not guardrail.enabled:
                continue
            
            # Check if guardrail applies to this tool
            if guardrail.applies_to_tools:
                if tool_name not in guardrail.applies_to_tools:
                    continue
            
            # Run custom check function
            if guardrail.check_fn:
                result = await guardrail.check_fn(tool_call)
                if not result.passed:
                    return result
            
            # Check blocked patterns against arguments
            if guardrail.blocked_patterns:
                for pattern in guardrail.blocked_patterns:
                    if pattern.search(args_str):
                        return GuardrailResult(
                            guardrail_name=guardrail.name,
                            action=guardrail.action,
                            passed=False,
                            reason=f"Blocked pattern detected: {pattern.pattern}",
                        )
        
        # All checks passed
        return GuardrailResult(
            guardrail_name="",
            action=GuardrailAction.ALLOW,
            passed=True,
            reason="All guardrails passed",
        )
    
    async def check_content(
        self,
        content: str,
        context: str = "output",
    ) -> GuardrailResult:
        """
        Check content against content guardrails.
        
        Args:
            content: The content to check
            context: Context for the check (e.g., "output", "input")
            
        Returns:
            The result of the check
        """
        for guardrail in self._guardrails:
            if not guardrail.enabled:
                continue
            
            if not guardrail.applies_to_content:
                continue
            
            # Check blocked patterns
            for pattern in guardrail.blocked_patterns:
                match = pattern.search(content)
                if match:
                    return GuardrailResult(
                        guardrail_name=guardrail.name,
                        action=guardrail.action,
                        passed=False,
                        reason=f"Blocked pattern in {context}: {pattern.pattern}",
                    )
            
            # Run custom check
            if guardrail.check_fn:
                result = await guardrail.check_fn(content)
                if not result.passed:
                    return result
        
        return GuardrailResult(
            guardrail_name="",
            action=GuardrailAction.ALLOW,
            passed=True,
            reason="Content passed all guardrails",
        )
    
    async def check_rate_limit(
        self,
        key: str,
        max_calls: int,
        window_seconds: float,
    ) -> GuardrailResult:
        """
        Check rate limiting.
        
        Args:
            key: Rate limit key (e.g., tool name, user ID)
            max_calls: Maximum calls allowed
            window_seconds: Time window in seconds
            
        Returns:
            The result of the check
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Get or create call history
        if key not in self._rate_limits:
            self._rate_limits[key] = []
        
        # Clean old entries
        self._rate_limits[key] = [
            t for t in self._rate_limits[key]
            if t > window_start
        ]
        
        # Check limit
        if len(self._rate_limits[key]) >= max_calls:
            return GuardrailResult(
                guardrail_name="rate_limit",
                action=GuardrailAction.BLOCK,
                passed=False,
                reason=f"Rate limit exceeded: {len(self._rate_limits[key])}/{max_calls} in {window_seconds}s",
            )
        
        # Record this call
        self._rate_limits[key].append(now)
        
        return GuardrailResult(
            guardrail_name="rate_limit",
            action=GuardrailAction.ALLOW,
            passed=True,
            reason=f"Rate limit OK: {len(self._rate_limits[key])}/{max_calls}",
        )
    
    async def check_all(
        self,
        tool_call: ToolCall | None = None,
        content: str | None = None,
        state: AgentState | None = None,
    ) -> list[GuardrailResult]:
        """
        Run all applicable guardrail checks.
        
        Args:
            tool_call: Tool call to check (optional)
            content: Content to check (optional)
            state: Agent state (optional)
            
        Returns:
            List of all check results
        """
        results = []
        
        if tool_call:
            result = await self.check_tool_call(tool_call, state)
            results.append(result)
        
        if content:
            result = await self.check_content(content)
            results.append(result)
        
        return results
    
    def get_guardrails(self) -> list[Guardrail]:
        """Get all guardrails."""
        return list(self._guardrails)
    
    def get_guardrail(self, name: str) -> Guardrail | None:
        """Get a guardrail by name."""
        for g in self._guardrails:
            if g.name == name:
                return g
        return None
    
    @property
    def enabled_count(self) -> int:
        """Count of enabled guardrails."""
        return sum(1 for g in self._guardrails if g.enabled)


class ContentFilter:
    """
    Content filtering utilities.
    
    Provides methods for sanitizing and filtering content.
    """
    
    @staticmethod
    def redact_secrets(content: str) -> str:
        """
        Redact potential secrets from content.
        
        Args:
            content: Content to redact
            
        Returns:
            Redacted content
        """
        patterns = [
            (re.compile(r'(api[_-]?key\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE), r'\1[REDACTED]\3'),
            (re.compile(r'(password\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE), r'\1[REDACTED]\3'),
            (re.compile(r'(secret\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE), r'\1[REDACTED]\3'),
            (re.compile(r'(token\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE), r'\1[REDACTED]\3'),
            (re.compile(r'(sk-[a-zA-Z0-9]{20,})', re.IGNORECASE), '[REDACTED_API_KEY]'),
            (re.compile(r'(-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----).*?(-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----)', re.DOTALL), r'\1[REDACTED]\2'),
        ]
        
        result = content
        for pattern, replacement in patterns:
            result = pattern.sub(replacement, result)
        
        return result
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize a file path to prevent directory traversal.
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized path
        """
        # Remove directory traversal attempts
        sanitized = re.sub(r'\.\./', '', path)
        sanitized = re.sub(r'\.\.\\', '', sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    @staticmethod
    def sanitize_sql(query: str) -> str:
        """
        Basic SQL sanitization (for logging, not security).
        
        Args:
            query: SQL query
            
        Returns:
            Sanitized query
        """
        # This is for logging purposes only - use parameterized queries for security
        dangerous_patterns = [
            (re.compile(r';\s*DROP\s+', re.IGNORECASE), '; /* BLOCKED: DROP */ '),
            (re.compile(r';\s*DELETE\s+', re.IGNORECASE), '; /* BLOCKED: DELETE */ '),
            (re.compile(r';\s*TRUNCATE\s+', re.IGNORECASE), '; /* BLOCKED: TRUNCATE */ '),
        ]
        
        result = query
        for pattern, replacement in dangerous_patterns:
            result = pattern.sub(replacement, result)
        
        return result
