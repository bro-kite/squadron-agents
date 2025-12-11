"""
Agent-to-Agent (A2A) Protocol Implementation

Provides horizontal orchestration between autonomous agents.
Based on the A2A protocol specification for multi-agent coordination.

Key concepts:
- Agent Cards: JSON manifests advertising agent capabilities
- Task Lifecycle: State machine for task delegation
- JSON-RPC 2.0: Communication protocol over HTTP
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class TaskState(str, Enum):
    """
    Task lifecycle states per A2A protocol.
    
    State machine:
    REQUESTED -> ACCEPTED -> RUNNING -> COMPLETED
                    |           |
                    v           v
                 REJECTED    FAILED
    """
    REQUESTED = "requested"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """A capability that an agent can perform."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """
    Agent Card - A2A discovery manifest.
    
    Published at .well-known/agent-card.json to advertise
    an agent's capabilities to the network.
    """
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: list[AgentCapability] = field(default_factory=list)
    
    # Topics this agent handles
    topics: list[str] = field(default_factory=list)
    
    # Authentication
    auth_type: str = "bearer"  # bearer, api_key, none
    auth_url: str | None = None
    
    # Endpoints
    base_url: str = ""
    task_endpoint: str = "/tasks"
    status_endpoint: str = "/tasks/{task_id}"
    webhook_supported: bool = True
    streaming_supported: bool = False
    
    # Metadata
    owner: str = ""
    contact: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "inputSchema": cap.input_schema,
                    "outputSchema": cap.output_schema,
                    "tags": cap.tags,
                }
                for cap in self.capabilities
            ],
            "topics": self.topics,
            "auth": {
                "type": self.auth_type,
                "url": self.auth_url,
            },
            "endpoints": {
                "base": self.base_url,
                "tasks": self.task_endpoint,
                "status": self.status_endpoint,
            },
            "features": {
                "webhooks": self.webhook_supported,
                "streaming": self.streaming_supported,
            },
            "metadata": {
                "owner": self.owner,
                "contact": self.contact,
                "createdAt": self.created_at.isoformat(),
            },
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        """Create from dictionary."""
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap.get("description", ""),
                input_schema=cap.get("inputSchema", {}),
                output_schema=cap.get("outputSchema", {}),
                tags=cap.get("tags", []),
            )
            for cap in data.get("capabilities", [])
        ]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            capabilities=capabilities,
            topics=data.get("topics", []),
            auth_type=data.get("auth", {}).get("type", "none"),
            auth_url=data.get("auth", {}).get("url"),
            base_url=data.get("endpoints", {}).get("base", ""),
            task_endpoint=data.get("endpoints", {}).get("tasks", "/tasks"),
            status_endpoint=data.get("endpoints", {}).get("status", "/tasks/{task_id}"),
            webhook_supported=data.get("features", {}).get("webhooks", False),
            streaming_supported=data.get("features", {}).get("streaming", False),
            owner=data.get("metadata", {}).get("owner", ""),
            contact=data.get("metadata", {}).get("contact", ""),
        )


@dataclass
class A2ATask:
    """
    A task delegated between agents.
    """
    id: UUID = field(default_factory=uuid4)
    
    # Task details
    capability: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    
    # State
    state: TaskState = TaskState.REQUESTED
    progress: float = 0.0  # 0.0 to 1.0
    
    # Results
    output_data: dict[str, Any] | None = None
    error: str | None = None
    
    # Routing
    from_agent: str = ""
    to_agent: str = ""
    
    # Callbacks
    webhook_url: str | None = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "capability": self.capability,
            "input": self.input_data,
            "state": self.state.value,
            "progress": self.progress,
            "output": self.output_data,
            "error": self.error,
            "fromAgent": self.from_agent,
            "toAgent": self.to_agent,
            "webhookUrl": self.webhook_url,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }


class A2AAgent:
    """
    A2A Agent - Handles agent-to-agent communication.
    
    Provides both client (delegating tasks) and server (receiving tasks)
    functionality for the A2A protocol.
    
    Example:
        ```python
        # Create an agent
        agent = A2AAgent(
            card=AgentCard(
                id="researcher",
                name="Research Agent",
                description="Performs web research",
                capabilities=[
                    AgentCapability(
                        name="search",
                        description="Search the web",
                    )
                ],
            )
        )
        
        # Register capability handlers
        @agent.capability("search")
        async def handle_search(task: A2ATask) -> dict:
            # Perform search
            return {"results": [...]}
        
        # Delegate to another agent
        result = await agent.delegate(
            agent_url="https://other-agent.example.com",
            capability="analyze",
            input_data={"text": "..."},
        )
        ```
    """
    
    def __init__(
        self,
        card: AgentCard,
        http_client: Any | None = None,
    ):
        """
        Initialize the A2A agent.
        
        Args:
            card: This agent's card
            http_client: HTTP client for outbound requests
        """
        self.card = card
        self._http_client = http_client
        
        # Capability handlers
        self._handlers: dict[str, Callable[[A2ATask], Awaitable[dict[str, Any]]]] = {}
        
        # Known agents (discovered via cards)
        self._known_agents: dict[str, AgentCard] = {}
        
        # Active tasks (both incoming and outgoing)
        self._incoming_tasks: dict[UUID, A2ATask] = {}
        self._outgoing_tasks: dict[UUID, A2ATask] = {}
    
    def capability(
        self,
        name: str,
    ) -> Callable[[Callable[[A2ATask], Awaitable[dict[str, Any]]]], Callable[[A2ATask], Awaitable[dict[str, Any]]]]:
        """
        Decorator to register a capability handler.
        
        Example:
            ```python
            @agent.capability("search")
            async def handle_search(task: A2ATask) -> dict:
                return {"results": [...]}
            ```
        """
        def decorator(
            func: Callable[[A2ATask], Awaitable[dict[str, Any]]]
        ) -> Callable[[A2ATask], Awaitable[dict[str, Any]]]:
            self._handlers[name] = func
            return func
        return decorator
    
    async def discover(self, agent_url: str) -> AgentCard | None:
        """
        Discover an agent by fetching its card.
        
        Args:
            agent_url: Base URL of the agent
            
        Returns:
            The agent's card, or None if discovery failed
        """
        try:
            card_url = f"{agent_url.rstrip('/')}/.well-known/agent-card.json"
            
            if self._http_client:
                async with self._http_client.get(card_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        card = AgentCard.from_dict(data)
                        card.base_url = agent_url
                        self._known_agents[card.id] = card
                        logger.info("Discovered agent", agent_id=card.id, name=card.name)
                        return card
            else:
                # Fallback to httpx if available
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(card_url)
                        if response.status_code == 200:
                            data = response.json()
                            card = AgentCard.from_dict(data)
                            card.base_url = agent_url
                            self._known_agents[card.id] = card
                            logger.info("Discovered agent", agent_id=card.id, name=card.name)
                            return card
                except ImportError:
                    logger.warning("No HTTP client available for discovery")
                    
        except Exception as e:
            logger.warning("Agent discovery failed", url=agent_url, error=str(e))
        
        return None
    
    async def delegate(
        self,
        agent_url: str | None = None,
        agent_id: str | None = None,
        capability: str = "",
        input_data: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        wait: bool = True,
        timeout: float = 300.0,
    ) -> A2ATask:
        """
        Delegate a task to another agent.
        
        Args:
            agent_url: URL of the target agent (or use agent_id)
            agent_id: ID of a known agent
            capability: The capability to invoke
            input_data: Input data for the task
            webhook_url: URL for status callbacks
            wait: Whether to wait for completion
            timeout: Maximum wait time in seconds
            
        Returns:
            The completed (or pending) task
        """
        # Resolve agent
        if agent_id and agent_id in self._known_agents:
            target_card = self._known_agents[agent_id]
            agent_url = target_card.base_url
        elif agent_url:
            # Try to discover if not known
            target_card = await self.discover(agent_url)
            if not target_card:
                raise ValueError(f"Could not discover agent at {agent_url}")
        else:
            raise ValueError("Must provide agent_url or agent_id")
        
        # Create task
        task = A2ATask(
            capability=capability,
            input_data=input_data or {},
            from_agent=self.card.id,
            to_agent=target_card.id,
            webhook_url=webhook_url,
        )
        
        self._outgoing_tasks[task.id] = task
        
        # Send task request
        task_url = f"{agent_url.rstrip('/')}{target_card.task_endpoint}"
        
        request_data = {
            "jsonrpc": "2.0",
            "id": str(task.id),
            "method": "tasks/create",
            "params": {
                "capability": capability,
                "input": input_data or {},
                "webhookUrl": webhook_url,
                "fromAgent": self.card.id,
            },
        }
        
        try:
            if self._http_client:
                async with self._http_client.post(task_url, json=request_data) as response:
                    result = await response.json()
            else:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(task_url, json=request_data)
                    result = response.json()
            
            # Update task state
            if "result" in result:
                task.state = TaskState(result["result"].get("state", "accepted"))
            elif "error" in result:
                task.state = TaskState.REJECTED
                task.error = result["error"].get("message", "Unknown error")
                return task
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            logger.error("Task delegation failed", error=str(e))
            return task
        
        # Wait for completion if requested
        if wait and task.state not in (TaskState.COMPLETED, TaskState.FAILED, TaskState.REJECTED):
            task = await self._wait_for_task(agent_url, target_card, task, timeout)
        
        return task
    
    async def _wait_for_task(
        self,
        agent_url: str,
        target_card: AgentCard,
        task: A2ATask,
        timeout: float,
    ) -> A2ATask:
        """Poll for task completion."""
        status_url_template = f"{agent_url.rstrip('/')}{target_card.status_endpoint}"
        status_url = status_url_template.replace("{task_id}", str(task.id))
        
        start_time = asyncio.get_event_loop().time()
        poll_interval = 1.0
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                task.state = TaskState.FAILED
                task.error = "Timeout waiting for task completion"
                break
            
            try:
                if self._http_client:
                    async with self._http_client.get(status_url) as response:
                        result = await response.json()
                else:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(status_url)
                        result = response.json()
                
                if "result" in result:
                    task.state = TaskState(result["result"].get("state", task.state.value))
                    task.progress = result["result"].get("progress", task.progress)
                    task.output_data = result["result"].get("output")
                    task.error = result["result"].get("error")
                    task.updated_at = datetime.utcnow()
                    
                    if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                        task.completed_at = datetime.utcnow()
                        break
                        
            except Exception as e:
                logger.warning("Status poll failed", error=str(e))
            
            await asyncio.sleep(poll_interval)
            # Exponential backoff up to 10 seconds
            poll_interval = min(poll_interval * 1.5, 10.0)
        
        return task
    
    async def handle_task_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle an incoming task request (server-side).
        
        This is called by your HTTP server when receiving a task.
        
        Args:
            request: JSON-RPC request
            
        Returns:
            JSON-RPC response
        """
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})
        
        if method == "tasks/create":
            return await self._handle_create_task(request_id, params)
        elif method == "tasks/status":
            return await self._handle_task_status(request_id, params)
        elif method == "tasks/cancel":
            return await self._handle_cancel_task(request_id, params)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}",
                },
            }
    
    async def _handle_create_task(
        self,
        request_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle task creation."""
        capability = params.get("capability", "")
        
        # Check if we support this capability
        if capability not in self._handlers:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Unknown capability: {capability}",
                },
            }
        
        # Create task
        task = A2ATask(
            id=uuid4(),
            capability=capability,
            input_data=params.get("input", {}),
            from_agent=params.get("fromAgent", ""),
            to_agent=self.card.id,
            webhook_url=params.get("webhookUrl"),
            state=TaskState.ACCEPTED,
        )
        
        self._incoming_tasks[task.id] = task
        
        # Start processing in background
        asyncio.create_task(self._process_task(task))
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "taskId": str(task.id),
                "state": task.state.value,
            },
        }
    
    async def _process_task(self, task: A2ATask) -> None:
        """Process a task asynchronously."""
        task.state = TaskState.RUNNING
        task.updated_at = datetime.utcnow()
        
        try:
            handler = self._handlers[task.capability]
            result = await handler(task)
            
            task.output_data = result
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.utcnow()
            
        except Exception as e:
            task.error = str(e)
            task.state = TaskState.FAILED
            task.completed_at = datetime.utcnow()
            logger.error("Task processing failed", task_id=str(task.id), error=str(e))
        
        task.updated_at = datetime.utcnow()
        
        # Send webhook if configured
        if task.webhook_url:
            await self._send_webhook(task)
    
    async def _send_webhook(self, task: A2ATask) -> None:
        """Send a webhook notification."""
        if not task.webhook_url:
            return
        
        try:
            payload = {
                "taskId": str(task.id),
                "state": task.state.value,
                "output": task.output_data,
                "error": task.error,
            }
            
            if self._http_client:
                await self._http_client.post(task.webhook_url, json=payload)
            else:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(task.webhook_url, json=payload)
                    
        except Exception as e:
            logger.warning("Webhook delivery failed", url=task.webhook_url, error=str(e))
    
    async def _handle_task_status(
        self,
        request_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle task status request."""
        task_id_str = params.get("taskId", "")
        
        try:
            task_id = UUID(task_id_str)
        except ValueError:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid task ID",
                },
            }
        
        task = self._incoming_tasks.get(task_id)
        if not task:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Task not found",
                },
            }
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "taskId": str(task.id),
                "state": task.state.value,
                "progress": task.progress,
                "output": task.output_data,
                "error": task.error,
            },
        }
    
    async def _handle_cancel_task(
        self,
        request_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle task cancellation."""
        task_id_str = params.get("taskId", "")
        
        try:
            task_id = UUID(task_id_str)
        except ValueError:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid task ID",
                },
            }
        
        task = self._incoming_tasks.get(task_id)
        if not task:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Task not found",
                },
            }
        
        if task.state in (TaskState.COMPLETED, TaskState.FAILED):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Task already completed",
                },
            }
        
        task.state = TaskState.CANCELLED
        task.completed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "taskId": str(task.id),
                "state": task.state.value,
            },
        }
    
    def find_agent_for_capability(self, capability: str) -> AgentCard | None:
        """
        Find a known agent that supports a capability.
        
        Args:
            capability: The capability name
            
        Returns:
            An agent card, or None if not found
        """
        for card in self._known_agents.values():
            for cap in card.capabilities:
                if cap.name == capability:
                    return card
        return None
    
    def find_agents_by_topic(self, topic: str) -> list[AgentCard]:
        """
        Find known agents that handle a topic.
        
        Args:
            topic: The topic to search for
            
        Returns:
            List of matching agent cards
        """
        return [
            card for card in self._known_agents.values()
            if topic in card.topics
        ]
    
    @property
    def known_agents(self) -> list[AgentCard]:
        """Get all known agents."""
        return list(self._known_agents.values())
    
    @property
    def pending_tasks(self) -> list[A2ATask]:
        """Get all pending incoming tasks."""
        return [
            task for task in self._incoming_tasks.values()
            if task.state in (TaskState.REQUESTED, TaskState.ACCEPTED, TaskState.RUNNING)
        ]
