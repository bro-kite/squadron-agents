"""
Core Agent Implementation

The main Agent class built on LangGraph's Functional API.
Implements the cyclic Plan → Act → Reflect loop.
"""

from typing import Any, Callable
from uuid import UUID, uuid4

import structlog
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from squadron.core.config import SquadronConfig
from squadron.core.state import (
    AgentPhase,
    AgentState,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)

logger = structlog.get_logger(__name__)


class Agent:
    """
    Squadron Agent - The core agent implementation.
    
    Built on LangGraph's Functional API for improved ergonomics and
    cleaner state management. Supports cyclic loops (Plan → Act → Reflect)
    rather than linear chains.
    
    Example:
        ```python
        agent = Agent(
            name="researcher",
            config=SquadronConfig(),
        )
        result = await agent.run("Analyze this codebase")
        ```
    """

    def __init__(
        self,
        name: str,
        config: SquadronConfig | None = None,
        memory: Any | None = None,  # GraphitiMemory
        reasoner: Any | None = None,  # LATSReasoner
        tools: list[Any] | None = None,
        checkpointer: Any | None = None,
    ):
        """
        Initialize the agent.
        
        Args:
            name: Agent identifier
            config: Configuration object
            memory: Memory system (Graphiti)
            reasoner: Reasoning engine (LATS)
            tools: Available tools
            checkpointer: LangGraph checkpointer for persistence
        """
        self.name = name
        self.config = config or SquadronConfig()
        self.memory = memory
        self.reasoner = reasoner
        self.tools = tools or []
        self.checkpointer = checkpointer or MemorySaver()
        
        # Tool registry for quick lookup
        self._tool_registry: dict[str, Callable] = {}
        for tool in self.tools:
            if hasattr(tool, "name"):
                self._tool_registry[tool.name] = tool
            elif hasattr(tool, "__name__"):
                self._tool_registry[tool.__name__] = tool

        # Build the execution graph
        self._graph = self._build_graph()
        
        logger.info(
            "Agent initialized",
            name=self.name,
            tools=list(self._tool_registry.keys()),
        )

    def _build_graph(self) -> Any:
        """Build the LangGraph execution graph."""
        
        @task
        async def plan(state: AgentState) -> AgentState:
            """
            Planning phase: Analyze the task and generate a plan.
            
            Uses the reasoning engine (LATS) if available to generate
            multiple candidate plans and select the best one.
            """
            logger.debug("Entering planning phase", iteration=state.iteration)
            
            # Update state to planning phase
            state = state.set_phase(AgentPhase.PLANNING)
            
            # Retrieve relevant memory context
            if self.memory:
                try:
                    context = await self.memory.retrieve(
                        query=state.task,
                        session_id=str(state.session_id),
                    )
                    state = state.update_memory_context(context)
                    logger.debug("Retrieved memory context", facts=len(context.get("facts", [])))
                except Exception as e:
                    logger.warning("Memory retrieval failed", error=str(e))
            
            # Generate plan using reasoner or simple LLM call
            if self.reasoner:
                # Use LATS for tree-search planning
                plan_result = await self.reasoner.plan(state)
                state = plan_result
            else:
                # Simple single-shot planning
                plan_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=f"Planning to address: {state.task}",
                    metadata={"phase": "planning"},
                )
                state = state.add_message(plan_message)
            
            return state

        @task
        async def act(state: AgentState) -> AgentState:
            """
            Action phase: Execute tools based on the plan.
            
            Handles tool execution, human-in-the-loop approval,
            and result collection.
            """
            logger.debug("Entering action phase", iteration=state.iteration)
            
            state = state.set_phase(AgentPhase.ACTING)
            
            # Check for pending tool calls
            if not state.pending_tool_calls:
                logger.debug("No pending tool calls")
                return state
            
            # Process each tool call
            for tool_call in state.pending_tool_calls:
                # Check if human approval is required
                if tool_call.tool_name in self.config.governance.require_human_approval:
                    logger.info(
                        "Requesting human approval",
                        tool=tool_call.tool_name,
                    )
                    state = state.request_approval({
                        "tool_call": tool_call.model_dump(),
                        "reason": f"Tool '{tool_call.tool_name}' requires approval",
                    })
                    # Interrupt execution for approval
                    interrupt({"approval_request": state.approval_request})
                    return state
                
                # Execute the tool
                result = await self._execute_tool(tool_call)
                state = state.add_tool_result(result)
                
                # Add tool result as message
                result_message = Message(
                    role=MessageRole.TOOL,
                    content=str(result.result) if result.success else f"Error: {result.error}",
                    name=tool_call.tool_name,
                    metadata={"tool_call_id": str(tool_call.id)},
                )
                state = state.add_message(result_message)
            
            return state

        @task
        async def reflect(state: AgentState) -> AgentState:
            """
            Reflection phase: Evaluate progress and decide next steps.
            
            Determines whether to continue planning, complete the task,
            or handle errors.
            """
            logger.debug("Entering reflection phase", iteration=state.iteration)
            
            state = state.set_phase(AgentPhase.REFLECTING)
            state = state.increment_iteration()
            
            # Check iteration limit
            if state.iteration >= state.max_iterations:
                logger.warning("Max iterations reached", iteration=state.iteration)
                state = state.add_error("Maximum iterations reached")
                return state.set_phase(AgentPhase.ERROR)
            
            # Analyze recent results
            recent_results = state.tool_results[-5:] if state.tool_results else []
            all_successful = all(r.success for r in recent_results)
            
            # Use reasoner for reflection if available
            if self.reasoner:
                reflection_result = await self.reasoner.reflect(state)
                state = reflection_result
            
            # Store interaction in memory
            if self.memory:
                try:
                    await self.memory.store(
                        messages=state.messages,
                        session_id=str(state.session_id),
                    )
                except Exception as e:
                    logger.warning("Memory storage failed", error=str(e))
            
            # Determine next phase
            # This is a simplified check - real implementation would use LLM
            if self._is_task_complete(state):
                return state.set_phase(AgentPhase.COMPLETED)
            
            return state.set_phase(AgentPhase.PLANNING)

        @task
        async def route(state: AgentState) -> Command:
            """Route to the next node based on current phase."""
            if state.phase == AgentPhase.COMPLETED:
                return Command(goto="__end__")
            elif state.phase == AgentPhase.ERROR:
                return Command(goto="__end__")
            elif state.phase == AgentPhase.PLANNING:
                return Command(goto="act")
            elif state.phase == AgentPhase.ACTING:
                return Command(goto="reflect")
            elif state.phase == AgentPhase.REFLECTING:
                return Command(goto="plan")
            else:
                return Command(goto="plan")

        @entrypoint(checkpointer=self.checkpointer)
        async def agent_graph(state: AgentState) -> AgentState:
            """
            Main agent execution graph.
            
            Implements the cyclic Plan → Act → Reflect loop.
            """
            while state.should_continue:
                if state.phase in (AgentPhase.PLANNING, AgentPhase.REFLECTING):
                    state = await plan(state)
                    if not state.should_continue:
                        break
                
                if state.phase == AgentPhase.PLANNING:
                    state = state.set_phase(AgentPhase.ACTING)
                
                if state.phase == AgentPhase.ACTING:
                    state = await act(state)
                    if not state.should_continue:
                        break
                    state = await reflect(state)
            
            return state

        return agent_graph

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        import time
        
        start_time = time.time()
        
        tool = self._tool_registry.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=None,
                error=f"Tool '{tool_call.tool_name}' not found",
            )
        
        try:
            # Execute the tool
            if callable(tool):
                result = await tool(**tool_call.arguments)
            else:
                result = await tool.invoke(tool_call.arguments)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=result,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Tool execution failed",
                tool=tool_call.tool_name,
                error=str(e),
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def _is_task_complete(self, state: AgentState) -> bool:
        """
        Determine if the task is complete.
        
        This is a simplified heuristic. Real implementation would
        use LLM-based evaluation.
        """
        # Check if we have any successful tool results
        if not state.tool_results:
            return False
        
        # Check recent messages for completion indicators
        recent_messages = state.messages[-3:] if state.messages else []
        completion_keywords = ["complete", "done", "finished", "success"]
        
        for msg in recent_messages:
            if any(kw in msg.content.lower() for kw in completion_keywords):
                return True
        
        return False

    async def run(
        self,
        task: str,
        session_id: UUID | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> AgentState:
        """
        Run the agent on a task.
        
        Args:
            task: The task description
            session_id: Optional session ID for continuity
            initial_context: Optional initial context
            
        Returns:
            Final agent state after execution
        """
        logger.info("Starting agent run", task=task[:100])
        
        # Create initial state
        state = AgentState(
            session_id=session_id or uuid4(),
            task=task,
            task_id=uuid4(),
            max_iterations=self.config.governance.max_iterations,
        )
        
        # Add initial user message
        user_message = Message(
            role=MessageRole.USER,
            content=task,
        )
        state = state.add_message(user_message)
        
        # Add initial context if provided
        if initial_context:
            state = state.update_memory_context(initial_context)
        
        # Run the graph
        try:
            final_state = await self._graph(state)
            logger.info(
                "Agent run completed",
                phase=final_state.phase,
                iterations=final_state.iteration,
            )
            return final_state
        except Exception as e:
            logger.error("Agent run failed", error=str(e))
            return state.add_error(str(e))

    async def resume(
        self,
        session_id: UUID,
        approval: bool = True,
    ) -> AgentState:
        """
        Resume a paused agent execution.
        
        Used after human-in-the-loop approval.
        
        Args:
            session_id: Session to resume
            approval: Whether the action was approved
            
        Returns:
            Final agent state after resumption
        """
        logger.info("Resuming agent", session_id=str(session_id), approval=approval)
        
        # Get the checkpointed state
        # This would retrieve from the checkpointer
        # For now, this is a placeholder
        raise NotImplementedError("Resume functionality requires checkpointer integration")

    def register_tool(self, tool: Any) -> None:
        """Register a new tool with the agent."""
        if hasattr(tool, "name"):
            name = tool.name
        elif hasattr(tool, "__name__"):
            name = tool.__name__
        else:
            raise ValueError("Tool must have a 'name' attribute or '__name__'")
        
        self._tool_registry[name] = tool
        self.tools.append(tool)
        logger.info("Tool registered", tool=name)

    def get_tool(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._tool_registry.get(name)

    @property
    def available_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._tool_registry.keys())

    async def load_mcp_tools(self, mcp_host: Any) -> int:
        """
        Load tools from an MCP host.
        
        Args:
            mcp_host: MCPHost instance with connected servers
            
        Returns:
            Number of tools loaded
        """
        from squadron.connectivity.mcp_host import MCPHost
        
        if not isinstance(mcp_host, MCPHost):
            raise TypeError("Expected MCPHost instance")
        
        count = 0
        for tool in mcp_host.get_all_tools():
            # Create a wrapper function for the MCP tool
            async def mcp_tool_wrapper(
                _host=mcp_host,
                _name=tool.name,
                **kwargs
            ):
                return await _host.call_tool(_name, kwargs)
            
            mcp_tool_wrapper.__name__ = tool.name
            mcp_tool_wrapper.__doc__ = tool.description
            
            self._tool_registry[tool.name] = mcp_tool_wrapper
            count += 1
        
        logger.info("Loaded MCP tools", count=count)
        return count

    def register_tool_pack(self, tool_pack: Any) -> int:
        """
        Register all tools from a tool pack.
        
        Args:
            tool_pack: A tool pack instance (CodingTools, ResearchTools, etc.)
            
        Returns:
            Number of tools registered
        """
        if not hasattr(tool_pack, "get_tools"):
            raise TypeError("Tool pack must have a get_tools() method")
        
        tools = tool_pack.get_tools()
        for tool in tools:
            self.register_tool(tool)
        
        logger.info("Registered tool pack", tools=len(tools))
        return len(tools)