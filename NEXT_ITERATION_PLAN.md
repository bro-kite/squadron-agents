# Squadron Agent Framework - Iteration Plan

## Status: Phases 1-4 COMPLETED

**Version**: 0.1.0 (Alpha) → Enhanced
**Status**: Core framework production-ready with full LLM integration

### What's Working Well
- Core agent framework with LangGraph (Plan → Act → Reflect loop)
- All 5 LLM providers (OpenAI, Anthropic, Ollama, HuggingFace, OpenAI-compatible)
- Comprehensive tool packs (Coding, Research, Ops)
- Security features (rate limiting, input validation, command injection prevention)
- Memory system with Graphiti (with in-memory fallback)
- MCP and A2A connectivity protocols
- SICA self-improvement engine
- 115 passing tests

### Areas Needing Enhancement
1. **LATS Reasoning** - Uses stub methods, doesn't generate LLM-based candidates
2. **Task Completion Detection** - Keyword matching heuristic instead of LLM evaluation
3. **Resume Functionality** - Raises NotImplementedError
4. **SSE Transport for MCP** - Not implemented

---

## Iteration Goals

This iteration focuses on **enhancing the reasoning layer** to make the agent truly intelligent rather than just functional.

---

## Phase 1: LLM-Based Action Generation for LATS

**Priority**: HIGH
**Files**: `src/squadron/reasoning/lats.py`

### Current Problem
The `_generate_candidate_calls()` method only yields a single hardcoded tool call using a default tool. This doesn't leverage the LLM to reason about which actions to take.

### Implementation

1. **Add LLM Client to LATSReasoner**
   - Accept an LLM client in the constructor
   - Use it to generate multiple candidate actions

2. **Create Action Generation Prompt**
   ```
   Given the current task and context, generate {n_candidates} possible next actions.
   For each action, provide:
   - thought: Your reasoning for this action
   - tool_name: Which tool to use
   - arguments: The tool arguments as JSON
   - expected_outcome: What you expect to happen
   ```

3. **Implement `_generate_candidate_calls_llm()`**
   - Format the prompt with current state and available tools
   - Parse LLM response into CandidatePlan objects
   - Fall back to current stub behavior if LLM fails

4. **Update ListWiseVerifier Integration**
   - Rank the LLM-generated candidates
   - Select best action based on verification score

### Tests to Add
- Test LLM-based candidate generation with mock LLM
- Test fallback behavior when LLM fails
- Test ranking of multiple candidates

---

## Phase 2: LLM-Based Task Completion Detection

**Priority**: HIGH
**Files**: `src/squadron/core/agent.py`

### Current Problem
`_is_task_complete()` uses simple keyword matching ("complete", "done", "finished"). This is unreliable for real-world tasks.

### Implementation

1. **Create Completion Evaluation Prompt**
   ```
   Task: {task}

   Conversation History:
   {messages}

   Tool Results:
   {tool_results}

   Has this task been completed successfully?
   Respond with JSON: {"completed": true/false, "reason": "..."}
   ```

2. **Add LLM Client to Agent**
   - Optionally inject an LLM client
   - Use for completion detection and potentially other evaluations

3. **Implement `_is_task_complete_llm()`**
   - Call LLM with completion prompt
   - Parse JSON response
   - Log completion reasoning
   - Fall back to heuristic if LLM unavailable

4. **Add Configurable Completion Threshold**
   - Allow setting confidence threshold
   - Support "soft completion" vs "hard completion"

### Tests to Add
- Test LLM-based completion detection with mock
- Test fallback to heuristic
- Test edge cases (empty results, error states)

---

## Phase 3: Implement Resume Functionality

**Priority**: MEDIUM
**Files**: `src/squadron/core/agent.py`

### Current Problem
`resume()` raises `NotImplementedError`. Users can't resume interrupted executions.

### Implementation

1. **Integrate with MemorySaver Checkpointer**
   - Store state with session_id as key
   - Retrieve state on resume

2. **Implement Resume Logic**
   ```python
   async def resume(self, session_id: UUID, approval: bool = True) -> AgentState:
       # Retrieve checkpointed state
       config = {"configurable": {"thread_id": str(session_id)}}
       state = await self.checkpointer.get(config)

       if state is None:
           raise ValueError(f"No checkpointed state for session {session_id}")

       # Update approval status if this was a HITL interrupt
       if state.approval_request and not approval:
           return state.add_error("Action rejected by user")

       # Continue execution from interrupted point
       return await self._graph(state, config)
   ```

3. **Add State Serialization Tests**
   - Ensure AgentState can be serialized/deserialized
   - Test resume after interrupt
   - Test resume with rejection

---

## Phase 4: MCTS Expand and Simulate Functions

**Priority**: MEDIUM
**Files**: `src/squadron/reasoning/mcts.py`, `src/squadron/reasoning/lats.py`

### Current Problem
MCTS controller has `_expand_stub` and `_simulate_stub` that do nothing. Tree search isn't actually happening.

### Implementation

1. **Implement Real Expand Function**
   - Generate multiple possible actions from current state
   - Create child nodes for each action
   - Use LLM to generate diverse action candidates

2. **Implement Real Simulate Function**
   - Evaluate the quality of a state
   - Use LLM to score states or use heuristics
   - Return value in [0, 1] range

3. **Wire Up to LATSReasoner**
   - Replace stub functions with real implementations
   - Add configuration for MCTS budget
   - Enable/disable full tree search via config

4. **Add Rollout Simulation** (Advanced)
   - Simulate future actions without executing tools
   - Score potential trajectories
   - Backpropagate scores through tree

---

## Phase 5: SSE Transport for MCP

**Priority**: LOW
**Files**: `src/squadron/connectivity/mcp_host.py`

### Current Problem
SSE (Server-Sent Events) transport raises `NotImplementedError`.

### Implementation

1. **Add aiohttp-sse-client dependency**
2. **Implement SSE connection handling**
   - Connect to SSE endpoint
   - Parse incoming events
   - Handle reconnection

3. **Wire up to MCPHost**
   - Detect transport type from config
   - Route to appropriate handler

---

## Testing Strategy

### New Test Files to Create
- `tests/test_lats_llm.py` - LLM-based reasoning tests
- `tests/test_completion_detection.py` - Task completion tests
- `tests/test_resume.py` - Resume functionality tests

### Mock Strategy
- Create `MockLLMClient` that returns predictable responses
- Use dependency injection for testability
- Test both success and failure paths

---

## Implementation Status

| Order | Phase | Status | Tests |
|-------|-------|--------|-------|
| 1 | Phase 1: LLM Action Generation | ✅ COMPLETED | 21 tests |
| 2 | Phase 2: LLM Completion Detection | ✅ COMPLETED | 17 tests |
| 3 | Phase 3: Resume Functionality | ✅ COMPLETED | 16 tests |
| 4 | Phase 4: MCTS Expand/Simulate | ✅ COMPLETED | 23 tests |
| 5 | Phase 5: SSE Transport | ⏳ PENDING (Low Priority) | - |

**Total New Tests: 77 passing**

---

## Success Criteria

1. ✅ **Phase 1 Complete**: Agent generates multiple tool candidates using LLM and ranks them
2. ✅ **Phase 2 Complete**: Agent accurately detects task completion using LLM evaluation
3. ✅ **Phase 3 Complete**: Users can resume interrupted agent executions
4. ✅ **Phase 4 Complete**: Full MCTS tree search with rollout simulation working
5. ⏳ **Phase 5 Pending**: MCP servers using SSE transport (low priority)

---

## Dependencies to Add

```toml
# pyproject.toml additions
aiohttp-sse-client = "^0.2.1"  # For SSE transport
```

---

## Configuration Additions

```env
# New configuration options
REASONING_USE_LLM_GENERATION=true
REASONING_USE_LLM_COMPLETION=true
REASONING_MCTS_BUDGET=50
REASONING_COMPLETION_CONFIDENCE=0.8
```
