# agentflow.a2a_integration

> **Official bridge between agentflow and the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A) via the `a2a-sdk`.**

This package lets you **expose** any agentflow `CompiledGraph` as an A2A-compliant agent, and **call** remote A2A agents from within agentflow graphs — all with minimal boilerplate.

---

## Table of Contents

- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Module Reference](#module-reference)
  - [executor.py — AgentFlowExecutor](#executorpy--agentflowexecutor)
  - [server.py — Server Helpers](#serverpy--server-helpers)
  - [client.py — Client Helpers](#clientpy--client-helpers)
- [Quick Start](#quick-start)
  - [Serving a Graph as an A2A Agent](#serving-a-graph-as-an-a2a-agent)
  - [Calling a Remote A2A Agent](#calling-a-remote-a2a-agent)
  - [Using a Remote Agent as a Graph Node](#using-a-remote-agent-as-a-graph-node)
- [Conversation Memory (context_id)](#conversation-memory-context_id)
- [Streaming](#streaming)
- [Custom Executors](#custom-executors)
- [Examples](#examples)
- [API Summary](#api-summary)

---

## Installation

The A2A integration requires the `a2a-sdk` extra:

```bash
pip install agentflow[a2a_sdk]
```

This installs `a2a-sdk`, `httpx`, `uvicorn`, and all transitive dependencies.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                        A2A Client                             │
│  (any A2A-compliant client, browser, or another agent)        │
└──────────────┬────────────────────────────────────────────────┘
               │  JSON-RPC / SSE over HTTP
               ▼
┌──────────────────────────────────────────────────────────────┐
│  a2a-sdk layer  (transport, JSON-RPC, task lifecycle, SSE)   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  DefaultRequestHandler  +  InMemoryTaskStore           │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │ calls execute()                       │
│  ┌────────────────────▼───────────────────────────────────┐  │
│  │  AgentFlowExecutor (executor.py)                       │  │
│  │  ─ extracts user text from A2A message parts           │  │
│  │  ─ resolves thread_id from context_id                  │  │
│  │  ─ runs CompiledGraph.ainvoke() or .astream()          │  │
│  │  ─ pushes results back via TaskUpdater                 │  │
│  └────────────────────┬───────────────────────────────────┘  │
└───────────────────────┼──────────────────────────────────────┘
                        │
               ┌────────▼────────┐
               │  CompiledGraph  │  ← your agentflow graph
               │  (any topology) │
               └─────────────────┘
```

**Key insight:** The `a2a-sdk` owns all transport (HTTP, JSON-RPC, SSE, task state machines). Agentflow owns all agent logic (LLM calls, tool execution, state management, checkpointing). `AgentFlowExecutor` is the sole bridge connecting the two.

---

## Module Reference

### `executor.py` — AgentFlowExecutor

The core bridge class. Implements the `a2a-sdk`'s `AgentExecutor` interface.

```python
class AgentFlowExecutor(AgentExecutor):
    def __init__(
        self,
        compiled_graph: CompiledGraph,
        config: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> None: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `compiled_graph` | `CompiledGraph` | A fully compiled agentflow graph |
| `config` | `dict` | Base config forwarded to `ainvoke`/`astream` (e.g. `recursion_limit`) |
| `streaming` | `bool` | When `True`, uses `astream` + sends `TaskState.working` progress events |

**What it does on each A2A request:**

1. Creates a `TaskUpdater` and marks the task as `submitted` → `working`
2. Extracts user text from the A2A message parts via `context.get_user_input()`
3. Wraps the text as an agentflow `Message` list
4. Resolves `thread_id` from `context.context_id` (falling back to `context.task_id`) for checkpointer-based conversation memory
5. Calls `graph.ainvoke()` (blocking) or `graph.astream()` (streaming)
6. Extracts the last assistant message from the result state
7. Pushes it back as an A2A artifact via `updater.add_artifact()` and marks `completed`

**Text extraction** walks backwards through `state.context` looking for the last `role="assistant"` message — this works regardless of how many nodes the graph has.

**Error handling** catches all exceptions and pushes a `TaskState.failed` status with the error message.

---

### `server.py` — Server Helpers

Three convenience functions to expose a graph as an A2A HTTP server:

#### `make_agent_card()`

Builds an `AgentCard` (the A2A discovery descriptor served at `/.well-known/agent-card.json`).

```python
def make_agent_card(
    name: str,
    description: str,
    url: str,
    *,
    skills: list[AgentSkill] | None = None,
    streaming: bool = False,
    version: str = "1.0.0",
) -> AgentCard: ...
```

If no `skills` are provided, a default `"run_graph"` skill is created automatically.

#### `build_a2a_app()`

Returns a **Starlette ASGI app** — useful when you need to compose the A2A endpoint with other routes (e.g. mount inside FastAPI) or run it in tests.

```python
def build_a2a_app(
    compiled_graph: CompiledGraph,
    agent_card: AgentCard,
    *,
    streaming: bool = False,
    executor_config: dict[str, Any] | None = None,
) -> Starlette: ...
```

#### `create_a2a_server()`

**One-call blocking server** — builds the app and starts `uvicorn`. Ideal for standalone agents.

```python
def create_a2a_server(
    compiled_graph: CompiledGraph,
    agent_card: AgentCard,
    *,
    host: str = "0.0.0.0",
    port: int = 9999,
    streaming: bool = False,
    executor_config: dict[str, Any] | None = None,
) -> None: ...
```

---

### `client.py` — Client Helpers

Utilities for calling **remote** A2A agents from within agentflow graphs.

#### `delegate_to_a2a_agent()`

Async one-shot helper — send text, get text back.

```python
async def delegate_to_a2a_agent(
    url: str,
    text: str,
    *,
    timeout: float = 30.0,
) -> str: ...
```

Sends a single `TextPart` message to the remote agent and returns the response text. Raises `RuntimeError` if the agent returns an error or no text content.

#### `create_a2a_client_node()`

Factory that returns an **agentflow graph node function** wrapping a remote A2A agent. The returned callable has the standard node signature `(state, config) -> list[Message]`.

```python
def create_a2a_client_node(
    url: str,
    *,
    timeout: float = 30.0,
    response_role: str = "assistant",
) -> Callable: ...
```

**Usage in a graph:**

```python
from agentflow.a2a_integration import create_a2a_client_node

graph.add_node("remote_agent", create_a2a_client_node("http://localhost:9999"))
graph.add_edge("some_node", "remote_agent")
graph.add_edge("remote_agent", END)
```

The node reads the last message from `state.context`, forwards its text to the remote A2A agent, and returns the response as a new `Message`.

---

## Quick Start

### Serving a Graph as an A2A Agent

```python
from agentflow.graph import StateGraph
from agentflow.state import AgentState
from agentflow.utils.constants import END
from agentflow.a2a_integration import (
    create_a2a_server,
    make_agent_card,
)

# 1. Build your agentflow graph
async def my_node(state: AgentState, config: dict):
    from agentflow.state import Message
    user_text = state.context[-1].text() if state.context else ""
    return [Message.text_message(f"You said: {user_text}", role="assistant")]

graph = StateGraph[AgentState](AgentState())
graph.add_node("main", my_node)
graph.set_entry_point("main")
graph.add_edge("main", END)
compiled = graph.compile()

# 2. Create the agent card
card = make_agent_card(
    name="EchoAgent",
    description="Echoes back whatever you say",
    url="http://localhost:9999",
)

# 3. Start the A2A server (blocking)
create_a2a_server(compiled, card, port=9999)
```

The agent is now discoverable at `http://localhost:9999/.well-known/agent-card.json` and accepts JSON-RPC requests at `http://localhost:9999/`.

### Calling a Remote A2A Agent

```python
from agentflow.a2a_integration import delegate_to_a2a_agent

response = await delegate_to_a2a_agent(
    "http://localhost:9999",
    "Hello, agent!",
)
print(response)  # "You said: Hello, agent!"
```

### Using a Remote Agent as a Graph Node

```python
from agentflow.a2a_integration import create_a2a_client_node
from agentflow.graph import StateGraph
from agentflow.state import AgentState
from agentflow.utils.constants import END

graph = StateGraph[AgentState](AgentState())
graph.add_node("user_input", some_input_node)
graph.add_node("remote", create_a2a_client_node("http://localhost:9999"))
graph.set_entry_point("user_input")
graph.add_edge("user_input", "remote")
graph.add_edge("remote", END)

compiled = graph.compile()
result = await compiled.ainvoke({"messages": [...]})
```

---

## Conversation Memory (context_id)

The A2A protocol has two key identifiers:

| Field | Purpose |
|-------|---------|
| `task_id` | Unique per message/request — changes every call |
| `context_id` | Stable per conversation session — stays the same across turns |

`AgentFlowExecutor` uses `context_id` as the agentflow checkpointer's `thread_id`. This means:

- **Same `context_id`** → same checkpointer thread → conversation history is restored
- **Different `context_id`** → fresh thread → new conversation
- **No `context_id`** → falls back to `task_id` → one-shot (no memory)

```
Turn 1:  context_id="abc" → thread_id="abc" → checkpointer saves state
Turn 2:  context_id="abc" → thread_id="abc" → checkpointer restores state ✓
Turn 3:  context_id="xyz" → thread_id="xyz" → fresh conversation
```

The client is responsible for sending a consistent `context_id` across turns. The `a2a-sdk` `ClientFactory` handles this automatically when configured.

### Sub-agent context_id isolation

When an agent delegates to another agent (e.g. PlannerAgent → CurrencyAgent), **do not forward the caller's `context_id` directly**. Both agents would share the same checkpointer thread, causing their conversation histories to collide.

Instead, derive a namespaced `context_id` for each sub-agent:

```python
# Inside a delegation tool
planner_ctx_id = state.a2a_context_id if state else ""
currency_ctx_id = f"{planner_ctx_id}:currency" if planner_ctx_id else ""
result = await _send_to_currency_agent(query, context_id=currency_ctx_id)
```

This ensures:
- PlannerAgent checkpoints under `context_id` (e.g. `"abc"`)
- CurrencyAgent checkpoints under `"abc:currency"`
- Both maintain independent, stable conversation memory across turns
- The currency result is still returned to the planner's state as a tool message

---

## Streaming

When `streaming=True`, the executor uses `CompiledGraph.astream()` instead of `ainvoke()`:

```python
# Server side
executor = AgentFlowExecutor(compiled, streaming=True)

# Or via the server helper
create_a2a_server(compiled, card, streaming=True)
```

**What happens during streaming:**

1. The graph yields `StreamChunk` objects as each node completes
2. For each chunk with a message, the executor pushes a `TaskState.working` status update via SSE
3. After the stream completes, the final text is emitted as an artifact with `TaskState.completed`

The client observes real-time progress via Server-Sent Events (SSE).

**Note:** This is A2A-level streaming — one SSE `TaskState.working` event is sent per `StreamEvent.MESSAGE` chunk emitted by the graph (i.e. whenever a node produces a message). Token-level LLM streaming is a separate concern from the A2A transport layer.

---

## Custom Executors

For advanced use cases (e.g. detecting `input_required`, merging custom state fields), subclass `AgentFlowExecutor`:

```python
from agentflow.a2a_integration.executor import AgentFlowExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState, TextPart

class MyCustomExecutor(AgentFlowExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.submit()
        await updater.start_work()

        user_text = context.get_user_input() if context.message else ""
        messages = [AFMessage.text_message(user_text, role="user")]

        run_config = {
            "thread_id": context.context_id or context.task_id or "",
        }

        result = await self.graph.ainvoke(
            {
                "messages": messages,
                # Merge custom fields into your AgentState subclass
                "state": {"my_custom_field": "some_value"},
            },
            config=run_config,
        )

        response_text = self._extract_response_text(result)

        # Custom logic: detect if agent needs user input
        if self._needs_user_input(result):
            msg = updater.new_agent_message(parts=[TextPart(text=response_text)])
            await updater.update_status(TaskState.input_required, message=msg)
        else:
            await updater.add_artifact([TextPart(text=response_text)])
            await updater.complete()
```

**Common customisation patterns:**

| Pattern | How |
|---------|-----|
| **`input_required` relay** | Inspect `result["state"]` for tool/status markers, set `TaskState.input_required` |
| **Custom state fields** | Subclass `AgentState` and pass `"state": {field: value}` in the `ainvoke` input dict to merge extra fields (e.g. `PlannerState.a2a_context_id`) into the graph state before the first node runs |
| **Multi-agent delegation** | Use `delegate_to_a2a_agent()` or a custom tool inside a `ToolNode`; derive a namespaced `context_id` per sub-agent (e.g. `f"{ctx_id}:currency"`) to keep each agent's checkpointer thread isolated |
| **Task metadata** | Access `context.task_id`, `context.context_id`, `context.message` for routing decisions |

---

## Examples

Working examples are in `examples/a2a_sdk/`:

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| **`currency/`** | Single ReAct agent with currency conversion tool | `AgentFlowExecutor`, `make_agent_card`, `create_a2a_server` |
| **`pattern1_human_agent/`** | Human-in-the-loop with `input_required` | Custom executor, `TaskState.input_required`, conversation memory |
| **`pattern2_orchestrator/`** | Central orchestrator routing to specialist agents | Multi-agent coordination, `delegate_to_a2a_agent` |
| **`pattern3_smart_client/`** | Client-side routing to multiple agents | Client-driven orchestration, per-agent `context_id` |
| **`pattern4_planner_delegates/`** | Planner ReAct graph delegates currency queries to CurrencyAgent via a `ToolNode` tool; detects `input_required` and relays it across agent boundaries | Custom `PlannerState` carrying `a2a_context_id`, namespaced sub-agent `context_id` (`f"{ctx}:currency"`), `_tool_requested_input` inspection of state, `TaskState.input_required` relay |

---

## API Summary

### Public Exports (`from agentflow.a2a_integration import ...`)

| Name | Type | Description |
|------|------|-------------|
| `AgentFlowExecutor` | Class | Bridges `CompiledGraph` into A2A's `AgentExecutor` interface |
| `make_agent_card` | Function | Builds an `AgentCard` with sensible defaults |
| `build_a2a_app` | Function | Returns a Starlette ASGI app (composable) |
| `create_a2a_server` | Function | One-call blocking server (uvicorn) |
| `delegate_to_a2a_agent` | Function | Async one-shot: send text to remote agent, get text back |
| `create_a2a_client_node` | Function | Factory returning a graph node that wraps a remote A2A agent |

### Dependencies

| Package | Purpose |
|---------|---------|
| `a2a-sdk` | A2A protocol implementation (transport, JSON-RPC, SSE, task lifecycle) |
| `httpx` | Async HTTP client for `delegate_to_a2a_agent` |
| `uvicorn` | ASGI server for `create_a2a_server` |
| `starlette` | ASGI framework (via `a2a-sdk`) |
