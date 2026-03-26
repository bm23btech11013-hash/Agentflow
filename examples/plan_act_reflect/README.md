# Plan-Act-Reflect Examples

This directory contains two runnable examples demonstrating the `PlanActReflectAgent` loop pattern:

## Files

- `basic_plan_act_reflect.py`  
  Single tool round:
  1. PLAN emits a tool call
  2. ACT executes tool
  3. REFLECT annotates result
  4. PLAN produces final answer → END

- `tool_plan_act_reflect.py`  
  Multi-tool, iterative variant with:
  - Multiple registered tools
  - Custom routing condition (overrides default heuristic)
  - Confidence tracking + max turn budget

## Pattern Overview

```
PLAN --(condition)--> ACT --> REFLECT
  ^                     |
  |---------------------|
```

Decision function returns one of:
- `ACT` (execute tools)
- `REFLECT` (analyze tool output and loop)
- `END` (terminate)

## Key Concepts

| Concept | Role |
|--------|------|
| PLAN | Generates next reasoning step and optional `tools_calls` list |
| ACT | Executes tool calls via `ToolNode` and appends tool result messages |
| REFLECT | Interprets tool results, updates confidence / strategy |
| Condition | Heuristic or custom function deciding next edge |

Heuristic (default) in agent:
- Assistant message with `tools_calls` ⇒ ACT
- Last message role `tool` ⇒ REFLECT
- Else ⇒ END

## Running

```bash
python examples/plan_act_reflect/basic_plan_act_reflect.py
python examples/plan_act_reflect/tool_plan_act_reflect.py
```

No API keys required (purely deterministic mock tools). You can extend PLAN to call real LLMs through your preferred SDK and emit structured tool requests.

## Extending

1. Add richer planning prompt + model invocation in PLAN.
2. Introduce evaluation node before END (quality / safety guard).
3. Persist intermediate states using a checkpointer (pass `checkpointer` to `compile`).
4. Combine with retrieval (prepend a retrieval node feeding context before PLAN).

## When to Use

Use Plan-Act-Reflect when you need explicit separation between:
- Generating intentions
- Performing tool executions
- Interpreting / validating results

Compared to ReAct, this pattern isolates reflection, making it easier to:
- Insert guardrails
- Track iteration metrics
- Swap planners or reflectors independently

## Minimal Skeleton

```python
from agentflow.prebuilt.agent.plan_act_reflect import PlanActReflectAgent
from agentflow.graph.tool_node import ToolNode
from agentflow.state.agent_state import AgentState
from agentflow.utils import Message


def plan(state: AgentState) -> AgentState: ...


def reflect(state: AgentState) -> AgentState: ...


tools = ToolNode([your_tool])

agent = PlanActReflectAgent[AgentState](state=AgentState())
app = agent.compile(plan_node=plan, tool_node=tools, reflect_node=reflect)
result = app.invoke({"messages": [Message.text_message("Hi", role="user")]})
```

## Related

- React examples: `examples/react/`
- RAG hybrid pipelines: `examples/rag/`
- Prebuilt reference: `docs/reference/prebuilt/agent/plan_act_reflect.md`