"""Skills Example — simple graph with Gemini 2.5 Flash + dynamic skills.

This example shows how to use the Agentflow Skills system so a single
agent can switch between specialised modes (code-review, data-analysis,
writing-assistant) at runtime without restarting or changing the graph.

How it works
------------
1. Three ``SKILL.md`` files live in ``./skills/``.
2. ``Agent(skills=SkillConfig(skills_dir=...))`` auto-discovers them,
   builds a trigger table that is appended to the system prompt, and
   registers a ``set_skill`` tool the LLM can call.
3. When the user's message matches a skill's triggers, the LLM calls
   ``set_skill("<name>")``.  The framework intercepts the tool result and
   stores the active skill in ``state.execution_meta.internal_data``.
4. On every subsequent LLM call, the full SKILL.md content is injected as
   an extra system message — completely outside the conversation context,
   so it is NEVER trimmed away even when ``trim_context=True``.
5. ``clear_skill()`` is available to return to general mode.

Skills survive context trimming
--------------------------------
``trim_context=True`` is safe to use with skills because:
- The active skill name lives in ``execution_meta.internal_data``, not in
  ``state.context``.
- Skill content is re-injected fresh on every LLM call from the
  ``SkillInjector`` — it never sits in the trimmed message window.

Run
---
    python graph.py

    # or try a specific query:
    python graph.py "Review this Python code: def add(a,b): return a+b"
    python graph.py "Help me write a professional apology email to a client"
    python graph.py "Analyse this data: sales=[120,95,140,88,160] by month"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState, Message
from agentflow.state.message_context_manager import MessageContextManager
from agentflow.skills import SkillConfig
from agentflow.utils.constants import END

load_dotenv()

# ---------------------------------------------------------------------------
# Skills directory — three SKILL.md files sit alongside this script
# ---------------------------------------------------------------------------
SKILLS_DIR = str(Path(__file__).parent / "skills")
# ---------------------------------------------------------------------------
# Agent — Gemini 2.5 Flash with skills + context trimming enabled
# ---------------------------------------------------------------------------
agent = Agent(
    model="google/gemini-2.5-flash",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a smart, multi-skilled assistant.\n"
                "You have access to specialised skill modes that give you "
                "deeper expertise in specific domains.\n"
                "When the user's request clearly matches a skill, activate "
                "it immediately by calling set_skill() before doing anything else.\n"
                "When the task is done, call clear_skill() to return to general mode."
            ),
        }
    ],
    skills=SkillConfig(
        skills_dir=SKILLS_DIR,
        inject_trigger_table=True,   # auto-appends skill trigger table to system prompt
        hot_reload=True,             # re-reads SKILL.md on every call (great for dev)
        auto_deactivate=True,        # activating a new skill deactivates the old one
    ),
    # trim_context=True is fully safe with skills — the active skill name is
    # stored in execution_meta (not context) and content is re-injected each call.
    trim_context=True,
)

# ---------------------------------------------------------------------------
# Tool node — use the public get_tool_node() method, not agent._tool_node.
# When skills are enabled, this ToolNode already contains set_skill +
# clear_skill — no extra setup required.
# ---------------------------------------------------------------------------
tool_node = agent.get_tool_node()

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_use_tools(state: AgentState) -> str:
    """Route MAIN → TOOL if there are tool calls, else → END."""
    if not state.context:
        return END

    last = state.context[-1]

    if (
        last.role == "assistant"
        and hasattr(last, "tools_calls")
        and last.tools_calls
    ):
        return "TOOL"

    if last.role == "tool":
        return "MAIN"   # got tool results → back to LLM for final answer

    return END

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(
    # Keep at most 20 messages in context — skills still work fine because
    # skill content is injected outside the context window.
    context_manager=MessageContextManager(max_messages=20),
)
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)

graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Accept an optional query from the command line
    user_input = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else (
            "Can you review this Python function for me?\n\n"
            "```python\n"
            "def calculate_average(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        total = total + n\n"
            "    return total / len(numbers)\n"
            "```"
        )
    )

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60 + "\n")

    inp = {"messages": [Message.text_message(user_input)]}
    config = {"thread_id": "skills-demo-1", "recursion_limit": 15}

    res = app.invoke(inp, config=config)

    for msg in res["messages"]:
        if msg.role == "assistant":
            print("-" * 60)
            print(f"ASSISTANT ({msg.role}):")
            print(msg.text() or "(no text)")
            print("-" * 60)
