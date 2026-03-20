"""Interactive Skills Chat — terminal REPL demo.

Start the chat:
    python examples/skills/chat.py

The agent will automatically activate the right skill based on what you type:
  - "review my code / find bugs"          → code-review skill
  - "analyse this data / statistics"      → data-analysis skill
  - "help me write / proofread this"      → writing-assistant skill
  - "humanize this / sounds too AI"       → humanizer skill
  - "quit" / "exit" / Ctrl-C             → exit

The prompt shows the currently active skill in brackets: [code-review] You:
"""

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from agentflow.graph import Agent, StateGraph
from agentflow.skills import SkillConfig
from agentflow.state import AgentState, Message
from agentflow.state.message_context_manager import MessageContextManager
from agentflow.utils.constants import END

load_dotenv()

SKILLS_DIR = str(Path(__file__).parent / "skills")

# ── Graph setup ────────────────────────────────────────────────────────────

agent = Agent(
    model="google/gemini-2.5-flash",
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a smart, multi-skilled assistant.\n"
                "You have access to specialised skill modes that give you "
                "deeper expertise in specific domains.\n\n"
                "Rules for skill activation:\n"
                "1. When the user's request matches a skill, ALWAYS call set_skill() "
                "with that skill name BEFORE doing anything else — even if a different "
                "skill is already active. auto_deactivate will handle clearing the old one.\n"
                "2. Skills are distinct — each one covers a specific domain. "
                "Use the available skills table to decide which one fits the request.\n"
                "3. When the task is done and the user moves to an unrelated topic, "
                "call clear_skill() to return to general mode."
            ),
        }
    ],
    skills=SkillConfig(
        skills_dir=SKILLS_DIR,
        inject_trigger_table=True,
        hot_reload=True,
        auto_deactivate=True,
    ),
    trim_context=True,
)

tool_node = agent.get_tool_node()


def should_use_tools(state: AgentState) -> str:
    if not state.context:
        return END
    last = state.context[-1]
    if last.role == "assistant" and hasattr(last, "tools_calls") and last.tools_calls:
        return "TOOL"
    if last.role == "tool":
        return "MAIN"
    return END


graph = StateGraph(context_manager=MessageContextManager(max_messages=20))
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()


# ── Helpers ────────────────────────────────────────────────────────────────

def print_banner(thread_id: str) -> None:
    skills = agent._skills_registry.get_all()  # type: ignore[union-attr]
    width = 54
    print("\n" + "=" * width)
    print("  Agentflow Skills — Interactive Chat")
    print("=" * width)
    if skills:
        print("Available skills:")
        for s in skills:
            triggers = "; ".join(f'"{t}"' for t in s.triggers[:2])
            print(f"  • {s.name:<22} → {triggers}")
    print()
    print(f"  Thread : {thread_id}")
    print("  Exit   : type 'quit', 'exit', or press Ctrl-C")
    print("=" * width + "\n")


def extract_active_skill(messages: list, current: str | None) -> tuple[str | None, list[str]]:
    """Scan tool messages for SKILL_ACTIVATED / SKILL_DEACTIVATED markers.

    Returns (new_active_skill, list_of_notifications).
    """
    notifications: list[str] = []
    active = current
    for msg in messages:
        if msg.role != "tool":
            continue
        text = msg.text() or ""
        if text.startswith("SKILL_ACTIVATED:"):
            name = text.split(":", 1)[1].strip()
            notifications.append(f"  >> Skill activated : {name}")
            active = name
        elif text == "SKILL_DEACTIVATED":
            notifications.append(f"  >> Skill cleared   : {active or '(none)'}")
            active = None
    return active, notifications


# ── REPL ───────────────────────────────────────────────────────────────────

def main() -> None:
    thread_id = f"skills-chat-{uuid4().hex[:8]}"
    active_skill: str | None = None

    print_banner(thread_id)

    while True:
        prompt = f"[{active_skill}] You: " if active_skill else "You: "
        try:
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        try:
            result = app.invoke(
                {"messages": [Message.text_message(user_input)]},
                config={"thread_id": thread_id, "recursion_limit": 20},
            )
        except Exception as exc:
            print(f"\n  [error] {exc}\n")
            continue

        # Update active skill and collect notifications
        active_skill, notes = extract_active_skill(result["messages"], active_skill)
        for note in notes:
            print(note)

        # Print the last assistant response
        for msg in reversed(result["messages"]):
            if msg.role == "assistant" and msg.text():
                print(f"\nAssistant: {msg.text()}\n")
                break


if __name__ == "__main__":
    main()
