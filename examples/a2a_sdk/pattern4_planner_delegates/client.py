"""
Pattern 4 — Interactive **streaming** client for PlannerAgent.

Shows how INPUT_REQUIRED propagates across two agent boundaries,
with real-time task-state updates printed in the terminal::

    submitted → working → input_required / completed

Usage::

    # 1. Start CurrencyAgent:
    python -m examples.a2a_sdk.pattern4_planner_delegates.currency_agent

    # 2. Start PlannerAgent:
    python -m examples.a2a_sdk.pattern4_planner_delegates.planner_agent

    # 3. Run this client:
    python -m examples.a2a_sdk.pattern4_planner_delegates.client
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    Message as A2AMessage,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

PLANNER_AGENT_URL = "http://localhost:10001"

# A persistent context_id ties all tasks in a session together,
# so the agents retain conversation history across completed tasks.
SESSION_CONTEXT_ID = str(uuid.uuid4())

# ANSI colour helpers (work in Windows Terminal / PowerShell 7+)
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _text_from_parts(parts) -> str:
    """Best-effort text extraction from A2A Part list."""
    for p in parts or []:
        root = getattr(p, "root", p)
        if hasattr(root, "text") and isinstance(root.text, str) and root.text:
            return root.text
    return ""


async def send_message_streaming(
    client,
    text: str,
    *,
    task_id: str | None = None,
) -> dict:
    """Send a message via SSE streaming, printing every state update.

    Returns dict with 'status', 'text', 'task_id'.
    """
    msg_kwargs: dict = {"context_id": SESSION_CONTEXT_ID}
    if task_id:
        msg_kwargs["task_id"] = task_id

    message = A2AMessage(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=text))],
        **msg_kwargs,
    )

    final_status = "unknown"
    final_text = ""
    final_task_id: str | None = None

    async for event in client.send_message(message):
        # New ClientFactory yields  tuple[Task, UpdateEvent | None]  or  Message
        if isinstance(event, A2AMessage):
            # Direct message response (rare — most agents use tasks)
            final_text = _text_from_parts(event.parts)
            final_status = "completed"
            print(f"  {DIM}← message response{RESET}")
            continue

        if not isinstance(event, tuple):
            continue

        task, update = event

        final_task_id = task.id

        if update is None:
            # Initial task object (no update yet)
            state = str(task.status.state.value) if task.status else "?"
            print(f"  {DIM}[task {task.id[:8]}…] {state}{RESET}")
            continue

        if isinstance(update, TaskStatusUpdateEvent):
            state = str(update.status.state.value)
            # Colour-code the state
            if state == "submitted":
                colour = DIM
            elif state == "working":
                colour = CYAN
            elif state == "input-required":
                colour = YELLOW
            elif state == "completed":
                colour = GREEN
            elif state == "failed":
                colour = RED
            else:
                colour = DIM
            print(f"  {colour}⬥ {state}{RESET}", end="")

            # Print status message text (if any)
            if update.status.message:
                txt = _text_from_parts(update.status.message.parts)
                if txt:
                    final_text = txt
                    print(f"  {DIM}— {txt[:80]}{'…' if len(txt) > 80 else ''}{RESET}")
                else:
                    print()
            else:
                print()

            final_status = state

        elif isinstance(update, TaskArtifactUpdateEvent):
            txt = _text_from_parts(update.artifact.parts)
            if txt:
                final_text = txt
                print(f"  {DIM}📎 artifact: {txt[:80]}{'…' if len(txt) > 80 else ''}{RESET}")

    return {"status": final_status, "text": final_text, "task_id": final_task_id}


async def main() -> None:
    """Interactive CLI client for the PlannerAgent — with streaming."""
    print("=" * 60)
    print("  Pattern 4 — Streaming Client for PlannerAgent")
    print("=" * 60)
    print()
    print(f"  PlannerAgent:  {PLANNER_AGENT_URL}")
    print("  CurrencyAgent: http://localhost:10000")
    print()
    print("  Every task-state update is printed in real time:")
    print(f"    {DIM}⬥ submitted{RESET}  {CYAN}⬥ working{RESET}  "
          f"{YELLOW}⬥ input-required{RESET}  {GREEN}⬥ completed{RESET}")
    print()
    print("  Try:")
    print('    > "convert currency"  → CurrencyAgent asks (INPUT_REQUIRED)')
    print('    > "100 USD to EUR"    → CurrencyAgent converts (completed)')
    print()
    print("  Type 'quit' to exit, 'new' to start a fresh conversation.")
    print()

    async with httpx.AsyncClient(timeout=60.0) as http:
        config = ClientConfig(httpx_client=http, streaming=True)
        client = await ClientFactory.connect(
            PLANNER_AGENT_URL, client_config=config,
        )

        current_task_id: str | None = None

        while True:
            prompt = "You (follow-up)" if current_task_id else "You"
            try:
                user_input = input(f"\n{prompt}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "new":
                current_task_id = None
                global SESSION_CONTEXT_ID
                SESSION_CONTEXT_ID = str(uuid.uuid4())
                print(f"\n--- New conversation (new context) ---")
                continue

            result = await send_message_streaming(
                client, user_input, task_id=current_task_id,
            )

            state = result["status"]

            if "input-required" in state or "input_required" in state:
                current_task_id = result["task_id"]
                print(f"\n{BOLD}Agent asks:{RESET} {result['text']}")
                print(f"  {DIM}(The agent needs more info — type your answer){RESET}")

            elif "completed" in state:
                current_task_id = None
                print(f"\n{BOLD}Agent:{RESET} {result['text']}")

            elif "failed" in state:
                current_task_id = None
                print(f"\n{RED}[FAILED]{RESET} {result['text']}")

            else:
                current_task_id = None
                print(f"\n[{state}] {result['text']}")


if __name__ == "__main__":
    asyncio.run(main())
