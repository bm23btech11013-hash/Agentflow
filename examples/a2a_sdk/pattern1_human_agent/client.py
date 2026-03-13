"""
Pattern 1 — Human Client for CurrencyAgent (Direct).

Interactive CLI that talks DIRECTLY to the CurrencyAgent.
Handles INPUT_REQUIRED by prompting the human for follow-up info.

This demonstrates the key advantage of agents over APIs:
  - An API returns data or an error.  That's it.
  - An agent can ask follow-up questions and have a conversation.

Flow::


    Human Client
        │  "convert 100 USD to INR"
        ▼
    CurrencyAgent
        │  INPUT_REQUIRED "which date?"
        ▼
    Human Client
        │  "today"
        ▼
    CurrencyAgent
        │  COMPLETED "100 USD = 8,362 INR"
        ▼
    Human Client

Usage::

    # Make sure the agent is running first:
    #   python -m examples.a2a_sdk.pattern1_human_agent.agent

    python -m examples.a2a_sdk.pattern1_human_agent.client
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
from a2a.client import A2AClient
from a2a.types import (
    Message as A2AMessage,
    MessageSendParams,
    Role,
    SendMessageRequest,
    TextPart,
)

AGENT_URL = "http://localhost:10000"


# ------------------------------------------------------------------ #
#  Text extraction helpers                                             #
# ------------------------------------------------------------------ #


def _get_status_message_text(payload) -> str:
    """Extract text from the status message of an INPUT_REQUIRED task."""
    try:
        for p in payload.status.message.parts:
            if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
                return p.root.text
            if hasattr(p, "text") and isinstance(p.text, str) and p.text:
                return p.text
    except Exception:
        pass
    return "Please provide more info."


def _extract_response_text(payload) -> str:
    """Extract text from a completed task's artifacts or status message."""
    try:
        if hasattr(payload, "artifacts") and payload.artifacts:
            texts: list[str] = []
            for artifact in payload.artifacts:
                for p in artifact.parts:
                    if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
                        texts.append(p.root.text)
                    elif hasattr(p, "text") and isinstance(p.text, str) and p.text:
                        texts.append(p.text)
            if texts:
                return "\n".join(texts)

        if hasattr(payload, "status") and payload.status and payload.status.message:
            for p in payload.status.message.parts:
                if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
                    return p.root.text
                if hasattr(p, "text") and isinstance(p.text, str) and p.text:
                    return p.text
    except Exception:
        pass
    return "No response."


# ------------------------------------------------------------------ #
#  Interactive chat loop                                               #
# ------------------------------------------------------------------ #


async def chat() -> None:
    """Interactive CLI that talks directly to the CurrencyAgent.

    Handles INPUT_REQUIRED by keeping the same ``task_id`` and prompting
    the human for follow-up information.  This is what makes agents
    different from APIs — they can have a conversation!
    """
    print("=" * 60)
    print("  Pattern 1 — Human ↔ CurrencyAgent (Direct)")
    print("=" * 60)
    print()
    print("  Talk directly to the CurrencyAgent.")
    print("  The agent will ask for missing info (INPUT_REQUIRED).")
    print()
    print("  Try:")
    print('    "convert currency"   → Agent asks what/how much')
    print('    "100 USD to INR"     → Agent converts directly')
    print('    "convert 50 EUR"     → Agent asks target currency')
    print()
    print("  Type 'quit' to exit, 'new' to reset conversation.")
    print()

    # Persistent context_id so the agent retains conversation history
    # across completed tasks within the same session.
    context_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=60.0) as http:
        client = A2AClient(httpx_client=http, url=AGENT_URL)
        task_id: str | None = None  # reuse for multi-turn INPUT_REQUIRED

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() == "quit":
                print("Bye!")
                break
            if user_input.lower() == "new":
                task_id = None
                context_id = str(uuid.uuid4())
                print("\n--- New conversation (context reset) ---\n")
                continue

            msg_kwargs: dict = {"context_id": context_id}
            if task_id:
                msg_kwargs["task_id"] = task_id

            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=A2AMessage(
                        role=Role.user,
                        message_id=str(uuid.uuid4()),
                        parts=[TextPart(text=user_input)],
                        **msg_kwargs,
                    ),
                ),
            )

            try:
                response = await client.send_message(request)
            except Exception as exc:
                print(f"  Error: {exc}\n")
                task_id = None
                continue

            result = response.root

            if hasattr(result, "error") and result.error:
                print(f"  Error: {result.error}\n")
                task_id = None
                continue

            payload = result.result

            # Check if the agent needs more input (INPUT_REQUIRED)
            if hasattr(payload, "status") and payload.status:
                state_str = str(payload.status.state).lower()
                if "input_required" in state_str:
                    # Keep task_id so next message continues this conversation
                    task_id = payload.id if hasattr(payload, "id") else None
                    msg_text = _get_status_message_text(payload)
                    print(f"Agent: {msg_text}")
                    print("       (waiting for your input)\n")
                    continue
                else:
                    task_id = None

            # Final answer
            text = _extract_response_text(payload)
            print(f"Agent: {text}\n")
            task_id = None  # conversation complete, start fresh


if __name__ == "__main__":
    asyncio.run(chat())
