"""
Pattern 2 — Human Client for OrchestratorAgent.

Interactive CLI that talks to the OrchestratorAgent.
The orchestrator classifies intent, gathers missing info, and
delegates to specialist agents (CurrencyAgent, WeatherAgent).

Usage::

    # Start all three agents first:
    #   python -m examples.a2a_sdk.pattern2_orchestrator.currency_agent
    #   python -m examples.a2a_sdk.pattern2_orchestrator.weather_agent
    #   python -m examples.a2a_sdk.pattern2_orchestrator.orchestrator

    python -m examples.a2a_sdk.pattern2_orchestrator.client
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

ORCHESTRATOR_URL = "http://localhost:10002"


# ------------------------------------------------------------------ #
#  Text extraction helpers                                             #
# ------------------------------------------------------------------ #


def _get_status_message_text(payload) -> str:
    """Extract text from an INPUT_REQUIRED status message."""
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
    """Extract text from a completed task."""
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
#  Chat loop                                                           #
# ------------------------------------------------------------------ #


async def chat() -> None:
    """Interactive CLI that talks to the OrchestratorAgent."""
    print("=" * 60)
    print("  Pattern 2 — Orchestrator Delegates to Specialists")
    print("=" * 60)
    print()
    print("  One entry point → multiple specialists behind it.")
    print()
    print("  Try:")
    print('    "100 USD to INR"          → routes to CurrencyAgent')
    print('    "weather in London"        → routes to WeatherAgent')
    print('    "convert currency"         → orchestrator gathers info first')
    print('    "what is the capital of France?" → answered directly')
    print()
    print("  Type 'quit' to exit, 'new' to reset conversation.")
    print()

    # Persistent context_id so the orchestrator retains conversation
    # history across completed tasks within the same session.
    context_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=60.0) as http:
        client = A2AClient(httpx_client=http, url=ORCHESTRATOR_URL)
        task_id: str | None = None

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

            # INPUT_REQUIRED — orchestrator is gathering info
            if hasattr(payload, "status") and payload.status:
                state_str = str(payload.status.state).lower()
                if "input_required" in state_str:
                    task_id = payload.id if hasattr(payload, "id") else None
                    msg_text = _get_status_message_text(payload)
                    print(f"Agent: {msg_text}")
                    print("       (orchestrator gathering info)\n")
                    continue
                else:
                    task_id = None

            text = _extract_response_text(payload)
            print(f"Agent: {text}\n")
            task_id = None


if __name__ == "__main__":
    asyncio.run(chat())
