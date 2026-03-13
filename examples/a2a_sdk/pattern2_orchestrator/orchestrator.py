"""
Pattern 2 — OrchestratorAgent (port 10002).

The single entry point for the human. Reasons about intent using its
own LLM, then delegates to the appropriate specialist agent.

Flow::

    Human
        │
        ▼
    OrchestratorAgent (has LLM, reasons about intent)
        │
        ├──► CurrencyAgent  ("100 USD to INR")
        │         │ COMPLETED
        │         ▼
        │    "8,362 INR"
        │
        ├──► WeatherAgent  ("weather in London")
        │         │ COMPLETED
        │         ▼
        │    "Rainy, 15°C"
        │
        ▼
    Human ← orchestrated final answer

Key design:
    - Orchestrator clarifies with human BEFORE delegating (INPUT_REQUIRED)
    - Specialists ALWAYS get complete info → run to COMPLETION
    - Specialists are autonomous — they have their own LLM, can handle
      complex reasoning, return rich artifacts. An API just returns raw data.

Usage::

    # Start specialist agents first:
    python -m examples.a2a_sdk.pattern2_orchestrator.currency_agent
    python -m examples.a2a_sdk.pattern2_orchestrator.weather_agent

    # Then start the orchestrator:
    python -m examples.a2a_sdk.pattern2_orchestrator.orchestrator
"""

from __future__ import annotations

import logging

import uvicorn
from dotenv import load_dotenv
from litellm import acompletion

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState, TextPart

from agentflow.a2a_integration.client import delegate_to_a2a_agent
from agentflow.a2a_integration.server import make_agent_card

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Specialist URLs                                                     #
# ------------------------------------------------------------------ #

CURRENCY_AGENT_URL = "http://localhost:10000"
WEATHER_AGENT_URL = "http://localhost:10001"

# ------------------------------------------------------------------ #
#  Prompts                                                             #
# ------------------------------------------------------------------ #

ROUTER_PROMPT = """\
You are a message classifier. Analyze the user's message and respond
with EXACTLY one word:
- "currency" if about currency conversion, exchange rates, or money
- "weather" if about weather, temperature, forecast, or climate
- "general" if about anything else

Respond with only that single word, nothing else.\
"""

GENERAL_PROMPT = """\
You are a helpful general assistant.
Answer the user's question helpfully and concisely.
If they ask about currency or weather, let them know you can help.\
"""

# The orchestrator gathers missing info BEFORE delegating to specialists.
# This keeps INPUT_REQUIRED at the orchestrator↔human boundary only.
CURRENCY_GATHER_PROMPT = """\
You are gathering currency conversion info from the user.

Needed: source currency, target currency, amount.

CONVERSATION SO FAR:
{history}

USER'S LATEST MESSAGE:
{user_message}

RULES:
- If you have ALL THREE pieces, respond with EXACTLY:
  READY: Convert <amount> <from> to <to>
  Example: READY: Convert 100 USD to INR
- If missing any, ask concisely. Example: "What amount?"
- Never call tools. Never make up values.\
"""

WEATHER_GATHER_PROMPT = """\
You are gathering weather query info from the user.

Needed: city name.

CONVERSATION SO FAR:
{history}

USER'S LATEST MESSAGE:
{user_message}

RULES:
- If you have the city name, respond with EXACTLY:
  READY: Weather in <city>
  Example: READY: Weather in London
- If missing, ask: "Which city would you like the weather for?"
- Never call tools. Never make up values.\
"""


# ------------------------------------------------------------------ #
#  Orchestrator Executor                                               #
# ------------------------------------------------------------------ #


class OrchestratorExecutor(AgentExecutor):
    """Orchestrator that owns the human conversation.

    - Classifies intent (currency / weather / general)
    - For currency & weather: gathers info via INPUT_REQUIRED, then
      delegates ONE complete request to the specialist
    - For general: answers directly with its own LLM
    - Specialists are NEVER exposed to INPUT_REQUIRED
    """

    def __init__(self) -> None:
        # context_id → {"intent": str, "history": [(role, text), ...]}
        # Keyed by context_id (not task_id) so history persists across
        # completed tasks within the same session.
        self._conversations: dict[str, dict] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.submit()
        await updater.start_work()

        try:
            user_text = context.get_user_input() if context.message else ""
            ctx_id = context.context_id or context.task_id or ""

            # Follow-up to an existing gathering conversation?
            if ctx_id in self._conversations and self._conversations[ctx_id].get("gathering"):
                conv = self._conversations[ctx_id]
                await self._continue_gathering(
                    user_text, updater, ctx_id, conv["intent"],
                )
                return

            # Ensure we have a history list for this context
            if ctx_id not in self._conversations:
                self._conversations[ctx_id] = {"intent": None, "history": [], "gathering": False}

            # New message — classify intent (using recent history for context)
            intent = await self._classify(user_text, self._conversations[ctx_id]["history"])

            if intent == "currency":
                self._conversations[ctx_id]["intent"] = "currency"
                self._conversations[ctx_id]["gathering"] = True
                await self._continue_gathering(user_text, updater, ctx_id, "currency")
            elif intent == "weather":
                self._conversations[ctx_id]["intent"] = "weather"
                self._conversations[ctx_id]["gathering"] = True
                await self._continue_gathering(user_text, updater, ctx_id, "weather")
            else:
                await self._handle_general(user_text, updater, ctx_id)

        except Exception as exc:
            logger.exception("OrchestratorExecutor failed")
            error_msg = updater.new_agent_message(
                parts=[TextPart(text=f"Error: {exc!s}")],
            )
            await updater.failed(message=error_msg)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()

    # ---- Classification ---- #

    async def _classify(self, text: str, history: list[tuple] | None = None) -> str:
        """Classify user intent as currency/weather/general.

        Uses recent conversation history for context so that follow-up
        messages like "convert same to EUR" are correctly classified.
        """
        msgs: list[dict] = [{"role": "system", "content": ROUTER_PROMPT}]
        # Include recent history for context
        for role, content in (history or [])[-4:]:
            msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": text})

        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=msgs,
        )
        result = (response.choices[0].message.content or "").strip().lower()
        if "currency" in result:
            return "currency"
        if "weather" in result:
            return "weather"
        return "general"

    # ---- Info Gathering ---- #

    async def _continue_gathering(
        self,
        user_text: str,
        updater: TaskUpdater,
        ctx_id: str,
        intent: str,
    ) -> None:
        """Gather required info, then delegate to specialist."""
        conv = self._conversations[ctx_id]
        history = conv["history"]
        history.append(("user", user_text))

        history_str = "\n".join(
            f"  {role}: {text}" for role, text in history
        ) if len(history) > 1 else "(No prior conversation)"

        prompt = (
            CURRENCY_GATHER_PROMPT if intent == "currency"
            else WEATHER_GATHER_PROMPT
        )

        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        history=history_str,
                        user_message=user_text,
                    ),
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()

        if answer.upper().startswith("READY:"):
            # We have all info — delegate to specialist
            complete_query = answer[len("READY:"):].strip()
            await self._delegate(complete_query, updater, ctx_id, intent)
        else:
            # Still missing info — ask the human (orchestrator↔human only)
            history.append(("assistant", answer))
            msg = updater.new_agent_message(
                parts=[TextPart(text=answer)],
            )
            await updater.update_status(
                TaskState.input_required, message=msg,
            )

    # ---- Delegation to Specialists ---- #

    async def _delegate(
        self,
        complete_query: str,
        updater: TaskUpdater,
        ctx_id: str,
        intent: str,
    ) -> None:
        """Send a complete query to the appropriate specialist agent.

        Specialists are autonomous agents with their own LLM — not APIs.
        They run to COMPLETION and return rich, reasoned results.
        """
        url = (
            CURRENCY_AGENT_URL if intent == "currency"
            else WEATHER_AGENT_URL
        )
        agent_name = "CurrencyAgent" if intent == "currency" else "WeatherAgent"

        try:
            result_text = await delegate_to_a2a_agent(
                url, complete_query, timeout=60.0,
            )
        except Exception as exc:
            logger.exception("Delegation to %s failed", agent_name)
            result_text = f"Sorry, {agent_name} returned an error: {exc!s}"

        # Mark gathering as done (keep history for future turns)
        if ctx_id in self._conversations:
            self._conversations[ctx_id]["gathering"] = False
            self._conversations[ctx_id]["history"].append(
                ("assistant", result_text),
            )

        await updater.add_artifact([TextPart(text=result_text)])
        await updater.complete()

    # ---- General ---- #

    async def _handle_general(
        self,
        text: str,
        updater: TaskUpdater,
        ctx_id: str,
    ) -> None:
        """Answer a non-specialist question with full conversation history."""
        conv = self._conversations.get(ctx_id)
        history = conv["history"] if conv else []

        msgs: list[dict] = [{"role": "system", "content": GENERAL_PROMPT}]
        for role, content in history[-10:]:
            msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": text})

        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=msgs,
        )
        answer = (response.choices[0].message.content or "").strip() or "No response."

        # Record in history
        if conv is not None:
            history.append(("user", text))
            history.append(("assistant", answer))

        await updater.add_artifact([TextPart(text=answer)])
        await updater.complete()


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #


def run_server() -> None:
    """Start OrchestratorAgent on port 10002."""
    card = make_agent_card(
        name="OrchestratorAgent",
        description=(
            "General assistant that routes queries to specialist agents "
            "(CurrencyAgent, WeatherAgent) via A2A protocol. Gathers "
            "all required info before delegating."
        ),
        url="http://localhost:10002/",
    )

    handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("=" * 60)
    print("  Pattern 2 — Orchestrator Delegates to Specialists")
    print("=" * 60)
    print()
    print("  OrchestratorAgent running on http://localhost:10002")
    print()
    print("  Make sure specialists are running first:")
    print("    CurrencyAgent on http://localhost:10000")
    print("    WeatherAgent  on http://localhost:10001")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    uvicorn.run(app.build(), host="0.0.0.0", port=10002)


if __name__ == "__main__":
    run_server()
