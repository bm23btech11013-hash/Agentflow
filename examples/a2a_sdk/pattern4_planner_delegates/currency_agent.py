"""
Pattern 4 — CurrencyAgent (port 10000) with INPUT_REQUIRED support.

Reuses the graph from ``examples.a2a_sdk.currency.agent`` and adds an
executor that emits INPUT_REQUIRED when the LLM asks for missing info.

Usage::

    python -m examples.a2a_sdk.pattern4_planner_delegates.currency_agent
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentSkill, TaskState, TextPart

from agentflow.a2a_integration import create_a2a_server, make_agent_card
from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.state import Message as AFMessage
from agentflow.utils.constants import ResponseGranularity

# Reuse the graph from the existing currency example
from examples.a2a_sdk.currency.agent import build_currency_graph

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  INPUT_REQUIRED heuristic                                            #
# ------------------------------------------------------------------ #

_ASKING_PHRASES = [
    "could you", "please provide", "please specify",
    "what amount", "which currency", "what currency",
    "let me know", "can you tell", "i need",
    "please tell", "what is the", "what date",
]


def _is_asking_for_input(text: str) -> bool:
    """Return True if the LLM is asking for more info."""
    low = text.lower().strip()
    return low.endswith("?") or any(p in low for p in _ASKING_PHRASES)


# ------------------------------------------------------------------ #
#  Executor — extends AgentFlowExecutor with INPUT_REQUIRED            #
# ------------------------------------------------------------------ #


class CurrencyAgentExecutor(AgentFlowExecutor):
    """Runs the currency graph; emits INPUT_REQUIRED for vague queries."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.submit()
        await updater.start_work()

        try:
            user_text = context.get_user_input() if context.message else ""
            messages = [AFMessage.text_message(user_text, role="user")]
            # Use context_id (not task_id) so conversation history
            # persists across multiple tasks in the same session.
            run_config = {"thread_id": context.context_id or context.task_id or ""}

            result = await self.graph.ainvoke(
                {"messages": messages},
                config=run_config,
                response_granularity=ResponseGranularity.FULL,
            )
            response_text = self._extract_response_text(result)

            if _is_asking_for_input(response_text):
                msg = updater.new_agent_message(parts=[TextPart(text=response_text)])
                await updater.update_status(TaskState.input_required, message=msg)
            else:
                await updater.add_artifact([TextPart(text=response_text)])
                await updater.complete()

        except Exception as exc:
            logger.exception("CurrencyAgentExecutor failed")
            err = updater.new_agent_message(parts=[TextPart(text=f"Error: {exc!s}")])
            await updater.failed(message=err)


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    compiled = build_currency_graph()

    card = make_agent_card(
        name="CurrencyAgent",
        description="Currency conversion with INPUT_REQUIRED for missing info.",
        url="http://localhost:10000/",
        streaming=True,
        skills=[
            AgentSkill(
                id="currency_conversion",
                name="Currency Conversion",
                description="Convert between currencies. Asks if info is missing.",
                tags=["currency", "finance", "exchange-rate"],
                examples=["How much is 100 USD in EUR?", "Convert 50 GBP to JPY"],
            )
        ],
    )

    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    import uvicorn

    handler = DefaultRequestHandler(
        agent_executor=CurrencyAgentExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("[CurrencyAgent] http://localhost:10000  (Ctrl+C to stop)")
    uvicorn.run(app.build(), host="0.0.0.0", port=10000)
