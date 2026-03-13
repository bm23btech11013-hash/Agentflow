"""
Pattern 1 — CurrencyAgent (port 10000).

A specialist currency agent that the human talks to DIRECTLY.

Features:
- ReAct tool-calling loop with Frankfurter API (free, no key needed)
- INPUT_REQUIRED when the LLM detects missing info (currency, amount, date)
- Multi-turn support via agentflow checkpointing (thread_id = task_id)

The agent's LLM decides when to ask the user for more information, and
when it has enough to call the tool and return a result.

Usage::

    python -m examples.a2a_sdk.pattern1_human_agent.agent
"""

from __future__ import annotations

import logging

import httpx
import uvicorn
from dotenv import load_dotenv
from litellm import acompletion

from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentSkill, TaskState, TextPart

from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.a2a_integration.server import make_agent_card
from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph, ToolNode
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState, Message as AFMessage
from agentflow.utils.constants import END, ResponseGranularity
from agentflow.utils.converter import convert_messages

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Tool — Frankfurter API                                              #
# ------------------------------------------------------------------ #


async def get_exchange_rate(
    currency_from: str,
    currency_to: str,
    amount: float = 1.0,
    currency_date: str = "latest",
) -> dict:
    """Get exchange rate between two currencies using the Frankfurter API.

    Args:
        currency_from: Source currency code (e.g. USD).
        currency_to:   Target currency code (e.g. INR).
        amount:        Amount to convert.
        currency_date: Date in YYYY-MM-DD format or 'latest'.

    Returns:
        dict with keys: amount, base, date, rates.
    """
    url = f"https://api.frankfurter.app/{currency_date}"
    params = {"from": currency_from, "to": currency_to, "amount": amount}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


tool_node = ToolNode([get_exchange_rate])

# ------------------------------------------------------------------ #
#  LLM node                                                            #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a currency conversion assistant that talks directly to a human.
You have access to the get_exchange_rate tool.

IMPORTANT: Before calling the tool you MUST have:
  - currency_from (e.g. USD)
  - currency_to (e.g. INR)
  - amount (e.g. 100)

If ANY of these are missing, do NOT call the tool.
Instead, ask the user for the missing information clearly.
Examples:
  "What amount would you like to convert?"
  "Which currency would you like to convert to?"
  "Could you tell me the source currency?"

You may also ask about the date if the user mentions a specific date.
Use 'latest' for the most recent rates if the user says 'today' or
doesn't specify.

Once you have all required info, call get_exchange_rate and return
a clear, formatted result with the conversion amount and rate date.\
"""


async def llm_node(state: AgentState):
    """Call the LLM, optionally providing tools."""
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )

    # If the last message is a tool result, call without tools so the
    # model summarises the result.
    if state.context and state.context[-1].role == "tool":
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools,
        )

    return ModelResponseConverter(response, converter="litellm")


# ------------------------------------------------------------------ #
#  Routing — ReAct conditional edge                                    #
# ------------------------------------------------------------------ #


def should_use_tools(state: AgentState) -> str:
    """Decide whether to route to the tool node or end."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return "MAIN"

    return END


# ------------------------------------------------------------------ #
#  Build the graph                                                     #
# ------------------------------------------------------------------ #


def build_currency_graph() -> CompiledGraph:
    """Construct and compile the currency agent graph."""
    graph = StateGraph()
    graph.add_node("MAIN", llm_node)
    graph.add_node("TOOL", tool_node)

    graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", "MAIN": "MAIN", END: END},
    )
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph.compile()


# ------------------------------------------------------------------ #
#  INPUT_REQUIRED heuristic                                            #
# ------------------------------------------------------------------ #


def _is_asking_for_input(text: str) -> bool:
    """Heuristic: returns True if the LLM response is asking for more info.

    Checks for trailing question marks and common asking-phrases.
    An agent (not an API) can ask follow-up questions — this is the key
    difference in Pattern 1.
    """
    text_lower = text.lower().strip()
    if text_lower.endswith("?"):
        return True
    asking_phrases = [
        "could you",
        "please provide",
        "please specify",
        "what amount",
        "which currency",
        "what currency",
        "let me know",
        "can you tell",
        "i need",
        "please tell",
        "what is the",
        "what date",
    ]
    return any(phrase in text_lower for phrase in asking_phrases)


# ------------------------------------------------------------------ #
#  Custom executor with INPUT_REQUIRED support                         #
# ------------------------------------------------------------------ #


class DirectCurrencyExecutor(AgentFlowExecutor):
    """Extends ``AgentFlowExecutor`` to emit INPUT_REQUIRED when the
    LLM response is a question (missing currency info).

    This is the core of Pattern 1: the human talks directly to the
    specialist agent, and the agent can ask follow-up questions.
    An API cannot do this — it either has the data or it doesn't.
    """

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
            messages = [AFMessage.text_message(user_text, role="user")]

            # Use task_id as thread_id for multi-turn checkpointing
            run_config = {"thread_id": context.task_id or ""}

            result = await self.graph.ainvoke(
                {"messages": messages},
                config=run_config,
                response_granularity=ResponseGranularity.FULL,
            )

            response_text = self._extract_response_text(result)
            is_question = _is_asking_for_input(response_text)

            if is_question:
                # Signal that the agent needs more info from the human
                # This is what makes agents different from APIs!
                question_msg = updater.new_agent_message(
                    parts=[TextPart(text=response_text)],
                )
                await updater.update_status(
                    TaskState.input_required,
                    message=question_msg,
                )
            else:
                # Agent has a complete answer
                await updater.add_artifact([TextPart(text=response_text)])
                await updater.complete()

        except Exception as exc:
            logger.exception("DirectCurrencyExecutor failed")
            error_msg = updater.new_agent_message(
                parts=[TextPart(text=f"Error: {exc!s}")],
            )
            await updater.failed(message=error_msg)


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #


def run_server() -> None:
    """Start the CurrencyAgent on port 10000."""
    compiled = build_currency_graph()

    card = make_agent_card(
        name="CurrencyAgent",
        description=(
            "A specialist currency conversion agent that talks directly "
            "to the human. Asks for missing info (INPUT_REQUIRED) before "
            "converting. Uses live rates from the Frankfurter API."
        ),
        url="http://localhost:10000/",
        skills=[
            AgentSkill(
                id="currency_conversion",
                name="Currency Conversion",
                description=(
                    "Convert between currencies. Asks for missing info "
                    "via INPUT_REQUIRED — this is what makes it an agent, "
                    "not just an API."
                ),
                tags=["currency", "finance", "exchange-rate"],
                examples=[
                    "How much is 100 USD in EUR?",
                    "Convert 50 GBP to JPY",
                    "I want to convert some currency",  # triggers INPUT_REQUIRED
                ],
            )
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=DirectCurrencyExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("=" * 60)
    print("  Pattern 1 — Human ↔ Agent (Direct)")
    print("=" * 60)
    print()
    print("  CurrencyAgent running on http://localhost:10000")
    print("  Agent card: http://localhost:10000/.well-known/agent-card.json")
    print()
    print("  Try these in the client:")
    print('    "convert currency"         → Agent asks what currency/amount')
    print('    "100 USD to INR"           → Agent converts directly')
    print('    "convert 50 EUR"           → Agent asks target currency')
    print()
    print("  Press Ctrl+C to stop.")
    print()

    uvicorn.run(app.build(), host="0.0.0.0", port=10000)


if __name__ == "__main__":
    run_server()
