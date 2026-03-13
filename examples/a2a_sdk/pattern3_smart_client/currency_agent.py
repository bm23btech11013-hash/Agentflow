"""
Pattern 3 — CurrencyAgent specialist (port 10000).

Same currency agent used in Pattern 2, also reusable here.
The Smart Client discovers this agent via its AgentCard and routes
currency queries directly to it — no orchestrator needed.

Usage::

    python -m examples.a2a_sdk.pattern3_smart_client.currency_agent
"""

from __future__ import annotations

import logging

import httpx
import uvicorn
from dotenv import load_dotenv
from litellm import acompletion

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill

from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.a2a_integration.server import make_agent_card
from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph, ToolNode
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState
from agentflow.utils.constants import END
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
You are a currency conversion specialist.
Use the get_exchange_rate tool. Return a clear, formatted result.\
"""


async def llm_node(state: AgentState):
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )
    if state.context and state.context[-1].role == "tool":
        response = await acompletion(
            model="gemini/gemini-2.5-flash", messages=messages,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash", messages=messages, tools=tools,
        )
    return ModelResponseConverter(response, converter="litellm")


def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"
    last = state.context[-1]
    if (
        hasattr(last, "tools_calls") and last.tools_calls
        and len(last.tools_calls) > 0 and last.role == "assistant"
    ):
        return "TOOL"
    if last.role == "tool":
        return "MAIN"
    return END


def build_currency_graph() -> CompiledGraph:
    graph = StateGraph()
    graph.add_node("MAIN", llm_node)
    graph.add_node("TOOL", tool_node)
    graph.add_conditional_edges(
        "MAIN", should_use_tools,
        {"TOOL": "TOOL", "MAIN": "MAIN", END: END},
    )
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile()


def run_server() -> None:
    compiled = build_currency_graph()
    card = make_agent_card(
        name="CurrencyAgent",
        description="Currency conversion agent with live Frankfurter API rates.",
        url="http://localhost:10000/",
        skills=[
            AgentSkill(
                id="currency_conversion",
                name="Currency Conversion",
                description="Convert between currencies using live exchange rates.",
                tags=["currency", "finance", "exchange-rate", "money"],
                examples=["Convert 100 USD to EUR", "50 GBP in JPY"],
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=AgentFlowExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("  [CurrencyAgent] Running on http://localhost:10000")
    uvicorn.run(app.build(), host="0.0.0.0", port=10000)


if __name__ == "__main__":
    run_server()
