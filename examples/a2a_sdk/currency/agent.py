"""
Currency conversion agent — agentflow graph with ReAct tool-calling loop.

Mirrors the official a2a-samples LangGraph currency agent but built
entirely with agentflow.  Uses the Frankfurter API (free, no key needed)
for live exchange rates.

Usage (standalone, without A2A)::

    python -m examples.a2a_sdk.currency.agent
"""

from __future__ import annotations

import httpx
from dotenv import load_dotenv
from litellm import acompletion

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph, ToolNode
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()

# ------------------------------------------------------------------ #
#  Tool — Frankfurter API                                              #
# ------------------------------------------------------------------ #


async def get_exchange_rate(
    currency_from: str,
    currency_to: str,
    currency_date: str = "latest",
    amount: float = 1.0,
) -> dict:
    """Get exchange rate between two currencies using the Frankfurter API.

    Args:
        currency_from: Source currency code (e.g. USD).
        currency_to:   Target currency code (e.g. INR).
        currency_date: Date in YYYY-MM-DD format or 'latest'.
        amount:        Amount to convert.

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
#  LLM node — follows react_weather_agent.py pattern exactly           #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "You are a helpful currency conversion assistant. "
    "Use the get_exchange_rate tool to look up live exchange rates "
    "from the Frankfurter API. Always tell the user the converted "
    "amount and the date of the rate."
)


async def llm_node(state: AgentState):
    """Call the LLM, optionally providing tools."""

    messages = convert_messages(
        system_prompts=[
            {"role": "system", "content": SYSTEM_PROMPT},
        ],
        state=state,
    )

    # If the last message is a tool result, make a final response
    # without tools so the model summarises the result.
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

    # Assistant message with tool calls → execute the tools
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # Tool result → go back to LLM for a final summary
    if last_message.role == "tool":
        return "MAIN"

    # Otherwise we're done
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