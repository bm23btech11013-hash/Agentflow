"""
Pattern 3 — GeneralAgent (port 10002).

A catch-all agent for any query that doesn't match currency or weather.
The Smart Client routes unmatched queries here.

Uses LiteLLM directly (no tools needed).

Usage::

    python -m examples.a2a_sdk.pattern3_smart_client.general_agent
"""

from __future__ import annotations

import logging

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
from agentflow.graph import StateGraph
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  LLM node — no tools, just answers questions                         #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a helpful general assistant.
Answer any question clearly and concisely.
If the user asks about currency or weather, answer as best you can.\
"""


async def llm_node(state: AgentState):
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )
    response = await acompletion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
    )
    return ModelResponseConverter(response, converter="litellm")


# ------------------------------------------------------------------ #
#  Build the graph (simple: one node → END)                            #
# ------------------------------------------------------------------ #


def build_general_graph() -> CompiledGraph:
    graph = StateGraph()
    graph.add_node("MAIN", llm_node)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    return graph.compile()


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #


def run_server() -> None:
    compiled = build_general_graph()
    card = make_agent_card(
        name="GeneralAgent",
        description="General-purpose assistant for any query.",
        url="http://localhost:10002/",
        skills=[
            AgentSkill(
                id="general_qa",
                name="General Q&A",
                description="Answer any general knowledge question.",
                tags=["general", "knowledge", "qa", "assistant"],
                examples=[
                    "What is the capital of France?",
                    "Explain quantum computing",
                    "Who wrote Hamlet?",
                ],
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=AgentFlowExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("  [GeneralAgent] Running on http://localhost:10002")
    uvicorn.run(app.build(), host="0.0.0.0", port=10002)


if __name__ == "__main__":
    run_server()
