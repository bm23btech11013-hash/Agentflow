"""
Pattern 4 — PlannerAgent as an **agentflow ReAct graph** (port 10001).

Built as a proper ``StateGraph`` following the same ReAct tool-calling
pattern as the CurrencyAgent — but instead of calling an HTTP API tool,
the LLM calls ``delegate_to_currency_agent``, a tool that delegates to
the remote CurrencyAgent via the A2A protocol.

This gives the PlannerAgent all agentflow benefits automatically:
- **Conversation memory** via the checkpointer (thread_id = context_id)
- **Streaming** via ``graph.astream()``
- **State management** via a custom ``PlannerState(AgentState)``

Flow::

    User ─► LLM node
              │
              ├─ tool call? ──► TOOL node (delegate_to_currency_agent) ──► LLM
              │
              └─ no tool   ──► END

Usage::

    python -m examples.a2a_sdk.pattern4_planner_delegates.planner_agent
"""

from __future__ import annotations

import logging
import uuid

import httpx
import uvicorn
from dotenv import load_dotenv
from litellm import acompletion

from a2a.client import A2AClient
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentSkill,
    Message as A2AMessage,
    MessageSendParams,
    Role,
    SendMessageRequest,
    TaskState,
    TextPart,
)

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.a2a_integration import make_agent_card
from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.graph import StateGraph, ToolNode
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState, Message as AFMessage
from agentflow.utils.constants import END, ResponseGranularity
from agentflow.utils.converter import convert_messages

load_dotenv()
logger = logging.getLogger(__name__)

CURRENCY_AGENT_URL = "http://localhost:10000"


# ------------------------------------------------------------------ #
#  Custom state — carries A2A context_id into the graph                #
# ------------------------------------------------------------------ #


class PlannerState(AgentState):
    """Extends AgentState with the A2A ``context_id``.

    The executor sets this field on every invocation so that the
    ``delegate_to_currency_agent`` tool can forward the same
    ``context_id`` to the remote CurrencyAgent — keeping both agents'
    conversation histories in sync.
    """

    a2a_context_id: str = ""


# ------------------------------------------------------------------ #
#  A2A client helper — sends a message and detects INPUT_REQUIRED      #
# ------------------------------------------------------------------ #


def _text_from_parts(parts) -> str:
    """Extract text from A2A message parts (handles Part union)."""
    for p in parts or []:
        if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
            return p.root.text
        if hasattr(p, "text") and isinstance(p.text, str) and p.text:
            return p.text
    return ""


async def _send_to_currency_agent(text: str, *, context_id: str = "") -> dict:
    """Low-level A2A call to the CurrencyAgent.

    Returns ``{"status": "completed"|"input_required"|"failed", "text": ...}``.
    """
    async with httpx.AsyncClient(timeout=60.0) as http:
        client = A2AClient(httpx_client=http, url=CURRENCY_AGENT_URL)
        msg_kwargs: dict = {}
        if context_id:
            msg_kwargs["context_id"] = context_id
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(
                message=A2AMessage(
                    role=Role.user,
                    message_id=str(uuid.uuid4()),
                    parts=[TextPart(text=text)],
                    **msg_kwargs,
                ),
            ),
        )
        response = await client.send_message(request)
        result = response.root

        if hasattr(result, "error") and result.error:
            return {"status": "failed", "text": str(result.error)}

        payload = result.result
        state_str = str(
            getattr(getattr(payload, "status", None), "state", "")
        ).lower()

        if "input_required" in state_str:
            txt = (
                _text_from_parts(payload.status.message.parts)
                if payload.status and payload.status.message
                else "Please provide more info."
            )
            return {"status": "input_required", "text": txt}

        # COMPLETED — try artifacts, then status message
        txt = ""
        if hasattr(payload, "artifacts") and payload.artifacts:
            for a in payload.artifacts:
                txt += _text_from_parts(a.parts)
        if not txt and payload.status and payload.status.message:
            txt = _text_from_parts(payload.status.message.parts)
        return {"status": "completed", "text": txt or "No response."}


# ------------------------------------------------------------------ #
#  Tool — the LLM calls this to delegate to CurrencyAgent             #
# ------------------------------------------------------------------ #


async def delegate_to_currency_agent(
    query: str,
    state: PlannerState | None = None,
) -> str:
    """Delegate a currency-related query to the remote CurrencyAgent.

    Use this tool for ANY query about currency conversion, exchange
    rates, or money.  Pass the user's FULL question as ``query``.

    Args:
        query: The user's currency-related request
              (e.g. "Convert 100 USD to EUR").
    """
    planner_ctx_id = state.a2a_context_id if state else ""
    currency_ctx_id = f"{planner_ctx_id}:currency" if planner_ctx_id else ""
    result = await _send_to_currency_agent(query, context_id=currency_ctx_id)

    if result["status"] == "input_required":
        return (
            f"The CurrencyAgent needs more information from the user: "
            f"{result['text']}\n"
            f"Please relay this question to the user exactly as stated."
        )
    if result["status"] == "completed":
        return result["text"]
    return f"CurrencyAgent error: {result['text']}"


tool_node = ToolNode([delegate_to_currency_agent])


# ------------------------------------------------------------------ #
#  LLM node — standard ReAct pattern                                   #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "You are a helpful planner assistant.\n\n"
    "STRICT RULES:\n"
    "1. If the user's message mentions currency, conversion, exchange rate, "
    "money, INR, USD, EUR, GBP, or ANY currency code — you MUST call "
    "delegate_to_currency_agent with the user's EXACT message as the query. "
    "Do NOT try to ask clarifying questions yourself. "
    "Do NOT answer currency queries on your own. "
    "ALWAYS delegate, even if the query seems incomplete.\n"
    "2. For everything else (general knowledge, greetings, etc.), answer directly.\n"
    "3. If the tool says the CurrencyAgent needs more information, "
    "relay the question to the user EXACTLY as stated — do NOT guess or rephrase."
)


async def llm_node(state: PlannerState):
    """Call the LLM, providing the delegation tool when appropriate."""
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )

    # After a tool result, call without tools so the LLM summarises.
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


def should_use_tools(state: PlannerState) -> str:
    """Route to TOOL if the LLM made a tool call, else END."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last = state.context[-1]

    if (
        hasattr(last, "tools_calls")
        and last.tools_calls
        and len(last.tools_calls) > 0
        and last.role == "assistant"
    ):
        return "TOOL"

    if last.role == "tool":
        return "MAIN"

    return END


# ------------------------------------------------------------------ #
#  Graph builder                                                       #
# ------------------------------------------------------------------ #


def build_planner_graph() -> CompiledGraph:
    """Build and compile the PlannerAgent ReAct graph.

    ::

        MAIN (LLM) ──conditional──► TOOL (delegate_to_currency_agent) ──► MAIN ──► END
    """
    graph = StateGraph[PlannerState](PlannerState())
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
#  INPUT_REQUIRED detection — checks the graph state, not heuristics   #
# ------------------------------------------------------------------ #

# Marker phrase that delegate_to_currency_agent injects when
# CurrencyAgent returns input_required.
_CURRENCY_NEEDS_INPUT_MARKER = "needs more information"


def _tool_requested_input(result: dict) -> bool:
    """Return True if the delegation tool received INPUT_REQUIRED.

    Instead of guessing from the final assistant text (which wrongly
    catches normal questions like "How can I help?"), we inspect the
    graph state for tool-result messages containing the marker phrase
    that ``delegate_to_currency_agent`` writes when CurrencyAgent
    returns ``input_required``.
    """
    state = result.get("state")
    if state is None:
        return False
    # Walk backwards through context — look for a recent tool result
    # with the marker.  Stop at the last user message (we only care
    # about the current turn's tool results).
    for msg in reversed(state.context):
        if msg.role == "user":
            break
        if msg.role == "tool":
            text = (msg.text() or "").lower()
            if _CURRENCY_NEEDS_INPUT_MARKER in text:
                return True
    return False


# ------------------------------------------------------------------ #
#  Executor — wraps the agentflow graph with INPUT_REQUIRED relay      #
# ------------------------------------------------------------------ #


class PlannerAgentExecutor(AgentFlowExecutor):
    """Runs the planner ReAct graph and detects INPUT_REQUIRED.

    When the CurrencyAgent asks for more info, the tool returns that
    info to the LLM, the LLM relays the question, and this executor
    sets ``TaskState.input_required`` on the A2A task.  The client's
    next message resumes the same conversation via the checkpointer.
    """

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

            # thread_id = context_id → checkpointer restores conversation
            run_config = {
                "thread_id": context.context_id or context.task_id or "",
            }

            result = await self.graph.ainvoke(
                {
                    "messages": messages,
                    # Merge a2a_context_id into PlannerState so the tool
                    # can forward it to CurrencyAgent.
                    "state": {
                        "a2a_context_id": context.context_id or "",
                    },
                },
                config=run_config,
                response_granularity=ResponseGranularity.FULL,
            )
            response_text = self._extract_response_text(result)

            if _tool_requested_input(result):
                msg = updater.new_agent_message(
                    parts=[TextPart(text=response_text)]
                )
                await updater.update_status(
                    TaskState.input_required, message=msg
                )
            else:
                await updater.add_artifact(
                    [TextPart(text=response_text)]
                )
                await updater.complete()

        except Exception as exc:
            logger.exception("PlannerAgentExecutor failed")
            err = updater.new_agent_message(
                parts=[TextPart(text=f"Error: {exc!s}")]
            )
            await updater.failed(message=err)


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    compiled = build_planner_graph()

    card = make_agent_card(
        name="PlannerAgent",
        description=(
            "Planner built as an agentflow ReAct graph.  Delegates "
            "currency queries to CurrencyAgent via tool-calling and "
            "relays INPUT_REQUIRED across agent boundaries."
        ),
        url="http://localhost:10001/",
        streaming=True,
        skills=[
            AgentSkill(
                id="planning",
                name="Planning & Delegation",
                description=(
                    "Routes queries — delegates currency to CurrencyAgent "
                    "via A2A, answers general questions directly."
                ),
                tags=["planner", "routing", "currency"],
                examples=[
                    "Convert 100 USD to EUR",
                    "What is the capital of France?",
                ],
            )
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=PlannerAgentExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("[PlannerAgent] http://localhost:10001  (CurrencyAgent must be on :10000)")
    uvicorn.run(app.build(), host="0.0.0.0", port=10001)
