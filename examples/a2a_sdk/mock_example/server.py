"""
A2A SDK Server Example — expose an agentflow graph as an A2A agent.

Run this first::

    python -m examples.a2a_sdk.server

Then in another terminal::

    python -m examples.a2a_sdk.client
"""

from agentflow.graph.state_graph import StateGraph
from agentflow.state.agent_state import AgentState
from agentflow.state.message import Message
from agentflow.utils.constants import END

from agentflow.a2a_integration import create_a2a_server, make_agent_card


# ------------------------------------------------------------------ #
#  Build a simple echo-style agentflow graph                           #
# ------------------------------------------------------------------ #


def echo_node(state: AgentState, config: dict) -> list[Message]:
    """Simple node that echoes the user's input back."""
    user_text = state.context[-1].text() if state.context else "nothing"
    return [
        Message.text_message(
            f"Echo from agentflow: {user_text}",
            role="assistant",
        )
    ]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("echo", echo_node)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    return graph.compile()


# ------------------------------------------------------------------ #
#  Serve it as an A2A agent                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    compiled = build_graph()

    card = make_agent_card(
        name="EchoAgent",
        description="A simple agentflow echo agent exposed via A2A protocol",
        url="http://localhost:9999/",
    )

    print("Starting A2A server on http://localhost:9999 ...")
    print("Agent card at  http://localhost:9999/.well-known/agent-card.json")
    print("Press Ctrl+C to stop.\n")

    create_a2a_server(compiled, card, port=9999)
