"""
agentflow.a2a_integration — official a2a-sdk bridge for agentflow.

This package exposes any agentflow ``CompiledGraph`` as a standard A2A
agent using the official ``a2a-sdk`` (pip: ``a2a-sdk``).  It also
provides a client helper to call remote A2A agents from within a graph.

Install the extra::

    pip install agentflow[a2a_sdk]

Quick start — server::

    from agentflow.a2a_integration import (
        AgentFlowExecutor,
        create_a2a_server,
        make_agent_card,
    )

    card = make_agent_card("My Agent", "Does cool stuff", "http://localhost:9999")
    create_a2a_server(compiled_graph, card, port=9999)

Quick start — client::

    from agentflow.a2a_integration import delegate_to_a2a_agent

    response = await delegate_to_a2a_agent("http://localhost:9999", "Hello!")
"""

from __future__ import annotations


try:
    import a2a  # noqa: F401 — probe for a2a-sdk availability
except ImportError as _exc:
    raise ImportError(
        "The 'a2a-sdk' package is required for agentflow.a2a_integration. "
        "Install it with:  pip install agentflow[a2a_sdk]"
    ) from _exc

from agentflow.a2a_integration.client import (
    create_a2a_client_node,
    delegate_to_a2a_agent,
)
from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.a2a_integration.server import (
    build_a2a_app,
    create_a2a_server,
    make_agent_card,
)


__all__ = [
    "AgentFlowExecutor",
    "build_a2a_app",
    "create_a2a_client_node",
    "create_a2a_server",
    "delegate_to_a2a_agent",
    "make_agent_card",
]
