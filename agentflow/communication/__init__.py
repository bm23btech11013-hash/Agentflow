"""Agent-to-Agent communication module."""

from agentflow.communication.a2a import A2ACommunicationManager
from agentflow.communication.registry import AgentRegistry, AgentRegistryEntry
from agentflow.communication.router import MessageRouter


__all__ = [
    "A2ACommunicationManager",
    "AgentRegistry",
    "AgentRegistryEntry",
    "MessageRouter",
]
