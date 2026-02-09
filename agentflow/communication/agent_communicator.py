# """
# Agent Communicator - Extension for adding A2A communication to agents.

# This module provides a mixin/wrapper to add communication capabilities to
# any agent in the system.
# """

# import logging
# from collections.abc import Callable, Coroutine
# from typing import Any

# from agentflow.communication.a2a import A2ACommunicationManager
# from agentflow.protocols.acp import ACPMessage


# logger = logging.getLogger(__name__)


# class AgentCommunicator:
#     """
#     Adds A2A communication capabilities to agents.

#     Can be used as a mixin or wrapper to enable agent-to-agent communication.
#     """

#     def __init__(
#         self,
#         agent_id: str,
#         agent_name: str,
#         comm_manager: A2ACommunicationManager,
#         agent_type: str = "agent",
#         capabilities: list[str] | None = None,
#         metadata: dict[str, Any] | None = None,
#     ):
#         """
#         Initialize agent communicator.

#         Args:
#             agent_id: Unique agent identifier
#             agent_name: Human-readable agent name
#             comm_manager: A2A communication manager instance
#             agent_type: Type of agent
#             capabilities: List of agent capabilities
#             metadata: Additional metadata
#         """
#         self.agent_id = agent_id
#         self.agent_name = agent_name
#         self.comm_manager = comm_manager
#         self.agent_type = agent_type
#         self.capabilities = capabilities or []
#         self.metadata = metadata or {}
#         self._message_callbacks: list[Callable[[ACPMessage], Coroutine[Any, Any, None]]] = []

#     async def register(
#         self,
#         message_handler: Callable[[ACPMessage], Coroutine[Any, Any, ACPMessage | None]]
#         | None = None,
#     ) -> bool:
#         """
#         Register this agent with the communication manager.

#         Args:
#             message_handler: Optional custom message handler

#         Returns:
#             True if newly registered
#         """
#         handler = message_handler or self._default_message_handler

#         return await self.comm_manager.register_agent(
#             agent_id=self.agent_id,
#             agent_name=self.agent_name,
#             agent_type=self.agent_type,
#             capabilities=self.capabilities,
#             metadata=self.metadata,
#             message_handler=handler,
#         )

#     async def unregister(self) -> bool:
#         """
#         Unregister this agent from the communication manager.

#         Returns:
#             True if unregistered successfully
#         """
#         return await self.comm_manager.unregister_agent(self.agent_id)

#     async def send_message(
#         self,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         priority: int = 5,
#         ttl: int | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage | None:
#         """
#         Send a message to another agent.

#         Args:
#             recipient_id: Recipient agent ID
#             action: Action to perform
#             data: Message data
#             priority: Message priority (1=highest, 10=lowest)
#             ttl: Time-to-live in seconds
#             **kwargs: Additional message fields

#         Returns:
#             Response message if any
#         """
#         return await self.comm_manager.send_message(
#             sender_id=self.agent_id,
#             recipient_id=recipient_id,
#             action=action,
#             data=data,
#             priority=priority,
#             ttl=ttl,
#             **kwargs,
#         )

#     async def broadcast(
#         self,
#         action: str,
#         data: dict[str, Any] | None = None,
#         priority: int = 5,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Broadcast a message to all registered agents.

#         Args:
#             action: Action to broadcast
#             data: Message data
#             priority: Message priority
#             **kwargs: Additional message fields
#         """
#         await self.comm_manager.broadcast_message(
#             sender_id=self.agent_id,
#             action=action,
#             data=data,
#             priority=priority,
#             **kwargs,
#         )

#     async def notify(
#         self,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Send a notification to another agent (no response expected).

#         Args:
#             recipient_id: Recipient agent ID
#             action: Notification action
#             data: Notification data
#             **kwargs: Additional message fields
#         """
#         await self.comm_manager.send_notification(
#             sender_id=self.agent_id,
#             recipient_id=recipient_id,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#     async def respond(
#         self,
#         request_message: ACPMessage,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Respond to a received message.

#         Args:
#             request_message: Original request message
#             action: Response action
#             data: Response data
#             **kwargs: Additional message fields
#         """
#         await self.comm_manager.send_response(
#             request_message=request_message,
#             sender_id=self.agent_id,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#     async def subscribe(self, topic: str) -> None:
#         """
#         Subscribe to a topic.

#         Args:
#             topic: Topic name
#         """
#         await self.comm_manager.subscribe_to_topic(self.agent_id, topic)

#     async def unsubscribe(self, topic: str) -> None:
#         """
#         Unsubscribe from a topic.

#         Args:
#             topic: Topic name
#         """
#         await self.comm_manager.unsubscribe_from_topic(self.agent_id, topic)

#     async def publish(
#         self,
#         topic: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Publish a message to a topic.

#         Args:
#             topic: Topic name
#             action: Message action
#             data: Message data
#             **kwargs: Additional message fields
#         """
#         await self.comm_manager.publish_to_topic(
#             sender_id=self.agent_id,
#             topic=topic,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#     async def heartbeat(self) -> None:
#         """Send a heartbeat to keep agent active."""
#         await self.comm_manager.send_heartbeat(self.agent_id)

#     def on_message(
#         self,
#         callback: Callable[[ACPMessage], Coroutine[Any, Any, None]],
#     ) -> None:
#         """
#         Register a callback for incoming messages.

#         Args:
#             callback: Async function to handle messages
#         """
#         self._message_callbacks.append(callback)

#     async def _default_message_handler(self, message: ACPMessage) -> ACPMessage | None:
#         """
#         Default message handler that triggers callbacks.

#         Args:
#             message: Incoming message

#         Returns:
#             Response message if any
#         """
#         logger.debug(
#             f"Agent {self.agent_id} received message: "
#             f"{message.message_type} from {message.sender_id}"
#         )

#         # Trigger all registered callbacks
#         for callback in self._message_callbacks:
#             try:
#                 await callback(message)
#             except Exception as e:
#                 logger.error(
#                     f"Error in message callback for agent {self.agent_id}: {e}",
#                     exc_info=True,
#                 )

#         # Default: no response
#         return None

#     async def get_info(self):
#         """Get agent registry information."""
#         return await self.comm_manager.get_agent(self.agent_id)

#     async def list_other_agents(
#         self,
#         agent_type: str | None = None,
#         status: str | None = None,
#     ):
#         """
#         List other registered agents.

#         Args:
#             agent_type: Filter by agent type
#             status: Filter by status

#         Returns:
#             List of agent entries (excluding self)
#         """
#         agents = await self.comm_manager.list_agents(
#             agent_type=agent_type,
#             status=status,
#         )
#         return [agent for agent in agents if agent.agent_id != self.agent_id]

#     async def find_agents_with_capability(self, capability: str):
#         """
#         Find agents with a specific capability.

#         Args:
#             capability: Capability to search for

#         Returns:
#             List of matching agents
#         """
#         agents = await self.comm_manager.find_agents_by_capability(capability)
#         return [agent for agent in agents if agent.agent_id != self.agent_id]
