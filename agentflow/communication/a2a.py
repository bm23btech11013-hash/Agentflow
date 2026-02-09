# """
# Agent-to-Agent (A2A) Communication Manager.

# Provides high-level API for agent communication, including direct messaging,
# broadcasting, and pub-sub patterns.
# """

# import logging
# from collections.abc import Callable, Coroutine
# from typing import Any

# from agentflow.communication.registry import AgentRegistry, AgentRegistryEntry
# from agentflow.communication.router import MessageRouter
# from agentflow.protocols.acp import ACPMessage, ACPProtocol


# logger = logging.getLogger(__name__)


# class A2ACommunicationManager:
#     """
#     Agent-to-Agent Communication Manager.

#     Central manager for all agent-to-agent communication in the system.
#     Integrates registry and routing for seamless agent interactions.
#     """

#     def __init__(
#         self,
#         heartbeat_timeout: int = 60,
#         enable_cleanup: bool = True,
#         cleanup_interval: int = 30,
#     ):
#         """
#         Initialize the A2A communication manager.

#         Args:
#             heartbeat_timeout: Seconds before considering an agent inactive
#             enable_cleanup: Enable automatic cleanup of inactive agents
#             cleanup_interval: Cleanup interval in seconds
#         """
#         self.registry = AgentRegistry(heartbeat_timeout=heartbeat_timeout)
#         self.router = MessageRouter()
#         self._protocol = ACPProtocol()
#         self._enable_cleanup = enable_cleanup

#         if enable_cleanup:
#             self.registry.start_cleanup_task(cleanup_interval)

#     async def register_agent(
#         self,
#         agent_id: str,
#         agent_name: str,
#         agent_type: str = "agent",
#         capabilities: list[str] | None = None,
#         metadata: dict[str, Any] | None = None,
#         message_handler: Callable[[ACPMessage], Coroutine[Any, Any, ACPMessage | None]]
#         | None = None,
#     ) -> bool:
#         """
#         Register an agent in the system.

#         Args:
#             agent_id: Unique agent identifier
#             agent_name: Human-readable agent name
#             agent_type: Type of agent
#             capabilities: List of agent capabilities
#             metadata: Additional metadata
#             message_handler: Optional message handler function

#         Returns:
#             True if newly registered, False if updated existing
#         """
#         entry = AgentRegistryEntry(
#             agent_id=agent_id,
#             agent_name=agent_name,
#             agent_type=agent_type,
#             capabilities=capabilities or [],
#             metadata=metadata or {},
#         )

#         registered = await self.registry.register(entry)

#         # Register message handler if provided
#         if message_handler:
#             await self.router.register_handler(agent_id, message_handler)
#             await self.router.register_broadcast_handler(agent_id, message_handler)

#         logger.info(
#             f"Agent '{agent_name}' ({agent_id}) {'registered' if registered else 'updated'}"
#         )
#         return registered

#     async def unregister_agent(self, agent_id: str) -> bool:
#         """
#         Unregister an agent from the system.

#         Args:
#             agent_id: Agent ID

#         Returns:
#             True if unregistered successfully
#         """
#         unregistered = await self.registry.unregister(agent_id)
#         await self.router.unregister_handler(agent_id)

#         if unregistered:
#             logger.info(f"Agent {agent_id} unregistered")

#         return unregistered

#     async def send_message(
#         self,
#         sender_id: str,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         priority: int = 5,
#         ttl: int | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage | None:
#         """
#         Send a direct message from one agent to another.

#         Args:
#             sender_id: Sending agent ID
#             recipient_id: Recipient agent ID
#             action: Action to perform
#             data: Message data
#             priority: Message priority (1=highest, 10=lowest)
#             ttl: Time-to-live in seconds
#             **kwargs: Additional message fields

#         Returns:
#             Response message if any
#         """
#         # Check if sender is registered
#         sender = await self.registry.get(sender_id)
#         if not sender:
#             logger.warning(f"Sender agent {sender_id} not registered")
#             return None

#         # Check if recipient exists
#         recipient = await self.registry.get(recipient_id)
#         if not recipient:
#             logger.warning(f"Recipient agent {recipient_id} not registered")
#             return None

#         # Create and route message
#         message = self._protocol.create_request(
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             action=action,
#             data=data,
#             priority=priority,
#             ttl=ttl,
#             **kwargs,
#         )

#         logger.debug(f"Sending message from {sender_id} to {recipient_id}: {action}")
#         response = await self.router.route_message(message)

#         # Update heartbeat
#         await self.registry.update_heartbeat(sender_id)

#         return response

#     async def broadcast_message(
#         self,
#         sender_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         priority: int = 5,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Broadcast a message to all registered agents.

#         Args:
#             sender_id: Sending agent ID
#             action: Action to broadcast
#             data: Message data
#             priority: Message priority
#             **kwargs: Additional message fields
#         """
#         # Check if sender is registered
#         sender = await self.registry.get(sender_id)
#         if not sender:
#             logger.warning(f"Sender agent {sender_id} not registered")
#             return

#         # Create broadcast message
#         message = self._protocol.create_broadcast(
#             sender_id=sender_id,
#             action=action,
#             data=data,
#             priority=priority,
#             **kwargs,
#         )

#         logger.debug(f"Broadcasting message from {sender_id}: {action}")
#         await self.router.route_message(message)

#         # Update heartbeat
#         await self.registry.update_heartbeat(sender_id)

#     async def send_notification(
#         self,
#         sender_id: str,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Send a one-way notification (no response expected).

#         Args:
#             sender_id: Sending agent ID
#             recipient_id: Recipient agent ID
#             action: Notification action
#             data: Notification data
#             **kwargs: Additional message fields
#         """
#         message = self._protocol.create_notification(
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#         logger.debug(f"Sending notification from {sender_id} to {recipient_id}")
#         await self.router.route_message(message)

#     async def send_response(
#         self,
#         request_message: ACPMessage,
#         sender_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Send a response to a previous request.

#         Args:
#             request_message: Original request message
#             sender_id: Responding agent ID
#             action: Response action
#             data: Response data
#             **kwargs: Additional message fields
#         """
#         response = self._protocol.create_response(
#             request_message=request_message,
#             sender_id=sender_id,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#         logger.debug(f"Sending response from {sender_id} to {request_message.sender_id}")
#         await self.router.route_message(response)

#     async def subscribe_to_topic(self, agent_id: str, topic: str) -> None:
#         """
#         Subscribe an agent to a topic.

#         Args:
#             agent_id: Agent ID
#             topic: Topic name
#         """
#         await self.router.subscribe(agent_id, topic)
#         logger.info(f"Agent {agent_id} subscribed to topic: {topic}")

#     async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> None:
#         """
#         Unsubscribe an agent from a topic.

#         Args:
#             agent_id: Agent ID
#             topic: Topic name
#         """
#         await self.router.unsubscribe(agent_id, topic)
#         logger.info(f"Agent {agent_id} unsubscribed from topic: {topic}")

#     async def publish_to_topic(
#         self,
#         sender_id: str,
#         topic: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Publish a message to a topic.

#         Args:
#             sender_id: Sending agent ID
#             topic: Topic name
#             action: Message action
#             data: Message data
#             **kwargs: Additional message fields
#         """
#         message = self._protocol.create_broadcast(
#             sender_id=sender_id,
#             action=action,
#             data=data,
#             **kwargs,
#         )

#         logger.debug(f"Publishing to topic {topic} from {sender_id}")
#         await self.router.publish_to_topic(topic, message)

#     async def send_heartbeat(self, agent_id: str) -> None:
#         """
#         Send a heartbeat for an agent.

#         Args:
#             agent_id: Agent ID
#         """
#         await self.registry.update_heartbeat(agent_id)
#         logger.debug(f"Heartbeat received from {agent_id}")

#     async def get_agent(self, agent_id: str) -> AgentRegistryEntry | None:
#         """
#         Get agent information.

#         Args:
#             agent_id: Agent ID

#         Returns:
#             Agent registry entry or None
#         """
#         return await self.registry.get(agent_id)

#     async def list_agents(
#         self,
#         agent_type: str | None = None,
#         status: str | None = None,
#     ) -> list[AgentRegistryEntry]:
#         """
#         List all registered agents.

#         Args:
#             agent_type: Filter by agent type
#             status: Filter by status

#         Returns:
#             List of agent entries
#         """
#         return await self.registry.list_agents(agent_type=agent_type, status=status)

#     async def find_agents_by_capability(self, capability: str) -> list[AgentRegistryEntry]:
#         """
#         Find agents with a specific capability.

#         Args:
#             capability: Capability to search for

#         Returns:
#             List of matching agents
#         """
#         return await self.registry.find_by_capability(capability)

#     async def get_agent_count(self) -> int:
#         """Get total number of registered agents."""
#         return await self.registry.get_agent_count()

#     async def update_agent_status(self, agent_id: str, status: str) -> bool:
#         """
#         Update agent status.

#         Args:
#             agent_id: Agent ID
#             status: New status

#         Returns:
#             True if updated successfully
#         """
#         return await self.registry.update_status(agent_id, status)

#     async def shutdown(self):
#         """Shutdown the communication manager."""
#         logger.info("Shutting down A2A communication manager")
#         self.registry.stop_cleanup_task()
#         self.router.stop_processing()
#         await self.registry.clear()
#         await self.router.clear()
