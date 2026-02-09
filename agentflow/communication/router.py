# """Message router for agent-to-agent communication."""

# import asyncio
# import logging
# from collections import defaultdict
# from collections.abc import Callable, Coroutine
# from typing import Any

# from agentflow.protocols.acp import ACPMessage


# logger = logging.getLogger(__name__)


# MessageHandler = Callable[[ACPMessage], Coroutine[Any, Any, ACPMessage | None]]


# class MessageRouter:
#     """
#     Routes messages between agents.

#     Handles message delivery, subscriptions, and routing logic.
#     """

#     def __init__(self):
#         """Initialize the message router."""
#         self._handlers: dict[str, MessageHandler] = {}
#         self._subscribers: dict[str, list[str]] = defaultdict(list)
#         self._broadcast_handlers: list[tuple[str, MessageHandler]] = []
#         self._message_queue: asyncio.Queue[ACPMessage] = asyncio.Queue()
#         self._processing_task: asyncio.Task | None = None
#         self._lock = asyncio.Lock()

#     async def register_handler(
#         self,
#         agent_id: str,
#         handler: MessageHandler,
#     ) -> None:
#         """
#         Register a message handler for an agent.

#         Args:
#             agent_id: Agent ID
#             handler: Async function to handle incoming messages
#         """
#         async with self._lock:
#             self._handlers[agent_id] = handler
#             logger.debug(f"Registered handler for agent: {agent_id}")

#     async def unregister_handler(self, agent_id: str) -> None:
#         """
#         Unregister a message handler.

#         Args:
#             agent_id: Agent ID
#         """
#         async with self._lock:
#             if agent_id in self._handlers:
#                 del self._handlers[agent_id]
#                 logger.debug(f"Unregistered handler for agent: {agent_id}")

#     async def register_broadcast_handler(
#         self,
#         agent_id: str,
#         handler: MessageHandler,
#     ) -> None:
#         """
#         Register a handler for broadcast messages.

#         Args:
#             agent_id: Agent ID
#             handler: Async function to handle broadcast messages
#         """
#         async with self._lock:
#             self._broadcast_handlers.append((agent_id, handler))
#             logger.debug(f"Registered broadcast handler for agent: {agent_id}")

#     async def subscribe(self, agent_id: str, topic: str) -> None:
#         """
#         Subscribe an agent to a topic.

#         Args:
#             agent_id: Agent ID
#             topic: Topic to subscribe to
#         """
#         async with self._lock:
#             if agent_id not in self._subscribers[topic]:
#                 self._subscribers[topic].append(agent_id)
#                 logger.debug(f"Agent {agent_id} subscribed to topic: {topic}")

#     async def unsubscribe(self, agent_id: str, topic: str) -> None:
#         """
#         Unsubscribe an agent from a topic.

#         Args:
#             agent_id: Agent ID
#             topic: Topic to unsubscribe from
#         """
#         async with self._lock:
#             if agent_id in self._subscribers[topic]:
#                 self._subscribers[topic].remove(agent_id)
#                 logger.debug(f"Agent {agent_id} unsubscribed from topic: {topic}")

#     async def route_message(self, message: ACPMessage) -> ACPMessage | None:
#         """
#         Route a message to its destination.

#         Args:
#             message: Message to route

#         Returns:
#             Response message if any
#         """
#         try:
#             # Handle broadcast messages
#             if message.is_broadcast():
#                 await self._handle_broadcast(message)
#                 return None

#             # Handle direct messages
#             async with self._lock:
#                 handler = self._handlers.get(message.recipient_id)

#             if handler:
#                 try:
#                     response = await handler(message)
#                     logger.debug(
#                         f"Message {message.message_id} delivered to {message.recipient_id}"
#                     )
#                     return response
#                 except Exception as e:
#                     logger.error(
#                         f"Error handling message {message.message_id}: {e}",
#                         exc_info=True,
#                     )
#                     return None
#             else:
#                 logger.warning(f"No handler found for agent: {message.recipient_id}")
#                 return None

#         except Exception as e:
#             logger.error(f"Error routing message: {e}", exc_info=True)
#             return None

#     async def _handle_broadcast(self, message: ACPMessage) -> None:
#         """
#         Handle broadcast message delivery.

#         Args:
#             message: Broadcast message
#         """
#         async with self._lock:
#             handlers = list(self._broadcast_handlers)

#         # Deliver to all broadcast handlers
#         tasks = []
#         for agent_id, handler in handlers:
#             if agent_id != message.sender_id:  # Don't send back to sender
#                 tasks.append(self._safe_handle(handler, message, agent_id))

#         if tasks:
#             await asyncio.gather(*tasks, return_exceptions=True)

#     async def _safe_handle(
#         self,
#         handler: MessageHandler,
#         message: ACPMessage,
#         agent_id: str,
#     ) -> None:
#         """
#         Safely execute a handler with error catching.

#         Args:
#             handler: Message handler
#             message: Message to handle
#             agent_id: Agent ID for logging
#         """
#         try:
#             await handler(message)
#         except Exception as e:
#             logger.error(
#                 f"Error in broadcast handler for agent {agent_id}: {e}",
#                 exc_info=True,
#             )

#     async def publish_to_topic(self, topic: str, message: ACPMessage) -> None:
#         """
#         Publish a message to a topic.

#         Args:
#             topic: Topic name
#             message: Message to publish
#         """
#         async with self._lock:
#             subscribers = self._subscribers.get(topic, []).copy()

#         for agent_id in subscribers:
#             if agent_id != message.sender_id:  # Don't send back to sender
#                 async with self._lock:
#                     handler = self._handlers.get(agent_id)

#                 if handler:
#                     try:
#                         await handler(message)
#                         logger.debug(f"Message published to {agent_id} on topic {topic}")
#                     except Exception as e:
#                         logger.error(
#                             f"Error publishing to {agent_id}: {e}",
#                             exc_info=True,
#                         )

#     def start_processing(self):
#         """Start the message processing task."""
#         if self._processing_task is None or self._processing_task.done():
#             self._processing_task = asyncio.create_task(self._process_queue())
#             logger.info("Message processing started")

#     async def _process_queue(self):
#         """Process messages from the queue."""
#         while True:
#             try:
#                 message = await self._message_queue.get()
#                 await self.route_message(message)
#                 self._message_queue.task_done()
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 logger.error(f"Error processing message queue: {e}", exc_info=True)

#     def stop_processing(self):
#         """Stop the message processing task."""
#         if self._processing_task and not self._processing_task.done():
#             self._processing_task.cancel()
#             logger.info("Message processing stopped")

#     async def enqueue_message(self, message: ACPMessage):
#         """
#         Add a message to the processing queue.

#         Args:
#             message: Message to enqueue
#         """
#         await self._message_queue.put(message)

#     async def get_queue_size(self) -> int:
#         """Get the current message queue size."""
#         return self._message_queue.qsize()

#     async def clear(self):
#         """Clear all handlers and subscriptions."""
#         async with self._lock:
#             self._handlers.clear()
#             self._subscribers.clear()
#             self._broadcast_handlers.clear()
