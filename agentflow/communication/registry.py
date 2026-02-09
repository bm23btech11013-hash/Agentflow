# """Agent Registry for managing active agents in the system."""

# import asyncio
# from datetime import UTC, datetime
# from typing import Any

# from pydantic import BaseModel, Field


# class AgentRegistryEntry(BaseModel):
#     """Entry for an agent in the registry."""

#     agent_id: str = Field(..., description="Unique agent identifier")
#     agent_name: str = Field(..., description="Human-readable agent name")
#     agent_type: str = Field(default="agent", description="Type of agent")
#     status: str = Field(default="active", description="Agent status")
#     capabilities: list[str] = Field(default_factory=list,
#       description="List of agent capabilities")
#     metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
#     registered_at: datetime = Field(
#         default_factory=lambda: datetime.now(UTC),
#         description="Registration timestamp",
#     )
#     last_heartbeat: datetime = Field(
#         default_factory=lambda: datetime.now(UTC),
#         description="Last heartbeat timestamp",
#     )


# class AgentRegistry:
#     """
#     Registry for managing active agents.

#     Provides agent discovery, registration, and lifecycle management.
#     """

#     def __init__(self, heartbeat_timeout: int = 60):
#         """
#         Initialize the agent registry.

#         Args:
#             heartbeat_timeout: Seconds before considering an agent inactive
#         """
#         self._agents: dict[str, AgentRegistryEntry] = {}
#         self._lock = asyncio.Lock()
#         self._heartbeat_timeout = heartbeat_timeout
#         self._cleanup_task: asyncio.Task | None = None

#     async def register(self, entry: AgentRegistryEntry) -> bool:
#         """
#         Register an agent.

#         Args:
#             entry: Agent registry entry

#         Returns:
#             True if registered successfully
#         """
#         async with self._lock:
#             if entry.agent_id in self._agents:
#                 # Update existing entry
#                 self._agents[entry.agent_id] = entry
#                 return False  # Was already registered
#             else:
#                 self._agents[entry.agent_id] = entry
#                 return True  # Newly registered

#     async def unregister(self, agent_id: str) -> bool:
#         """
#         Unregister an agent.

#         Args:
#             agent_id: ID of agent to unregister

#         Returns:
#             True if agent was unregistered
#         """
#         async with self._lock:
#             if agent_id in self._agents:
#                 del self._agents[agent_id]
#                 return True
#             return False

#     async def get(self, agent_id: str) -> AgentRegistryEntry | None:
#         """
#         Get an agent entry by ID.

#         Args:
#             agent_id: Agent ID

#         Returns:
#             Agent entry or None if not found
#         """
#         async with self._lock:
#             return self._agents.get(agent_id)

#     async def list_agents(
#         self,
#         agent_type: str | None = None,
#         status: str | None = None,
#     ) -> list[AgentRegistryEntry]:
#         """
#         List all registered agents with optional filters.

#         Args:
#             agent_type: Filter by agent type
#             status: Filter by status

#         Returns:
#             List of agent entries
#         """
#         async with self._lock:
#             agents = list(self._agents.values())

#             if agent_type:
#                 agents = [a for a in agents if a.agent_type == agent_type]

#             if status:
#                 agents = [a for a in agents if a.status == status]

#             return agents

#     async def update_heartbeat(self, agent_id: str) -> bool:
#         """
#         Update agent heartbeat timestamp.

#         Args:
#             agent_id: Agent ID

#         Returns:
#             True if updated successfully
#         """
#         async with self._lock:
#             if agent_id in self._agents:
#                 self._agents[agent_id].last_heartbeat = datetime.now(UTC)
#                 return True
#             return False

#     async def update_status(self, agent_id: str, status: str) -> bool:
#         """
#         Update agent status.

#         Args:
#             agent_id: Agent ID
#             status: New status

#         Returns:
#             True if updated successfully
#         """
#         async with self._lock:
#             if agent_id in self._agents:
#                 self._agents[agent_id].status = status
#                 return True
#             return False

#     async def find_by_capability(self, capability: str) -> list[AgentRegistryEntry]:
#         """
#         Find agents with a specific capability.

#         Args:
#             capability: Capability to search for

#         Returns:
#             List of agents with the capability
#         """
#         async with self._lock:
#             return [agent for agent in self._agents.values() if capability in agent.capabilities]

#     async def get_agent_count(self) -> int:
#         """Get total number of registered agents."""
#         async with self._lock:
#             return len(self._agents)

#     async def cleanup_inactive_agents(self) -> int:
#         """
#         Remove agents that haven't sent heartbeat within timeout.

#         Returns:
#             Number of agents removed
#         """
#         now = datetime.now(UTC)
#         removed_count = 0

#         async with self._lock:
#             inactive_ids = [
#                 agent_id
#                 for agent_id, agent in self._agents.items()
#                 if (now - agent.last_heartbeat).total_seconds() > self._heartbeat_timeout
#             ]

#             for agent_id in inactive_ids:
#                 del self._agents[agent_id]
#                 removed_count += 1

#         return removed_count

#     def start_cleanup_task(self, interval: int = 30):
#         """
#         Start background task to cleanup inactive agents.

#         Args:
#             interval: Cleanup interval in seconds
#         """
#         if self._cleanup_task is None or self._cleanup_task.done():
#             self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval))

#     async def _cleanup_loop(self, interval: int):
#         """Background cleanup loop."""
#         while True:
#             await asyncio.sleep(interval)
#             removed = await self.cleanup_inactive_agents()
#             if removed > 0:
#                 print(f"Cleaned up {removed} inactive agents")

#     def stop_cleanup_task(self):
#         """Stop the cleanup background task."""
#         if self._cleanup_task and not self._cleanup_task.done():
#             self._cleanup_task.cancel()

#     async def clear(self):
#         """Clear all registered agents."""
#         async with self._lock:
#             self._agents.clear()
