"""
InvokeNodeHandler utilities for TAF agent graph execution.

This module provides the InvokeNodeHandler class, which manages the invocation of node functions,
tool nodes, and agent instances within the agent graph. It supports dependency injection,
callback hooks, event publishing, and error recovery for regular, tool-based, and agent nodes.

Classes:
    InvokeNodeHandler: Handles execution of node functions, tool nodes, and agents with DI
        and callbacks.

Usage:
    handler = InvokeNodeHandler(name, func, publisher)
    result = await handler.invoke(config, state)
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from injectq import Inject

from agentflow.exceptions import NodeError
from agentflow.graph.tool_node import ToolNode
from agentflow.graph.utils.utils import process_node_result
from agentflow.prebuilt.tools.handoff import is_handoff_tool
from agentflow.publisher import BasePublisher
from agentflow.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.publisher.publish import publish_event
from agentflow.state import AgentState, Message
from agentflow.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    call_sync_or_async,
)
from agentflow.utils.command import Command

from .handler_mixins import BaseLoggingMixin


if TYPE_CHECKING:
    from agentflow.graph.base_agent import BaseAgent


logger = logging.getLogger("agentflow.graph")


class InvokeNodeHandler(BaseLoggingMixin):
    """
    Handles invocation of node functions, tool nodes, and agent instances in the agent graph.

    Supports dependency injection, callback hooks, event publishing, and error recovery.

    Args:
        name (str): Name of the node.
        func (Callable | ToolNode | Agent): The function, ToolNode, or Agent to execute.
        publisher (BasePublisher, optional): Event publisher for execution events.
    """

    # Class-level cache for function signatures to avoid repeated inspection
    _signature_cache: dict[Callable, inspect.Signature] = {}

    @classmethod
    def clear_signature_cache(cls) -> None:
        """Clear the function signature cache. Useful for testing or memory management."""
        cls._signature_cache.clear()

    def __init__(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
        publisher: BasePublisher | None = Inject[BasePublisher],
    ):
        self.name = name
        self.func = func
        self.publisher = publisher

    async def _handle_single_tool(
        self,
        tool_call: dict[str, Any],
        state: AgentState,
        config: dict[str, Any],
    ) -> Message:
        """
        Execute a single tool call using the ToolNode.

        Args:
            tool_call (dict): Tool call specification.
            state (AgentState): Current agent state.
            config (dict): Node configuration.

        Returns:
            Message: Resulting message from tool execution.
        """
        function_name = tool_call.get("function", {}).get("name", "")
        function_args: dict = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        tool_call_id = tool_call.get("id", "")

        logger.info(
            "Node '%s' executing tool '%s' with %d arguments",
            self.name,
            function_name,
            len(function_args),
        )
        logger.debug("Tool arguments: %s", function_args)

        # Execute the tool function with injectable parameters
        tool_result = await self.func.invoke(  # type: ignore
            function_name,  # type: ignore
            function_args,
            tool_call_id=tool_call_id,
            state=state,
            config=config,
        )
        logger.debug("Node '%s' tool execution completed successfully", self.name)

        return tool_result

    async def _call_tools(
        self,
        last_message: Message,
        state: "AgentState",
        config: dict[str, Any],
    ) -> list[Message] | Command:
        """
        Execute all tool calls present in the last message.

        Args:
            last_message (Message): The last message containing tool calls.
            state (AgentState): Current agent state.
            config (dict): Node configuration.

        Returns:
            list[Message]: List of messages from tool executions.

        Raises:
            NodeError: If no tool calls are present.
        """
        logger.debug("Node '%s' calling tools from message", self.name)
        result: list[Message] = []
        if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
        ):
            # NEW: Check for handoff BEFORE executing any tools
            for tool_call in last_message.tools_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                is_handoff, target_agent = is_handoff_tool(tool_name)

                if is_handoff:
                    logger.info(
                        "Handoff detected in node '%s': tool '%s' -> agent '%s'",
                        self.name,
                        tool_name,
                        target_agent,
                    )

                    # Return Command to navigate directly to target agent
                    # This will be handled by the graph execution layer
                    return Command(  # type: ignore
                        update=None,
                        goto=target_agent,
                    )

            # Continue with normal tool execution if no handoff detected
            # Execute tool calls in parallel (preserve order of the input list)
            logger.info(
                "Node '%s' executing %d tool calls in parallel",
                self.name,
                len(last_message.tools_calls),
            )

            tasks = [
                self._handle_single_tool(tool_call, state, config)
                for tool_call in last_message.tools_calls
            ]

            # asyncio.gather preserves the order corresponding to the tasks list
            result = await asyncio.gather(*tasks)
        else:
            # No tool calls to execute, return available tools
            logger.exception("Node '%s': No tool calls to execute", self.name)
            raise NodeError(
                message="No tool calls to execute",
                error_code="NODE_002",
                context={"node_name": self.name},
            )

        return result

    def _get_cached_signature(self, func: Callable) -> inspect.Signature:
        """Get cached signature for a function, computing it if not cached."""
        if func not in self._signature_cache:
            self._signature_cache[func] = inspect.signature(func)
        return self._signature_cache[func]

    def _prepare_input_data(
        self,
        state: "AgentState",
        config: dict[str, Any],
    ) -> dict:
        """
        Prepare input data for function invocation, handling injectable parameters.
        Uses cached function signature to avoid repeated inspection overhead.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.

        Returns:
            dict: Input data for function call.

        Raises:
            TypeError: If required parameters are missing.
        """
        # Use cached signature inspection for performance
        sig = self._get_cached_signature(self.func)  # type: ignore Tool node won't come here
        input_data = {}
        default_data = {
            "state": state,
            "config": config,
        }

        # Collect all required parameters (excluding *args/**kwargs)
        required_params = []
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.default is inspect.Parameter.empty:
                required_params.append((param_name, param))

        # If function has exactly one required parameter and it's not 'state' or 'config',
        # assume it should receive the state (common pattern for simple lambdas like lambda s: s)
        if len(required_params) == 1:
            param_name, param = required_params[0]
            if param_name not in ["state", "config"]:
                input_data[param_name] = state
                return input_data

        # Otherwise, handle standard named parameters
        for param_name, param in sig.parameters.items():
            # Skip *args/**kwargs
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # check its state, config
            if param_name in ["state", "config"]:
                input_data[param_name] = default_data[param_name]
            # Include regular function arguments
            elif param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"Missing required parameter '{param_name}' for function '{self.func}'"
                )

        return input_data

    async def _call_normal_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any]:
        """
        Execute a regular node function with callback hooks and event publishing.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.
            callback_mgr (CallbackManager): Callback manager for hooks.

        Returns:
            dict: Result containing new state, messages, and next node.

        Raises:
            Exception: If function execution fails and cannot be recovered.
        """
        logger.debug("Node '%s' calling normal function", self.name)
        result: dict[str, Any] = {}

        logger.debug("Node '%s' is a regular function, executing with callbacks", self.name)
        # This is a regular function - likely AI function
        # Create callback context for AI invocation
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=getattr(self.func, "__name__", str(self.func)),
            metadata={"config": config},
        )

        # Event publishing logic (similar to stream_node_handler)

        input_data = self._prepare_input_data(
            state,
            config,
        )

        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={
                "node": self.name,
                "function_name": getattr(self.func, "__name__", str(self.func)),
                "last_message": last_message.model_dump() if last_message else None,
            },
        )
        publish_event(event)

        try:
            logger.debug("Node '%s' executing before_invoke callbacks", self.name)
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)
            logger.debug("Node '%s' executing function", self.name)
            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Function execution started"
            publish_event(event)

            # Execute the actual function
            result = await call_sync_or_async(
                self.func,  # type: ignore
                **input_data,
            )
            logger.debug("Node '%s' function execution completed", self.name)

            logger.debug("Node '%s' executing after_invoke callbacks", self.name)
            # Execute after_invoke callbacks
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            # Process result and publish END event
            messages = []
            new_state, messages, next_node = await process_node_result(result, state, messages)
            event.data["state"] = new_state.model_dump()
            event.event_type = EventType.END
            event.metadata["status"] = "Function execution completed"
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            event.data["next_node"] = next_node
            # mirror simple content + structured blocks for the last message
            if messages:
                last = messages[-1]
                event.content = last.text() if isinstance(last.content, list) else last.content
                if isinstance(last.content, list):
                    event.content_blocks = last.content

            publish_event(event)

            return {
                "state": new_state,
                "messages": messages,
                "next_node": next_node,
            }

        except Exception as e:
            logger.warning(
                "Node '%s' execution failed, executing error callbacks: %s", self.name, e
            )
            # Execute error callbacks
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if recovery_result is not None:
                logger.info(
                    "Node '%s' recovered from error using callback result",
                    self.name,
                )
                # Use recovery result instead of raising the error
                event.event_type = EventType.END
                event.metadata["status"] = "Function execution recovered from error"
                event.data["message"] = recovery_result.model_dump()
                event.content_type = [ContentType.MESSAGE, ContentType.STATE]
                publish_event(event)
                return {
                    "state": state,
                    "messages": [recovery_result],
                    "next_node": None,
                }
            # Re-raise the original error
            logger.error("Node '%s' could not recover from error", self.name)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Function execution failed: {e}"
            event.data["error"] = str(e)
            event.content_type = [ContentType.ERROR, ContentType.STATE]
            publish_event(event)
            raise

    async def _call_agent_node(
        self,
        state: "AgentState",
        config: dict[str, Any],
        callback_mgr: CallbackManager,
    ) -> dict[str, Any]:
        """Execute an Agent instance node with callback hooks and event publishing.

        Args:
            state (AgentState): Current agent state.
            config (dict): Node configuration.
            callback_mgr (CallbackManager): Callback manager for hooks.

        Returns:
            dict: Result containing new state, messages, and next node.
        """
        logger.debug("Node '%s' is an Agent instance, executing agent logic", self.name)

        agent = self.func  # type: ignore - func is Agent instance here

        # Create callback context for AI invocation
        context = CallbackContext(
            invocation_type=InvocationType.AI,
            node_name=self.name,
            function_name=getattr(agent, "__class__", type(agent)).__name__,
            metadata={"config": config},
        )

        input_data = {"state": state, "config": config}

        last_message = state.context[-1] if state.context and len(state.context) > 0 else None

        event = EventModel.default(
            config,
            data={"state": state.model_dump()},
            event=Event.NODE_EXECUTION,
            content_type=[ContentType.STATE],
            node_name=self.name,
            extra={
                "node": self.name,
                "function_name": type(agent).__name__,
                "last_message": last_message.model_dump() if last_message else None,
            },
        )
        publish_event(event)

        try:
            # Execute before_invoke callbacks
            input_data = await callback_mgr.execute_before_invoke(context, input_data)

            event.event_type = EventType.PROGRESS
            event.metadata["status"] = "Agent execution started"
            publish_event(event)

            # Execute the agent's logic
            converter = await agent.execute(state, config)  # type: ignore

            # Process the converter result (invoke for non-streaming)
            message = await converter.invoke()

            # Update state with new message
            new_state = state.model_copy(deep=True)
            new_state.context.append(message)

            result = {
                "state": new_state,
                "messages": [message],
                "next_node": None,
            }

            # Execute after_invoke callbacks
            result = await callback_mgr.execute_after_invoke(context, input_data, result)

            # Publish END event
            messages = result.get("messages", [])
            event.data["state"] = result["state"].model_dump()
            event.event_type = EventType.END
            event.metadata["status"] = "Agent execution completed"
            event.data["messages"] = [m.model_dump() for m in messages] if messages else []
            event.data["next_node"] = result.get("next_node")
            if messages:
                last = messages[-1]
                event.content = last.text() if isinstance(last.content, list) else last.content
                if isinstance(last.content, list):
                    event.content_blocks = last.content
            publish_event(event)

            return result

        except Exception as e:
            logger.warning(
                "Node '%s' agent execution failed, executing error callbacks: %s",
                self.name,
                e,
            )
            recovery_result = await callback_mgr.execute_on_error(context, input_data, e)

            if recovery_result is not None:
                logger.info("Node '%s' recovered from error using callback result", self.name)
                event.event_type = EventType.END
                event.metadata["status"] = "Agent execution recovered from error"
                event.data["message"] = recovery_result.model_dump()
                event.content_type = [ContentType.MESSAGE, ContentType.STATE]
                publish_event(event)
                return {
                    "state": state,
                    "messages": [recovery_result],
                    "next_node": None,
                }
            logger.error("Node '%s' could not recover from error", self.name)
            event.event_type = EventType.ERROR
            event.metadata["status"] = f"Agent execution failed: {e}"
            event.data["error"] = str(e)
            event.content_type = [ContentType.ERROR, ContentType.STATE]
            publish_event(event)
            raise

    async def invoke(
        self,
        config: dict[str, Any],
        state: AgentState,
        callback_mgr: CallbackManager = Inject[CallbackManager],
    ) -> dict[str, Any] | list[Message] | Command:
        """
        Execute the node function or ToolNode with dependency injection and callback hooks.

        Args:
            config (dict): Node configuration.
            state (AgentState): Current agent state.
            callback_mgr (CallbackManager, optional): Callback manager for hooks.

        Returns:
            dict | list[Message] | Command: Result of node execution (regular node,
            tool node, or a Command for handoff).

        Raises:
            NodeError: If execution fails or context is missing for tool nodes.
        """
        logger.info("Executing node '%s'", self.name)
        logger.debug(
            "Node '%s' execution with state context size=%d, config keys=%s",
            self.name,
            len(state.context) if state.context else 0,
            list(config.keys()) if config else [],
        )

        try:
            # Import Agent here to avoid circular dependency
            from agentflow.graph.agent import Agent
            from agentflow.graph.base_agent import BaseAgent

            if isinstance(self.func, Agent | BaseAgent):
                logger.debug("Node '%s' is an Agent instance, executing agent", self.name)
                result = await self._call_agent_node(
                    state,
                    config,
                    callback_mgr,
                )
            elif isinstance(self.func, ToolNode):
                logger.debug("Node '%s' is a ToolNode, executing tool calls", self.name)
                # This is tool execution - handled separately in ToolNode
                if state.context and len(state.context) > 0:
                    last_message = state.context[-1]
                    logger.debug("Node '%s' processing tool calls from last message", self.name)
                    result = await self._call_tools(
                        last_message,
                        state,
                        config,
                    )
                else:
                    # No context, return available tools
                    error_msg = "No context available for tool execution"
                    logger.error("Node '%s': %s", self.name, error_msg)
                    raise NodeError(
                        message=error_msg,
                        error_code="NODE_003",
                        context={"node_name": self.name},
                    )

            else:
                result = await self._call_normal_node(
                    state,
                    config,
                    callback_mgr,
                )

            logger.info("Node '%s' execution completed successfully", self.name)
            return result
        except Exception as e:
            # This is the final catch-all for node execution errors
            logger.exception("Node '%s' execution failed: %s", self.name, e)
            raise NodeError(
                message=f"Error in node '{self.name}': {e!s}",
                error_code="NODE_001",
                context={
                    "node_name": self.name,
                    "error_type": type(e).__name__,
                },
            ) from e
