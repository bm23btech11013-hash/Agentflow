"""Execution helpers for Agent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from injectq import Inject, InjectQ
from injectq.utils.exceptions import DependencyNotFoundError

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph.tool_node import ToolNode
from agentflow.state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.utils.converter import convert_messages


logger = logging.getLogger("agentflow.agent")


class AgentExecutionMixin:
    """Execution flow, tool resolution, and provider dispatch helpers."""

    def _setup_tools(self) -> ToolNode | None:
        """Normalize the tools input to a ToolNode instance."""
        if self.tools is None:
            logger.debug("No tools provided")
            return None

        if isinstance(self.tools, ToolNode):
            logger.debug("Tools already a ToolNode instance")
            return self.tools

        logger.debug("Converting %d tool functions to ToolNode", len(self.tools))
        return ToolNode(self.tools)

    async def _trim_context(
        self,
        state: AgentState,
        context_manager: BaseContextManager | None = Inject[BaseContextManager],
    ) -> AgentState:
        """Trim state context when a context manager is configured."""
        if not self.trim_context:
            logger.debug("Context trimming not enabled")
            return state

        if context_manager is None:
            logger.warning("trim_context is enabled but no context manager is available")
            return state

        try:
            new_state = await context_manager.atrim_context(state)
            logger.debug("Context trimmed using context manager")
            return new_state
        except AttributeError:
            logger.warning(
                "trim_context is enabled but no BaseContextManager is registered. "
                "Skipping context trimming."
            )
            return state

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Route requests to the active provider and API style."""
        logger.debug(
            "Calling LLM: provider=%s, output_type=%s, model=%s, stream=%s",
            self.provider,
            self.output_type,
            self.model,
            stream,
        )

        if self.provider == "openai":
            if self.api_style == "responses":
                if self.base_url:
                    try:
                        result = await self._call_openai_responses(
                            messages, tools, stream, **kwargs
                        )
                        self._effective_api_style = "responses"
                        return result
                    except Exception as exc:
                        logger.warning(
                            "Responses API not supported at %s (%s). "
                            "Falling back to chat.completions.create().",
                            self.base_url,
                            exc,
                        )
                        self._effective_api_style = "chat"
                        if self.reasoning_config and self.reasoning_config.get("effort"):
                            kwargs.setdefault("reasoning_effort", self.reasoning_config["effort"])
                        return await self._call_openai(messages, tools, stream, **kwargs)

                self._effective_api_style = "responses"
                return await self._call_openai_responses(messages, tools, stream, **kwargs)

            self._effective_api_style = "chat"
            if self.reasoning_config and self.reasoning_config.get("effort"):
                kwargs.setdefault("reasoning_effort", self.reasoning_config["effort"])
            if self.base_url and self.reasoning_config:
                existing_extra = kwargs.get("extra_body", {})
                existing_extra["reasoning"] = self.reasoning_config
                kwargs["extra_body"] = existing_extra
            return await self._call_openai(messages, tools, stream, **kwargs)

        if self.provider == "google":
            return await self._call_google(messages, tools, stream, **kwargs)

        raise ValueError(f"Unsupported provider: {self.provider}")

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> ModelResponseConverter:
        """Execute the Agent node against the current graph state."""
        container = InjectQ.get_instance()

        state = await self._trim_context(state)
        messages = convert_messages(
            system_prompts=self.system_prompt,
            state=state,
            extra_messages=self.extra_messages or [],
        )
        is_stream = config.get("is_stream", False)

        if state.context and state.context[-1].role == "tool":
            response = await self._call_llm(messages=messages, stream=is_stream)
        else:
            tools = await self._resolve_tools(container)
            response = await self._call_llm(
                messages=messages,
                tools=tools if tools else None,
                stream=is_stream,
            )

        converter_key = self._get_converter_key()
        return ModelResponseConverter(response, converter=converter_key)

    async def _resolve_tools(self, container: InjectQ) -> list[dict[str, Any]]:
        """Resolve tool definitions from inline tools and named ToolNodes."""
        tools: list[dict[str, Any]] = []
        if self._tool_node:
            tools = await self._tool_node.all_tools(tags=self.tools_tags)

        if not self.tool_node_name:
            return tools

        try:
            node = container.call_factory("get_node", self.tool_node_name)
        except (KeyError, DependencyNotFoundError):
            logger.warning(
                "ToolNode with name '%s' not found in InjectQ registry.",
                self.tool_node_name,
            )
            return tools

        if node and isinstance(node.func, ToolNode):
            return await node.func.all_tools(tags=self.tools_tags)
        return tools

    def _extract_prompt(self, messages: list[dict[Any, Any]]) -> str:
        """Extract the last user message as a plain string for non-chat generation endpoints.

        Used by both OpenAI (image / audio) and Google (image / video / audio) providers.
        """
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return str(content) if content else ""
        return ""

    def _get_converter_key(self) -> str:
        """Return the correct response converter key for the active provider."""
        effective = getattr(self, "_effective_api_style", self.api_style)
        if self.provider == "openai" and effective == "responses":
            return "openai_responses"
        return self.provider
