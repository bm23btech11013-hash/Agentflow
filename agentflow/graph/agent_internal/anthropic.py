"""Anthropic request helpers for Agent."""

from __future__ import annotations

import logging
from typing import Any

from .constants import CALL_EXCLUDED_KWARGS


logger = logging.getLogger("agentflow.agent")


class AgentAnthropicMixin:
    """Anthropic (Claude) API request helpers."""

    def _convert_to_anthropic_format(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert chat-completion style messages into Anthropic message format.

        Anthropic separates the system prompt from the messages array and uses
        a ``tool_result`` content block for tool responses.

        Returns:
            (system_prompt_str | None, list of Anthropic-style message dicts)
        """
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # System messages are accumulated into a single system string
            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue

            # Tool result messages become user messages with tool_result blocks
            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(content) if content else "",
                            }
                        ],
                    }
                )
                continue

            # Assistant messages with tool calls
            if role == "assistant" and message.get("tool_calls"):
                parts: list[dict[str, Any]] = []
                if content:
                    parts.append({"type": "text", "text": str(content)})
                for tool_call in message["tool_calls"]:
                    import json

                    function = tool_call.get("function", {})
                    try:
                        args = json.loads(function.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        args = {}
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "input": args,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": parts})
                continue

            # Regular user / assistant messages
            anthropic_messages.append({"role": role, "content": str(content) if content else ""})

        system_prompt = "\n".join(system_parts) if system_parts else None
        return system_prompt, anthropic_messages

    def _convert_tools_to_anthropic_format(self, tools: list) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool definitions to Anthropic tool format."""
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                function = tool["function"]
                anthropic_tool: dict[str, Any] = {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    async def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the Anthropic messages API."""
        call_kwargs: dict[str, Any] = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in CALL_EXCLUDED_KWARGS
        }

        system_prompt, anthropic_messages = self._convert_to_anthropic_format(messages)

        if system_prompt:
            call_kwargs["system"] = system_prompt

        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            if anthropic_tools:
                call_kwargs["tools"] = anthropic_tools

        # Apply extended thinking if reasoning_config is provided
        if self.reasoning_config and self.output_type == "text":
            budget = self.reasoning_config.get("budget_tokens")
            effort = self.reasoning_config.get("effort")
            if budget is None and effort:
                _effort_budgets = {"low": 1024, "medium": 8000, "high": 16000}
                budget = _effort_budgets.get(effort, 8000)
            if budget is not None:
                call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}

        # Ensure max_tokens is always set (required by Anthropic)
        call_kwargs.setdefault("max_tokens", 8192)

        if self.output_type == "text":
            if stream:
                logger.debug(
                    "Calling Anthropic messages.create(stream=True) with model=%s", self.model
                )
                # Use stream=True to get an AsyncMessageStream (async iterable)
                return await self.client.messages.create(
                    model=self.model,
                    messages=anthropic_messages,
                    stream=True,
                    **call_kwargs,
                )

            logger.debug("Calling Anthropic messages.create with model=%s", self.model)
            return await self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                **call_kwargs,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for Anthropic provider")
