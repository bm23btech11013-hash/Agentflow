"""OpenAI request helpers for Agent."""

from __future__ import annotations

import logging
from typing import Any

from .constants import CALL_EXCLUDED_KWARGS


logger = logging.getLogger("agentflow.agent")


class AgentOpenAIMixin:
    """OpenAI and OpenAI-compatible API request helpers."""

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call OpenAI chat, image, or audio endpoints."""
        call_kwargs = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in CALL_EXCLUDED_KWARGS
        }

        if self.output_type == "text":
            if tools:
                call_kwargs["tools"] = tools

            logger.debug("Calling OpenAI chat.completions.create with model=%s", self.model)
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **call_kwargs,
            )

        if self.output_type == "image":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling OpenAI images.generate with model=%s", self.model)
            return await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                **call_kwargs,
            )

        if self.output_type == "audio":
            text = self._extract_prompt(messages)
            logger.debug("Calling OpenAI audio.speech.create with model=%s", self.model)
            return await self.client.audio.speech.create(
                model=self.model,
                input=text,
                **call_kwargs,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for OpenAI provider")

    async def _call_openai_responses(  # noqa: PLR0912
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the OpenAI Responses API using chat-style messages as input."""
        call_kwargs: dict[str, Any] = {
            key: value
            for key, value in {**self.llm_kwargs, **kwargs}.items()
            if key not in CALL_EXCLUDED_KWARGS
        }

        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "")
            if role == "system":
                instructions_parts.append(str(message.get("content", "")))
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", ""),
                        "output": str(message.get("content", "")),
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                text_content = message.get("content", "")
                if text_content:
                    input_items.append({"role": "assistant", "content": text_content})

                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                            "call_id": tool_call.get("id", ""),
                        }
                    )
            else:
                input_items.append({"role": role, "content": message.get("content", "")})

        instructions = "\n".join(instructions_parts) if instructions_parts else None

        responses_tools: list[dict[str, Any]] | None = None
        if tools:
            responses_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    function = tool["function"]
                    response_tool: dict[str, Any] = {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                    }
                    if "parameters" in function:
                        response_tool["parameters"] = function["parameters"]
                    if "strict" in function:
                        response_tool["strict"] = function["strict"]
                    responses_tools.append(response_tool)
                else:
                    responses_tools.append(tool)

        if instructions:
            call_kwargs["instructions"] = instructions
        if responses_tools:
            call_kwargs["tools"] = responses_tools
        if self.reasoning_config:
            call_kwargs["reasoning"] = self.reasoning_config

        call_kwargs.pop("reasoning_effort", None)

        logger.debug("Calling OpenAI responses.create with model=%s", self.model)
        return await self.client.responses.create(
            model=self.model,
            input=input_items,
            stream=stream,
            **call_kwargs,
        )
