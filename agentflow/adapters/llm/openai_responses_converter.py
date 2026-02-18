"""Converter for OpenAI Responses API output.

The Responses API (``client.responses.create()``) uses a fundamentally
different response schema compared with Chat Completions.  This module
maps that schema into the same normalised ``Message`` objects used by
every other AgentFlow converter.

Key differences from Chat Completions:
- Response items live in ``response.output`` (list), not ``response.choices``
- Each item has a ``type``: ``"message"``, ``"reasoning"``, ``"function_call"``
- Token usage names differ: ``input_tokens`` / ``output_tokens``
- Streaming uses semantic events like ``response.output_text.delta``
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from agentflow.state.message import (
    Message,
    TokenUsages,
    generate_id,
)
from agentflow.state.message_block import (
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)

from .base_converter import BaseConverter
from .reasoning_utils import (
    parse_think_tags,  # noqa: F401
)


logger = logging.getLogger("agentflow.adapters.openai_responses")


# ---------------------------------------------------------------------------
# Helper: detect whether an object is a Responses API response
# ---------------------------------------------------------------------------


def is_responses_api_response(response: Any) -> bool:
    """Return *True* if *response* looks like an OpenAI Responses API object.

    We check for the ``output`` attribute (list of output items) **and** the
    absence of the ``choices`` attribute (which is specific to Chat
    Completions).
    """
    return hasattr(response, "output") and not hasattr(response, "choices")


# ---------------------------------------------------------------------------
# Helper: parse <reasoning>…</reasoning> tags from text
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


class OpenAIResponsesConverter(BaseConverter):
    """Convert OpenAI **Responses API** objects to AgentFlow ``Message``.

    Handles both non-streaming responses (``Response`` objects with an
    ``output`` list) and streaming event iterators.
    """

    # ---- non-streaming -------------------------------------------------

    async def convert_response(self, response: Any) -> Message:
        """Convert a non-streaming Responses API result to ``Message``."""

        # --- token usage --------------------------------------------------
        usages = self._extract_token_usage(response)

        # --- iterate output items -----------------------------------------
        blocks: list = []
        reasoning_text = ""
        tool_calls_raw: list[dict] = []

        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                r = self._extract_reasoning_from_item(item)
                # Create a ReasoningBlock even when summary is empty —
                # the API still consumed reasoning tokens and the item
                # signals that reasoning occurred.
                summary = r if r else "(reasoning performed - enable summary to see details)"
                blocks.append(ReasoningBlock(summary=summary))
                reasoning_text += ("\n" + summary) if reasoning_text else summary

            elif item_type == "message":
                t = self._extract_text_from_message_item(item)
                if t:
                    blocks.append(TextBlock(text=t))

            elif item_type == "function_call":
                tb, raw = self._extract_tool_call_from_item(item)
                if tb:
                    blocks.append(tb)
                    tool_calls_raw.append(raw)

        # --- build Message ------------------------------------------------
        resp_id = getattr(response, "id", None)
        model = getattr(response, "model", "unknown")
        status = getattr(response, "status", "completed")
        created = getattr(response, "created_at", None) or datetime.now().timestamp()

        return Message(
            message_id=generate_id(resp_id),
            role="assistant",
            content=blocks,
            reasoning=reasoning_text,
            timestamp=created,
            metadata={
                "provider": "openai",
                "model": model,
                "finish_reason": status,
            },
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=tool_calls_raw if tool_calls_raw else None,
        )

    # ---- streaming ------------------------------------------------------

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Convert a Responses API stream to ``Message`` chunks.

        Yields intermediate ``Message(delta=True)`` per event and a final
        ``Message(delta=False)`` when the stream completes.
        """
        # If it's not iterable at all, try the non-streaming path
        if not hasattr(response, "__aiter__") and not hasattr(response, "__iter__"):
            if is_responses_api_response(response):
                yield await self.convert_response(response)
                return
            raise TypeError("Unsupported response type for OpenAIResponsesConverter")

        async for msg in self._handle_stream(config, node_name, response, meta):
            yield msg

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Internal stream handler."""

        accumulated_text = ""
        accumulated_reasoning = ""
        tool_calls: list[dict] = []

        # Track current function call being built
        current_fc_name = ""
        current_fc_args = ""
        current_fc_call_id = ""

        is_awaitable = inspect.isawaitable(stream)
        if is_awaitable:
            stream = await stream

        # --- async iteration ---
        try:
            async for event in stream:
                event_type = getattr(event, "type", "")

                # Text delta
                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    accumulated_text += delta_text
                    yield Message(
                        message_id=generate_id(None),
                        role="assistant",
                        content=[TextBlock(text=delta_text)] if delta_text else [],
                        delta=True,
                    )

                # Reasoning summary delta
                elif event_type == "response.reasoning_summary_text.delta":
                    delta_r = getattr(event, "delta", "")
                    accumulated_reasoning += delta_r

                # Function call argument delta
                elif event_type == "response.function_call_arguments.delta":
                    current_fc_args += getattr(event, "delta", "")

                # New output item
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item:
                        itype = getattr(item, "type", "")
                        if itype == "function_call":
                            current_fc_name = getattr(item, "name", "")
                            current_fc_call_id = getattr(item, "call_id", "")
                            current_fc_args = ""

                # Output item done
                elif event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item:
                        itype = getattr(item, "type", "")
                        if itype == "function_call":
                            # Finalise the function call
                            name = getattr(item, "name", current_fc_name)
                            raw_args = getattr(item, "arguments", current_fc_args)
                            call_id = getattr(item, "call_id", current_fc_call_id)
                            try:
                                args_dict = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                args_dict = {}
                            tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": raw_args or "{}",
                                    },
                                }
                            )
                            yield Message(
                                message_id=generate_id(None),
                                role="assistant",
                                content=[ToolCallBlock(name=name, args=args_dict, id=call_id)],
                                delta=True,
                            )
                            # Reset accumulators
                            current_fc_name = ""
                            current_fc_args = ""
                            current_fc_call_id = ""

                        elif itype == "reasoning":
                            r = self._extract_reasoning_from_item(item)
                            if r:
                                accumulated_reasoning += ("\n" + r) if accumulated_reasoning else r

                # Stream completed — we'll build the final message below
                elif event_type == "response.completed":
                    # Extract usage from the completed response if available
                    completed_response = getattr(event, "response", None)
                    if completed_response:
                        usages = self._extract_token_usage(completed_response)
                    else:
                        usages = TokenUsages()
                    break
            else:
                # Stream ended without response.completed event
                usages = TokenUsages()

        except TypeError:
            # Not async-iterable — try sync iteration
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    accumulated_text += delta_text
                    yield Message(
                        message_id=generate_id(None),
                        role="assistant",
                        content=[TextBlock(text=delta_text)] if delta_text else [],
                        delta=True,
                    )
                elif event_type == "response.reasoning_summary_text.delta":
                    accumulated_reasoning += getattr(event, "delta", "")
                elif event_type == "response.function_call_arguments.delta":
                    current_fc_args += getattr(event, "delta", "")
                elif event_type == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        current_fc_name = getattr(item, "name", "")
                        current_fc_call_id = getattr(item, "call_id", "")
                        current_fc_args = ""
                elif event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "function_call":
                        name = getattr(item, "name", current_fc_name)
                        raw_args = getattr(item, "arguments", current_fc_args)
                        call_id = getattr(item, "call_id", current_fc_call_id)
                        try:
                            args_dict = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            args_dict = {}
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": raw_args or "{}"},
                            }
                        )
                        yield Message(
                            message_id=generate_id(None),
                            role="assistant",
                            content=[ToolCallBlock(name=name, args=args_dict, id=call_id)],
                            delta=True,
                        )
                        current_fc_name = ""
                        current_fc_args = ""
                        current_fc_call_id = ""
                    elif item and getattr(item, "type", "") == "reasoning":
                        r = self._extract_reasoning_from_item(item)
                        if r:
                            accumulated_reasoning += ("\n" + r) if accumulated_reasoning else r
                elif event_type == "response.completed":
                    completed_response = getattr(event, "response", None)
                    if completed_response:
                        usages = self._extract_token_usage(completed_response)
                    else:
                        usages = TokenUsages()
                    break
            else:
                usages = TokenUsages()

        # --- final message -------------------------------------------------
        metadata = meta or {}
        metadata["provider"] = "openai"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        final_blocks: list = []
        if accumulated_text:
            final_blocks.append(TextBlock(text=accumulated_text))
        if accumulated_reasoning:
            final_blocks.append(ReasoningBlock(summary=accumulated_reasoning))
        for tc in tool_calls:
            fd = tc.get("function", {})
            try:
                args = json.loads(fd.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            final_blocks.append(
                ToolCallBlock(name=fd.get("name", ""), args=args, id=tc.get("id", ""))
            )

        yield Message(
            message_id=generate_id(None),
            role="assistant",
            content=final_blocks,
            reasoning=accumulated_reasoning,
            delta=False,
            tools_calls=tool_calls if tool_calls else None,
            metadata=metadata,
            usages=usages,
        )

    # ---- private helpers ------------------------------------------------

    @staticmethod
    def _extract_reasoning_from_item(item: Any) -> str:
        """Pull reasoning summary text out of a ``type='reasoning'`` item."""
        summary_list = getattr(item, "summary", None) or []
        parts: list[str] = []
        for entry in summary_list:
            # Each entry may be an object with .text or a dict with "text"
            if hasattr(entry, "text"):
                parts.append(entry.text)
            elif isinstance(entry, dict):
                parts.append(entry.get("text", ""))
        return "\n".join(parts)

    @staticmethod
    def _extract_text_from_message_item(item: Any) -> str:
        """Pull text out of a ``type='message'`` item."""
        content_list = getattr(item, "content", []) or []
        parts: list[str] = []
        for entry in content_list:
            entry_type = getattr(entry, "type", None)
            if entry_type == "output_text":
                parts.append(getattr(entry, "text", ""))
            elif isinstance(entry, dict) and entry.get("type") == "output_text":
                parts.append(entry.get("text", ""))
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_tool_call_from_item(item: Any) -> tuple[ToolCallBlock | None, dict]:
        """Extract a ``ToolCallBlock`` from a ``type='function_call'`` item."""
        name = getattr(item, "name", "")
        raw_args = getattr(item, "arguments", "")
        call_id = getattr(item, "call_id", "")

        try:
            args_dict = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args_dict = {}

        block = ToolCallBlock(name=name, args=args_dict, id=call_id)

        raw = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": raw_args or "{}",
            },
        }
        return block, raw

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsages:
        """Map Responses API usage fields to ``TokenUsages``."""
        usage = getattr(response, "usage", None)
        if not usage:
            return TokenUsages(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens

        # Reasoning tokens
        output_details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) if output_details else 0

        # Cache tokens
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", 0) if input_details else 0

        return TokenUsages(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens or 0,
            cache_read_input_tokens=cached_tokens or 0,
        )
