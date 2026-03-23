"""
Converter for Anthropic SDK responses to agentflow Message format.

This module provides conversion utilities for Anthropic's SDK,
supporting both standard and streaming responses for Claude models.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, cast

from agentflow.state.message import (
    Message,
    TokenUsages,
    generate_id,
)
from agentflow.state.message_block import (
    ImageBlock,
    MediaRef,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)

from .base_converter import BaseConverter


logger = logging.getLogger("agentflow.adapters.anthropic")


try:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import MessageStreamEvent

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AnthropicMessage = None  # type: ignore
    MessageStreamEvent = None  # type: ignore


class AnthropicConverter(BaseConverter):
    """
    Converter for Anthropic responses to agentflow Message format.

    Handles both standard and streaming responses, extracting content, reasoning,
    tool calls, and token usage details from Anthropic's Message format.

    Supports:
    - Message responses (Claude models)
    - Streaming MessageStreamEvent responses
    - Text content blocks
    - Image content blocks
    - Tool use blocks
    - Extended thinking (reasoning) support
    """

    async def convert_response(self, response: AnthropicMessage) -> Message:  # type: ignore[reportInvalidTypeForm]
        """
        Convert an Anthropic Message to agentflow Message.

        Args:
            response (AnthropicMessage): The Anthropic Message response object.

        Returns:
            Message: The converted message object.

        Raises:
            ImportError: If anthropic SDK is not installed.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK is required for Anthropic converter. "
                "Install it with: pip install anthropic"
            )

        # Extract usage information
        usage = response.usage
        usages = TokenUsages(
            completion_tokens=usage.output_tokens if usage else 0,
            prompt_tokens=usage.input_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            reasoning_tokens=0,  # Anthropic doesn't expose this separately yet
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0)
            if usage
            else 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
        )

        # Extract content blocks and tool calls
        blocks, tools_calls, reasoning_content = self._process_content_blocks(response.content)

        logger.debug("Creating message from Anthropic response with id: %s", response.id)

        return Message(
            message_id=generate_id(response.id),
            role=response.role,
            content=blocks,
            reasoning=reasoning_content,
            timestamp=datetime.now().timestamp(),
            metadata={
                "provider": "anthropic",
                "model": response.model,
                "stop_reason": response.stop_reason or "UNKNOWN",
                "stop_sequence": response.stop_sequence,
            },
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=tools_calls if tools_calls else None,
        )

    def _process_content_blocks(self, content_blocks: list) -> tuple[list, list, str]:
        """
        Process Anthropic content blocks to extract text, images, tool calls, etc.

        Args:
            content_blocks: List of Anthropic content block objects.

        Returns:
            tuple: (blocks, tools_calls, reasoning_content)
        """
        blocks = []
        tools_calls = []
        reasoning_content = ""

        for block in content_blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    blocks.append(TextBlock(text=text))

            elif block_type == "thinking":
                # Extended thinking block (Claude with thinking enabled)
                thinking = getattr(block, "thinking", "")
                if thinking:
                    blocks.append(ReasoningBlock(summary=thinking))
                    reasoning_content = thinking

            elif block_type == "image":
                source = getattr(block, "source", None)
                if source:
                    media = self._extract_image_media(source)
                    if media:
                        blocks.append(ImageBlock(media=media))

            elif block_type == "tool_use":
                tool_use_id = getattr(block, "id", generate_id(None))
                tool_name = getattr(block, "name", "")
                tool_input = getattr(block, "input", {})

                blocks.append(
                    ToolCallBlock(
                        name=tool_name,
                        args=tool_input,
                        id=tool_use_id,
                    )
                )

                tools_calls.append(
                    {
                        "id": tool_use_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        },
                    }
                )

        return blocks, tools_calls, reasoning_content

    def _extract_image_media(self, source: Any) -> MediaRef | None:
        """
        Extract MediaRef from an Anthropic image source.

        Args:
            source: Image source object from Anthropic content block.

        Returns:
            MediaRef or None if extraction fails.
        """
        try:
            source_type = getattr(source, "type", None)

            if source_type == "base64":
                data = getattr(source, "data", None)
                media_type = getattr(source, "media_type", "image/png")
                if data:
                    return MediaRef(
                        kind="data",
                        data_base64=data,
                        mime_type=media_type,
                    )

            elif source_type == "url":
                url = getattr(source, "url", None)
                if url:
                    return MediaRef(
                        kind="url",
                        url=url,
                    )

        except Exception as e:
            logger.warning("Failed to extract image media: %s", e)

        return None

    def _extract_delta_content_blocks(
        self,
        delta: Any,
    ) -> tuple[str, str, list]:
        """
        Extract content blocks from a streaming delta.

        Args:
            delta: Delta object from a streaming event.

        Returns:
            tuple: (text_part, reasoning_part, content_blocks)
        """
        text_part = ""
        reasoning_part = ""
        content_blocks = []

        delta_type = getattr(delta, "type", None)

        if delta_type == "text_delta":
            text = getattr(delta, "text", "")
            if text:
                text_part = text
                content_blocks.append(TextBlock(text=text))

        elif delta_type == "thinking_delta":
            thinking = getattr(delta, "thinking", "")
            if thinking:
                reasoning_part = thinking
                content_blocks.append(ReasoningBlock(summary=thinking))

        elif delta_type == "input_json_delta":
            # Partial tool JSON — complete tool_use arrives in the final message
            pass

        return text_part, reasoning_part, content_blocks

    def _process_stream_event(
        self,
        event: Any,
        accumulated_content: str,
        accumulated_reasoning: str,
        tool_calls: list,
        tool_ids: set,
    ) -> tuple[str, str, list, Message | None]:
        """
        Process a single streaming event from Anthropic.

        Args:
            event: The streaming event object.
            accumulated_content: Accumulated text content so far.
            accumulated_reasoning: Accumulated reasoning content so far.
            tool_calls: List of tool calls detected so far.
            tool_ids: Set of tool call IDs to avoid duplicates.

        Returns:
            tuple: (accumulated_content, accumulated_reasoning, tool_calls, Message | None)
        """
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta:
                text_part, reasoning_part, content_blocks = self._extract_delta_content_blocks(
                    delta
                )
                accumulated_content += text_part
                accumulated_reasoning += reasoning_part

                if content_blocks:
                    output_message = Message(
                        message_id=generate_id(None),
                        role="assistant",
                        content=content_blocks,
                        reasoning=accumulated_reasoning,
                        delta=True,
                    )
                    return (
                        accumulated_content,
                        accumulated_reasoning,
                        tool_calls,
                        output_message,
                    )

        elif event_type == "content_block_start":
            content_block = getattr(event, "content_block", None)
            if content_block:
                block_type = getattr(content_block, "type", None)
                if block_type == "tool_use":
                    tool_use_id = getattr(content_block, "id", None)
                    if tool_use_id and tool_use_id not in tool_ids:
                        tool_ids.add(tool_use_id)

        # content_block_stop and other events don't produce delta messages
        return accumulated_content, accumulated_reasoning, tool_calls, None

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Handle an Anthropic streaming response and yield Message objects.

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            stream: The Anthropic streaming response object.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message events from the stream.
        """
        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls: list = []
        tool_ids: set = set()

        is_awaitable = inspect.isawaitable(stream)
        if is_awaitable:
            stream = await stream

        # Try async iteration first
        try:
            async for event in stream:  # type: ignore
                accumulated_content, accumulated_reasoning, tool_calls, message = (
                    self._process_stream_event(
                        event,
                        accumulated_content,
                        accumulated_reasoning,
                        tool_calls,
                        tool_ids,
                    )
                )
                if message:
                    yield message
        except Exception:  # noqa: S110 # nosec B110
            pass

        # Fall back to sync iteration
        try:
            for event in stream:
                accumulated_content, accumulated_reasoning, tool_calls, message = (
                    self._process_stream_event(
                        event,
                        accumulated_content,
                        accumulated_reasoning,
                        tool_calls,
                        tool_ids,
                    )
                )
                if message:
                    yield message
        except Exception:  # noqa: S110 # nosec B110
            pass

        # Yield final aggregated message
        metadata = meta or {}
        metadata["provider"] = "anthropic"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        blocks = []
        if accumulated_content:
            blocks.append(TextBlock(text=accumulated_content))
        if accumulated_reasoning:
            blocks.append(ReasoningBlock(summary=accumulated_reasoning))
        if tool_calls:
            for tc in tool_calls:
                func_data = tc.get("function", {})
                args_str = func_data.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                blocks.append(
                    ToolCallBlock(
                        name=func_data.get("name", ""),
                        args=args,
                        id=tc.get("id", ""),
                    )
                )

        logger.debug(
            "Stream done - Content: %s, Reasoning: %s, Tool Calls: %s",
            accumulated_content,
            accumulated_reasoning,
            len(tool_calls),
        )

        yield Message(
            role="assistant",
            message_id=generate_id(None),
            content=blocks,
            delta=False,
            reasoning=accumulated_reasoning,
            tools_calls=tool_calls if tool_calls else None,
            metadata=metadata,
        )

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Convert an Anthropic streaming or standard response to Message(s).

        Args:
            config (dict): Node configuration parameters.
            node_name (str): Name of the node processing the response.
            response (Any): The Anthropic response object (stream or standard Message).
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Converted message(s) from the response.

        Raises:
            ImportError: If anthropic SDK is not installed.
            Exception: If response type is unsupported.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK is required for Anthropic converter. "
                "Install it with: pip install anthropic"
            )

        if hasattr(response, "__aiter__") or hasattr(response, "__iter__"):
            async for event in self._handle_stream(
                config or {},
                node_name or "",
                response,
                meta,
            ):
                yield event
        elif isinstance(response, AnthropicMessage):  # type: ignore
            message = await self.convert_response(cast(AnthropicMessage, response))  # type: ignore
            yield message
        else:
            raise Exception("Unsupported response type for AnthropicConverter")
