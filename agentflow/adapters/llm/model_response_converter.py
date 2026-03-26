from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from agentflow.state.message import Message

from .base_converter import BaseConverter


logger = logging.getLogger("agentflow.adapters")


class ModelResponseConverter:
    """Wrap an LLM SDK call and normalize its output via a converter.

    Supports sync/async invocation and streaming. Use `invoke()` for
    non-streaming calls and `stream()` for streaming calls.
    """

    def __init__(
        self,
        response: Any | Callable[..., Any],
        converter: BaseConverter | str,
    ) -> None:
        """
        Initialize ModelResponseConverter.

        Args:
            response (Any | Callable[..., Any]): The LLM response or a callable returning
                a response.
            converter (BaseConverter | str): Converter instance or string identifier
                (e.g., "openai").

        Raises:
            ValueError: If the converter is not supported.
        """
        self.response = response

        if isinstance(converter, str):
            if converter == "openai":
                from .openai_converter import OpenAIConverter

                self.converter = OpenAIConverter()
                logger.debug("Using OpenAIConverter for response conversion")

            elif converter == "openai_responses":
                from .openai_responses_converter import OpenAIResponsesConverter

                self.converter = OpenAIResponsesConverter()
                logger.debug("Using OpenAIResponsesConverter for response conversion")

            elif converter == "google":
                from .google_genai_converter import GoogleGenAIConverter

                self.converter = GoogleGenAIConverter()
                logger.debug("Using GoogleGenAIConverter for response conversion")

            else:
                logger.error(f"Unsupported converter: {converter}")
                raise ValueError(
                    f"Unsupported converter: {converter}. "
                    "Supported: 'openai', 'openai_responses', 'google'"
                )

        elif isinstance(converter, BaseConverter):
            self.converter = converter
            logger.debug(f"Using custom converter: {type(converter).__name__}")
        else:
            logger.error(f"Unsupported converter: {converter}")
            raise ValueError(f"Unsupported converter: {converter}")

    async def invoke(self) -> Message:
        """
        Call the underlying function and convert a non-streaming response to Message.

        Returns:
            Message: The normalized message from the LLM response.

        Raises:
            Exception: If the underlying function or converter fails.
        """
        logger.debug("Invoking model response converter")
        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                logger.debug("Detected async callable response")
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        logger.debug("Converting response to Message")
        return await self.converter.convert_response(response)  # type: ignore

    async def stream(
        self,
        config: dict,
        node_name: str,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Call the underlying function and yield normalized streaming events and final Message.

        Args:
            config (dict): Node configuration parameters for streaming.
            node_name (str): Name of the node processing the response.
            meta (dict | None): Optional metadata for conversion.

        Yields:
            Message: Normalized streaming message events from the LLM response.

        Raises:
            ValueError: If config is not provided.
            Exception: If the underlying function or converter fails.
        """
        if not config:
            logger.error("Config must be provided for streaming conversion")
            raise ValueError("Config must be provided for streaming conversion")

        if callable(self.response):
            if inspect.iscoroutinefunction(self.response):
                logger.debug("Detected async callable response for streaming")
                response = await self.response()
            else:
                response = self.response()
        else:
            response = self.response

        async for item in self.converter.convert_streaming_response(  # type: ignore
            config,
            node_name=node_name,
            response=response,
            meta=meta,
        ):
            yield item

        logger.debug("Completed streaming conversion")
