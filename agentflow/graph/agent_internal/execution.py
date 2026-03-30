"""Execution helpers for Agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from injectq import Inject, InjectQ
from injectq.utils.exceptions import DependencyNotFoundError

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph.tool_node import ToolNode
from agentflow.state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.utils.converter import convert_messages

from .constants import RetryConfig


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

    def get_tool_node(self) -> ToolNode | None:
        """Return the agent's internal ToolNode.

        Use this public method instead of accessing ``agent._tool_node``
        directly when wiring the tool node into the graph. When skills are
        enabled, the returned ToolNode already contains the ``set_skill`` tool.

        Example::

            agent = Agent(model="gpt-4o", tools=[my_tool], skills=SkillConfig(...))
            graph.add_node("TOOL", agent.get_tool_node())
        """
        return self._tool_node

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

    # ------------------------------------------------------------------
    # Retry / fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """Best-effort extraction of an HTTP status code from an SDK exception."""
        # OpenAI SDK: openai.APIStatusError has .status_code
        status = getattr(exc, "status_code", None)
        if status is not None:
            return int(status)
        # Google GenAI and generic HTTP errors often embed a code attribute
        code = getattr(exc, "code", None)
        if code is not None:
            try:
                return int(code)
            except (TypeError, ValueError):
                pass
        # Fallback: inspect the string representation for common patterns
        exc_str = str(exc)
        for code in (503, 502, 500, 429, 529):
            if str(code) in exc_str:
                return code
        return None

    def _is_retryable_error(self, exc: Exception, retry_cfg: RetryConfig) -> bool:
        """Determine whether *exc* is a transient error worth retrying."""
        status = self._extract_status_code(exc)
        if status is not None and status in retry_cfg.retryable_status_codes:
            return True
        # Connection-level / transport errors are always retryable
        if isinstance(exc, ConnectionError | TimeoutError | OSError):
            return True
        exc_name = type(exc).__name__.lower()
        return any(
            keyword in exc_name
            for keyword in ("timeout", "connection", "unavailable", "serviceunav")
        )

    async def _call_llm_with_retry(  # noqa: PLR0912
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Wrap ``_call_llm`` with retry + exponential back-off + fallback models.

        Execution order:
        1. Try the primary model up to ``retry_config.max_retries`` times.
        2. For each fallback model, try up to ``retry_config.max_retries`` times.
        3. If everything fails, raise the last exception.
        """
        retry_cfg: RetryConfig | None = getattr(self, "retry_config", None)
        fallback_models: list[tuple[str, str | None]] = getattr(self, "fallback_models", [])

        # Fast-path: no retry config at all → single attempt, no catch
        if retry_cfg is None and not fallback_models:
            return await self._call_llm(messages, tools, stream, **kwargs)

        max_retries = retry_cfg.max_retries if retry_cfg else 0

        # Build the ordered attempt list: primary + fallbacks
        attempts: list[tuple[str, str, Any, str | None]] = [
            (self.model, self.provider, self.client, getattr(self, "base_url", None)),
        ]
        for fb_model, fb_provider in fallback_models:
            attempts.append((fb_model, fb_provider or self.provider, None, None))

        last_exc: Exception | None = None

        for attempt_idx, (model, provider, fallback_client, base_url) in enumerate(attempts):
            is_fallback = attempt_idx > 0

            if is_fallback:
                logger.info(
                    "Switching to fallback model %s (provider=%s)",
                    model,
                    provider,
                )

            for retry in range(max_retries + 1):  # 0 .. max_retries
                try:
                    if is_fallback:
                        # Temporarily swap model/provider/client for the call
                        orig_model, orig_provider, orig_client, orig_base_url = (
                            self.model,
                            self.provider,
                            self.client,
                            getattr(self, "base_url", None),
                        )
                        self.model = model
                        self.provider = provider
                        self.base_url = base_url
                        active_client = fallback_client
                        if active_client is None:
                            active_client = self._create_client(provider, base_url)
                        self.client = active_client
                        try:
                            result = await self._call_llm(messages, tools, stream, **kwargs)
                        finally:
                            # Restore originals regardless of outcome
                            self.model = orig_model
                            self.provider = orig_provider
                            self.client = orig_client
                            self.base_url = orig_base_url
                    else:
                        result = await self._call_llm(messages, tools, stream, **kwargs)

                    if is_fallback or retry > 0:
                        logger.info(
                            "LLM call succeeded on %s (attempt %d/%d, model_index=%d)",
                            model,
                            retry + 1,
                            max_retries + 1,
                            attempt_idx,
                        )
                    return result

                except Exception as exc:
                    last_exc = exc

                    if retry_cfg is None or not self._is_retryable_error(exc, retry_cfg):
                        logger.warning(
                            "Non-retryable error from %s: %s",
                            model,
                            exc,
                        )
                        # Non-retryable → skip remaining retries, try next fallback
                        break

                    if retry < max_retries:
                        delay = min(
                            retry_cfg.initial_delay * (retry_cfg.backoff_factor**retry),
                            retry_cfg.max_delay,
                        )
                        logger.warning(
                            "Retryable error from %s (attempt %d/%d): %s. Retrying in %.1fs …",
                            model,
                            retry + 1,
                            max_retries + 1,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            "All %d retries exhausted for model %s.",
                            max_retries + 1,
                            model,
                        )

        # Every model exhausted → re-raise the last exception
        assert last_exc is not None  # noqa: S101
        raise last_exc

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

        # Build effective system prompts (with trigger table if skills configured)
        effective_system_prompt = list(self.system_prompt)

        if hasattr(self, "_build_skill_prompts") and callable(self._build_skill_prompts):
            effective_system_prompt = self._build_skill_prompts(state, self.system_prompt)

        messages = convert_messages(
            system_prompts=effective_system_prompt,
            state=state,
            extra_messages=self.extra_messages or [],
        )
        is_stream = config.get("is_stream", False)

        if state.context and state.context[-1].role == "tool":
            response = await self._call_llm_with_retry(messages=messages, stream=is_stream)
        else:
            tools = await self._resolve_tools(container)
            response = await self._call_llm_with_retry(
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
