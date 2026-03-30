"""Public Agent facade for graph-based LLM interactions.

The public import path remains ``agentflow.graph.agent.Agent`` while the
implementation lives in smaller internal modules under ``agentflow.graph.agent_internal``.
"""

import logging
from collections.abc import Callable
from typing import Any

from agentflow.graph.base_agent import BaseAgent
from agentflow.graph.tool_node import ToolNode
from agentflow.skills.models import SkillConfig
from agentflow.state.message import Message

from .agent_internal.constants import DEFAULT_RETRY_CONFIG, REASONING_DEFAULT, RetryConfig
from .agent_internal.execution import AgentExecutionMixin
from .agent_internal.google import AgentGoogleMixin
from .agent_internal.openai import AgentOpenAIMixin
from .agent_internal.providers import AgentProviderMixin
from .agent_internal.skills import AgentSkillsMixin


logger = logging.getLogger("agentflow.agent")


class Agent(
    AgentExecutionMixin,
    AgentGoogleMixin,
    AgentOpenAIMixin,
    AgentProviderMixin,
    AgentSkillsMixin,
    BaseAgent,
):
    """A smart node function wrapper for LLM interactions.

    This class handles common boilerplate for agent implementations including:
    - Automatic message conversion
    - LLM calls via native provider SDKs (OpenAI, Google)
    - Tool handling with conditional logic
    - Optional learning/RAG capabilities
    - Response conversion

    The Agent is designed to be used as a node within a StateGraph, providing
    a high-level interface while maintaining full graph flexibility.

    Example:
        ```python
        # Create an agent node with OpenAI
        agent = Agent(
            model="gpt-4o",
            provider="openai",
            system_prompt="You are a helpful assistant",
            tools=[weather_tool],
        )

        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)  # Agent acts as a node function
        graph.add_node("TOOL", agent.get_tool_node())
        # ... setup edges
        ```

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "gemini-2.0-flash")
        provider: Provider name ("openai", "google")
        system_prompt: System prompt string or list of message dicts
        tools: List of tool functions or ToolNode instance
        client: Optional custom client instance (escape hatch for power users)
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens to generate
        llm_kwargs: Additional provider-specific parameters
    """

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        output_type: str = "text",  # NEW: Explicit output type
        system_prompt: list[dict[str, Any]] | None = None,
        tools: list[Callable] | ToolNode | None = None,
        tool_node_name: str | None = None,
        extra_messages: list[Message] | None = None,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        api_style: str = "chat",
        reasoning_config: dict[str, Any] | bool | None = REASONING_DEFAULT,  # type: ignore
        skills: "SkillConfig | None" = None,
        retry_config: RetryConfig | bool | None = True,
        fallback_models: list[str | tuple[str, str]] | None = None,
        **kwargs,
    ):
        """Initialize an Agent node.

        Args:
            model: Model identifier (any model name - no parsing required).
                Examples: "gpt-4o", "gemini-2.0-flash-exp", "qwen-2.5-72b", "deepseek-chat"
            provider: Provider name ("openai", "google"). If None, will auto-detect from model.
            output_type: Type of output to generate (default: "text").
                - "text": Text generation (default, most common)
                - "image": Image generation
                - "video": Video generation
                - "audio": Audio/TTS generation
            system_prompt: System prompt as list of message dicts.
                Supports state variable interpolation using placeholders like ``{field_name}``.
                At execution time, placeholders are replaced with actual values from the state.

                Example with interpolation::

                    class MyState(AgentState):
                        user_name: str = "Guest"
                        occasion: str = "casual"

                    agent = Agent(
                        model="gpt-4o",
                        system_prompt=[{
                            "role": "system",
                            "content": "You are helping {user_name} with {occasion} planning."
                        }]
                    )
                    # At runtime, placeholders are replaced with state values
            tools: List of tool functions, ToolNode instance, or None.
                If list is provided, will be converted to ToolNode internally.
            tool_node_name: Name of the existing ToolNode. You can send list of tools
                or provide ToolNode instance via `tools` parameter instead.
            extra_messages: Additional messages to include in every interaction.
            trim_context: Whether to trim context using context manager.
            tools_tags: Optional tags to filter tools.
            base_url (via **kwargs): Optional base URL for OpenAI-compatible APIs
                (ollama, vllm, openrouter, deepseek, etc.). Default: ``None``.
            api_style (via **kwargs): API style for OpenAI provider. ``"chat"`` uses
                Chat Completions, ``"responses"`` uses the Responses API.
                Default: ``"chat"``.
            reasoning_config: Unified reasoning control for all providers. Default
                is ``{"effort": "medium"}`` (on). Pass ``None`` to turn off.
                ``effort`` applies to both providers; ``summary`` is OpenAI-only;
                ``thinking_budget`` is Google-only and overrides ``effort``.

                For Google, ``effort`` is translated to ``thinking_budget`` automatically:
                ``"low"`` → 512, ``"medium"`` → 8192 (default), ``"high"`` → 24576.
                So thinking is **on by default** for Google with ``thinking_budget=8192``.

                Examples::

                    reasoning_config=None                        # OFF for both
                    reasoning_config={"effort": "high"}          # high, both providers
                    reasoning_config={"effort": "low", "summary": "auto"}  # OpenAI: low+summary
                    reasoning_config={"thinking_budget": 5000}   # Google exact budget
            retry_config: Controls automatic retry with exponential back-off for
                transient LLM errors (429, 500, 502, 503, 529).  Default is
                ``True`` which uses ``RetryConfig()`` (3 retries, 1 s initial
                delay, 2x backoff, 30 s cap).  Pass ``False`` or ``None`` to
                disable.  Pass a ``RetryConfig`` instance for fine-grained
                control::

                    retry_config = RetryConfig(max_retries=5, initial_delay=2.0)
                    retry_config = False  # disable retries entirely

            fallback_models: Ordered list of fallback models to try when the
                primary model exhausts all retries.  Each entry is either a
                plain model string (inherits the agent's provider) or a
                ``(model, provider)`` tuple for cross-provider fallback::

                    fallback_models = ["gpt-4o-mini"]
                    fallback_models = [("gemini-2.0-flash", "google")]
                    fallback_models = ["gpt-4o-mini", ("gemini-2.0-flash", "google")]

            **llm_kwargs: Additional provider-specific parameters
                (temperature, max_tokens, top_p, or model args, organization_id, project_id).

        Raises:
            ImportError: If required provider SDK is not installed.
            ValueError: If provider cannot be determined or doesn't support output_type.

        Example:
            ```python
            # Text generation (default - no need to specify output_type)
            text_agent = Agent(
                model="openai/gpt-4o",
                system_prompt="You are a helpful assistant",
                tools=[weather_tool, calculator],
                temperature=0.8,
            )

            # Image generation (explicit)
            image_agent = Agent(
                model="openai/dall-e-3",
                output_type="image",
            )

            # Video generation (explicit)
            video_agent = Agent(
                model="google/veo-2.0",
                provider="google",
                output_type="video",
            )

            # Multi-modal workflow (Google ADK style)
            prompt_agent = Agent(
                model="google/gemini-2.0-flash-exp",
                system_prompt="Generate detailed image prompts",
            )

            imagen_agent = Agent(
                model="google/imagen-3.0-generate-001",
                output_type="image",
            )

            # Third-party models (Qwen, DeepSeek, Ollama)
            qwen_agent = Agent(
                model="qwen-2.5-72b-instruct",
                provider="openai",
                base_url="https://api.qwen.com/v1",
            )

            ollama_agent = Agent(
                model="llama3:70b",
                provider="openai",
                base_url="http://localhost:11434/v1",
            )

            # With retry and fallback
            resilient_agent = Agent(
                model="gemini-2.5-flash",
                provider="google",
                retry_config=RetryConfig(max_retries=5, initial_delay=2.0),
                fallback_models=[
                    "gemini-2.0-flash",  # same provider
                    ("gpt-4o-mini", "openai"),  # cross-provider fallback
                ],
            )
            ```
        """
        # Pop kwargs-only params before passing to parent
        base_url: str | None = kwargs.pop("base_url", None)
        # Note: api_style is already a named parameter, don't pop from kwargs
        # Call parent constructor
        super().__init__(
            model=model, system_prompt=system_prompt or [], tools=tools, base_url=base_url, **kwargs
        )

        # check user sending model and provider as prefix, if provider is not explicitly provided
        if "/" in model and provider is None:
            provider, model = model.split("/", 1)
            self.model = model

        # Store output type
        self.output_type = output_type.lower()

        # Determine provider; self.llm_kwargs is set by super().__init__ and is
        # already available here for _create_client().
        if provider is not None:
            self.provider = provider.lower()
            self.base_url = base_url
            self.client = self._create_client(self.provider, base_url)
        else:
            # Auto-detect provider from model name
            self.provider = self._detect_provider_from_model(model)
            self.base_url = base_url
            self.client = self._create_client(self.provider, base_url)

        # Validate that provider supports the output type
        self._validate_output_type()

        self.extra_messages = extra_messages
        self.trim_context = trim_context
        self.tools_tags = tools_tags
        self.tool_node_name = tool_node_name

        # Internal setup
        self._tool_node = self._setup_tools()

        # API style & reasoning configuration
        if api_style not in ("chat", "responses"):
            raise ValueError(f"Invalid api_style '{api_style}'. Supported: 'chat', 'responses'")
        self.api_style = api_style

        # Apply default (medium effort) when not explicitly provided;
        # True or sentinel = enable with defaults; False or None = disabled.
        if reasoning_config is REASONING_DEFAULT or reasoning_config is True:
            reasoning_config = {"effort": "medium"}
        self.reasoning_config: dict[str, Any] | None = (
            None if (reasoning_config is False or reasoning_config is None) else reasoning_config
        )  # type: ignore

        # Retry & fallback configuration
        if retry_config is True:
            self.retry_config: RetryConfig | None = DEFAULT_RETRY_CONFIG
        elif isinstance(retry_config, RetryConfig):
            self.retry_config = retry_config
        else:
            self.retry_config = None

        # Normalise fallback_models to list[tuple[str, str | None]]
        self.fallback_models: list[tuple[str, str | None]] = []
        if fallback_models:
            for entry in fallback_models:
                if isinstance(entry, str):
                    self.fallback_models.append((entry, None))  # inherit provider
                else:
                    self.fallback_models.append(tuple(entry))  # type: ignore[arg-type]

        logger.info(
            f"Agent initialized: model={model}, provider={self.provider}, "
            f"output_type={self.output_type}, has_tools={self._tool_node is not None}"
        )

        # Skills setup (via mixin)
        self._setup_skills(skills)
