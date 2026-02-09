"""Agent class for simplified LLM interactions in TAF graphs.

This module provides a high-level Agent wrapper that acts as a smart node function,
handling common boilerplate for LLM interactions including message conversion,
tool handling, and optional learning/RAG capabilities.

The Agent class is designed to be used as a node within a StateGraph, not as
a replacement for the graph itself.
"""

import logging
from collections.abc import Callable
from typing import Any

from injectq import Inject, InjectQ

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph.base_agent import BaseAgent
from agentflow.graph.tool_node import ToolNode
from agentflow.state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.state.message import Message

# from agentflow.store.base_store import BaseStore
from agentflow.utils.converter import convert_messages


logger = logging.getLogger("agentflow.agent")

# Constants
CONTENT_PREVIEW_LENGTH = 200


class Agent(BaseAgent):
    """A smart node function wrapper for LLM interactions.

    This class handles common boilerplate for agent implementations including:
    - Automatic message conversion
    - LLM calls via native provider SDKs (OpenAI, Anthropic, Google)
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

        # Or with Anthropic
        agent = Agent(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            system_prompt="You are a helpful assistant",
        )

        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)  # Agent acts as a node function
        graph.add_node("TOOL", agent.get_tool_node())
        # ... setup edges
        ```

    Attributes:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        provider: Provider name ("openai", "anthropic", "google")
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
        base_url: str | None = None,  # For OpenAI-compatible APIs (ollama, vllm, etc.)
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
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
            tools: List of tool functions, ToolNode instance, or None.
                If list is provided, will be converted to ToolNode internally.
            tool_node_name: Name of the existing ToolNode. You can send list of tools
                or provide ToolNode instance via `tools` parameter instead.
            extra_messages: Additional messages to include in every interaction.
            base_url: Optional base URL for OpenAI-compatible APIs
                (ollama, vllm, openrouter, deepseek, etc.).
            trim_context: Whether to trim context using context manager.
            tools_tags: Optional tags to filter tools.
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
            ```
        """
        # Call parent constructor
        super().__init__(model=model, system_prompt=system_prompt or [], tools=tools, **kwargs)

        # check user sending model and provider as prefix, if provider is not explicitly provided
        if "/" in model:
            provider, model = model.split("/", 1)

        # Store output type
        self.output_type = output_type.lower()

        # Determine provider
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
        self.tools = tools
        self.llm_kwargs = kwargs
        self.trim_context = trim_context
        self.tools_tags = tools_tags
        self.tool_node_name = tool_node_name

        # Internal setup
        self._tool_node = self._setup_tools()

        logger.info(
            f"Agent initialized: model={model}, provider={self.provider}, "
            f"output_type={self.output_type}, has_tools={self._tool_node is not None}"
        )

    def _setup_tools(self) -> ToolNode | None:
        """Convert tools to ToolNode if needed.

        Returns:
            ToolNode instance or None if no tools provided.
        """
        if self.tools is None:
            logger.debug("No tools provided")
            return None

        if isinstance(self.tools, ToolNode):
            logger.debug("Tools already a ToolNode instance")
            return self.tools

        logger.debug(f"Converting {len(self.tools)} tool functions to ToolNode")
        return ToolNode(self.tools)

    def _validate_output_type(self) -> None:
        """Validate that provider supports the requested output type.

        Raises:
            ValueError: If provider doesn't support the output type.
        """
        valid_output_types = ["text", "image", "video", "audio"]
        if self.output_type not in valid_output_types:
            raise ValueError(
                f"Invalid output_type '{self.output_type}'. Supported types: {valid_output_types}"
            )

        # Provider-specific validation
        if self.provider == "google":
            supported = ["text", "image", "video", "audio"]
            if self.output_type not in supported:
                raise ValueError(
                    f"Google provider doesn't support output_type='{self.output_type}'. "
                    f"Supported: {supported}"
                )
            logger.debug(f"Google provider supports output_type='{self.output_type}'")

        elif self.provider == "openai":
            supported = ["text", "image", "audio"]
            if self.output_type not in supported:
                raise ValueError(
                    f"OpenAI provider doesn't support output_type='{self.output_type}'. "
                    f"Supported: {supported}"
                )
            logger.debug(f"OpenAI provider supports output_type='{self.output_type}'")

    async def _trim_context(
        self,
        state: AgentState,
        context_manager: BaseContextManager | None = Inject[BaseContextManager],
    ) -> AgentState:
        """Trim context using the provided context manager.

        Args:
            state: Current agent state.
            context_manager: Context manager for trimming.

        Returns:
            Trimmed agent state.
        """
        if self.trim_context:
            if context_manager is None:
                logger.warning("trim_context is enabled but no context manager is available")
                return state

            # Try to access context_manager to check if it's actually None (InjectQ proxy)
            try:
                new_state = await context_manager.trim_context(state)
                logger.debug("Context trimmed using context manager")
                return new_state
            except AttributeError:
                logger.warning(
                    "trim_context is enabled but no BaseContextManager is registered. "
                    "Skipping context trimming."
                )
                return state

        logger.debug("Context trimming not enabled")
        return state

    def _detect_provider_from_model(self, model: str) -> str:
        """Auto-detect provider from model name.

        Args:
            model: Model identifier string

        Returns:
            Provider name ("openai", "google")

        Raises:
            ValueError: If provider cannot be determined.
        """
        model_lower = model.lower()

        if model_lower.startswith(("gpt-", "o1-")):
            return "openai"
        if model_lower.startswith("gemini-"):
            return "google"
        # Default to openai for unknown models (works with ollama, vllm, etc.)
        logger.info(
            f"Could not auto-detect provider for model '{model}'. "
            "Defaulting to 'openai'. If using a different provider, specify explicitly."
        )
        return "openai"

    def _create_client(self, provider: str, base_url: str | None = None) -> Any:
        """Create a client instance for the specified provider.

        Args:
            provider: Provider name ("openai", "google")
            model: Model identifier
            base_url: Optional base URL for OpenAI-compatible APIs

        Returns:
            Client instance

        Raises:
            ImportError: If required SDK is not installed.
        """
        if provider == "openai":
            try:
                import os

                from openai import AsyncOpenAI

                # Get API key from environment
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning(
                        "OPENAI_API_KEY environment variable not set. "
                        "API calls may fail if not using custom client or base_url without auth."
                    )

                if base_url:
                    logger.info(f"Using custom base_url for OpenAI client: {base_url}")
                    return AsyncOpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        **self.llm_kwargs,
                    )

                return AsyncOpenAI(
                    api_key=api_key,
                    **self.llm_kwargs,
                )
            except ImportError:
                raise ImportError(
                    "openai SDK is required for OpenAI provider. "
                    "Install it with: pip install 10xscale-agentflow[openai]"
                )

        elif provider == "google":
            try:
                import os

                from google import genai

                # Get API key from environment
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set "
                        "for Google provider"
                    )

                # Use Client (has both sync and async methods via .aio)
                logger.info("Creating Google GenAI Client with async support")
                return genai.Client(
                    api_key=api_key,
                    **self.llm_kwargs,
                )
            except ImportError:
                raise ImportError(
                    "google-genai SDK is required for Google provider. "
                    "Install it with: pip install 10xscale-agentflow[google-genai]"
                )

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: 'openai', 'google'"
            )

    def _extract_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Extract prompt from messages for non-text generation (image/video/audio).

        Args:
            messages: List of message dicts

        Returns:
            Extracted prompt string (from last user message)
        """
        # Get the last user message content
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return str(content) if content else ""
        return ""

    def _convert_to_google_format(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[str]]:
        """Convert messages to Google GenAI format.

        Args:
            messages: List of message dicts

        Returns:
            Tuple of (system_instruction, google_contents)
        """
        system_instruction = None
        google_contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if system_instruction is None:
                    system_instruction = str(content)
                else:
                    system_instruction += "\n" + str(content)
            else:
                google_contents.append(str(content))

        return system_instruction, google_contents

    def _convert_tools_to_google_format(self, tools: list) -> list:
        """Convert tools to Google GenAI format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Google format uses FunctionDeclaration:
        FunctionDeclaration(
            name="get_weather",
            description="Get weather...",
            parameters_json_schema={  # Note: parameters_json_schema, not parameters
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        )

        Args:
            tools: List of tool dicts in OpenAI format

        Returns:
            List of Google FunctionDeclaration objects
        """
        from google.genai import types

        google_tools = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                func = tool["function"]

                # Create Google FunctionDeclaration
                func_decl_kwargs = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                }

                # Add parameters_json_schema if parameters exist
                if "parameters" in func:
                    func_decl_kwargs["parameters_json_schema"] = func["parameters"]

                google_tools.append(types.FunctionDeclaration(**func_decl_kwargs))
        return google_tools

    def _build_google_config(
        self,
        system_instruction: str | None,
        tools: list | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        """Build Google GenAI config.

        Args:
            system_instruction: System instruction string
            tools: List of tools
            call_kwargs: Additional call kwargs

        Returns:
            GenerateContentConfig instance or None
        """
        from google.genai import types

        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Add LLM kwargs
        if "temperature" in call_kwargs:
            config_kwargs["temperature"] = call_kwargs.pop("temperature")
        if "max_tokens" in call_kwargs or "max_output_tokens" in call_kwargs:
            config_kwargs["max_output_tokens"] = call_kwargs.pop(
                "max_tokens", call_kwargs.pop("max_output_tokens", None)
            )

        # Add tools for text generation only
        if tools and self.output_type == "text":
            function_declarations = self._convert_tools_to_google_format(tools)
            if function_declarations:
                # Wrap FunctionDeclarations in a Tool object
                config_kwargs["tools"] = [types.Tool(function_declarations=function_declarations)]

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call OpenAI API (or OpenAI-compatible APIs).

        Routes to appropriate endpoint based on output_type.

        Args:
            messages: List of message dicts
            tools: Optional list of tool specifications
            stream: Whether to stream the response
            **kwargs: Additional call parameters

        Returns:
            OpenAI response object
        """
        call_kwargs = {**self.llm_kwargs, **kwargs}

        if self.output_type == "text":
            # Standard chat completions (default)
            if tools:
                call_kwargs["tools"] = tools

            logger.debug(f"Calling OpenAI chat.completions.create with model={self.model}")
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **call_kwargs,
            )

        if self.output_type == "image":
            # DALL-E or compatible image generation
            prompt = self._extract_prompt(messages)
            logger.debug(f"Calling OpenAI images.generate with model={self.model}")
            return await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                **call_kwargs,
            )

        if self.output_type == "audio":
            # TTS or audio generation
            text = self._extract_prompt(messages)
            logger.debug(f"Calling OpenAI audio.speech.create with model={self.model}")
            return await self.client.audio.speech.create(
                model=self.model,
                input=text,
                **call_kwargs,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for OpenAI provider")

    async def _call_google(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call Google GenAI API.

        Routes to appropriate method based on output_type.

        Args:
            messages: List of message dicts
            tools: Optional list of tool specifications
            stream: Whether to stream the response
            **kwargs: Additional call parameters

        Returns:
            Google GenAI response object
        """
        call_kwargs = {**self.llm_kwargs, **kwargs}

        # Convert messages to Google format
        system_instruction, google_contents = self._convert_to_google_format(messages)

        # Build config
        config = self._build_google_config(system_instruction, tools, call_kwargs)

        if self.output_type == "text":
            # Text generation - generate_content
            if stream:
                logger.debug(
                    f"Calling Google aio.models.generate_content_stream with model={self.model}"
                )
                return await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=google_contents,
                    config=config,
                )
            logger.debug(f"Calling Google aio.models.generate_content with model={self.model}")
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=google_contents,
                config=config,
            )

        if self.output_type == "image":
            # Image generation - generate_images
            prompt = self._extract_prompt(messages)
            logger.debug(f"Calling Google aio.models.generate_images with model={self.model}")
            return await self.client.aio.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "video":
            # Video generation - generate_videos
            prompt = self._extract_prompt(messages)
            logger.debug(f"Calling Google aio.models.generate_videos with model={self.model}")
            return await self.client.aio.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "audio":
            # Audio generation (if supported by model)
            prompt = self._extract_prompt(messages)
            logger.debug(f"Calling Google aio.models.generate_audio with model={self.model}")
            # Note: This may need adjustment based on actual Google GenAI audio API
            return await self.client.aio.models.generate_audio(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for Google provider")

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM using the appropriate native SDK.

        Routes to provider-specific method based on provider and output_type.

        Args:
            messages: List of message dicts for the LLM
            tools: Optional list of tool specifications
            stream: Whether to stream the response
            **kwargs: Additional LLM call parameters

        Returns:
            Provider-specific response object

        Raises:
            ValueError: If provider is not supported
        """
        logger.debug(
            f"Calling LLM: provider={self.provider}, output_type={self.output_type}, "
            f"model={self.model}, stream={stream}"
        )

        if self.provider == "openai":
            return await self._call_openai(messages, tools, stream, **kwargs)
        if self.provider == "google":
            return await self._call_google(messages, tools, stream, **kwargs)
        raise ValueError(f"Unsupported provider: {self.provider}")

    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ):
        container = InjectQ.get_instance()

        # check store
        # store: BaseStore | None = container.try_get(BaseStore)

        # if self.learning and store is None:
        #     logger.warning(
        #         "Learning is enabled but no BaseStore instance found in InjectQ container."
        #         " Ignoring learning."
        #     )

        # if store and self.learning:
        #     # retrieve relevant past interactions
        #     state = await store.asearch(state)

        # trim context if enabled
        state = await self._trim_context(state)

        # convert messages for LLM
        messages = convert_messages(
            system_prompts=self.system_prompt,
            state=state,
            extra_messages=self.extra_messages or [],
        )

        # check is is_stream enabled or not
        is_stream = config.get("is_stream", False)

        # If tool results just came in, make final response without tools
        if state.context and len(state.context) > 0 and state.context[-1].role == "tool":
            # Make final response without tools since we just got tool results
            response = await self._call_llm(
                messages=messages,
                stream=is_stream,
            )
        else:
            # Regular response with tools available
            tools = []
            if self._tool_node:
                tools = await self._tool_node.all_tools(tags=self.tools_tags)

            # check form tool node name
            if self.tool_node_name:
                try:
                    node = container.call_factory("get_node", self.tool_node_name)
                except KeyError:
                    logger.warning(
                        f"ToolNode with name '{self.tool_node_name}' not found in InjectQ registry."
                    )
                    node = None

                if node and isinstance(node.func, ToolNode):
                    tools = await node.func.all_tools(tags=self.tools_tags)

            response = await self._call_llm(
                messages=messages,
                tools=tools if tools else None,
                stream=is_stream,
            )

        # Use provider-specific converter
        return ModelResponseConverter(
            response,
            converter=self.provider,
        )
