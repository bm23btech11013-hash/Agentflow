"""Provider detection, validation, and client creation helpers for Agent."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from .constants import (
    CLIENT_CONSTRUCTOR_KWARGS,
    GOOGLE_OUTPUT_TYPES,
    OPENAI_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES,
)


logger = logging.getLogger("agentflow.agent")


class _ProviderAgentLike(Protocol):
    output_type: str
    provider: str
    llm_kwargs: dict[str, Any]


class AgentProviderMixin:
    """Provider-specific validation and client creation helpers."""

    def _validate_output_type(self: _ProviderAgentLike) -> None:
        """Validate that the selected provider supports the requested output type."""
        if self.output_type not in VALID_OUTPUT_TYPES:
            raise ValueError(
                "Invalid output_type "
                f"'{self.output_type}'. Supported types: {list(VALID_OUTPUT_TYPES)}"
            )

        if self.provider == "google" and self.output_type not in GOOGLE_OUTPUT_TYPES:
            raise ValueError(
                f"Google provider doesn't support output_type='{self.output_type}'. "
                f"Supported: {list(GOOGLE_OUTPUT_TYPES)}"
            )

        if self.provider == "openai" and self.output_type not in OPENAI_OUTPUT_TYPES:
            raise ValueError(
                f"OpenAI provider doesn't support output_type='{self.output_type}'. "
                f"Supported: {list(OPENAI_OUTPUT_TYPES)}"
            )

        logger.debug("%s provider supports output_type='%s'", self.provider, self.output_type)

    def _detect_provider_from_model(self, model: str) -> str:
        """Infer the provider from the model name when not explicitly supplied."""
        model_lower = model.lower()

        if model_lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return "openai"
        if model_lower.startswith(("gemini-", "imagen-", "veo-", "chirp")):
            return "google"

        logger.info(
            "Could not auto-detect provider for model '%s'. Defaulting to 'openai'. "
            "If using a different provider, specify explicitly.",
            model,
        )
        return "openai"

    def _create_client(
        self: _ProviderAgentLike,
        provider: str,
        base_url: str | None = None,
    ) -> Any:
        """Create a native SDK client for the selected provider."""
        if provider == "openai":
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai SDK is required for OpenAI provider. "
                    "Install it with: pip install 10xscale-agentflow[openai]"
                ) from exc

            api_key = self.llm_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY environment variable not set. "
                    "API calls may fail if not using custom client or base_url without auth."
                )

            client_kwargs = {
                key: value
                for key, value in self.llm_kwargs.items()
                if key in CLIENT_CONSTRUCTOR_KWARGS
            }
            if base_url:
                logger.info("Using custom base_url for OpenAI client: %s", base_url)
                return AsyncOpenAI(api_key=api_key, base_url=base_url, **client_kwargs)
            return AsyncOpenAI(api_key=api_key, **client_kwargs)

        if provider == "google":
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai SDK is required for Google provider. "
                    "Install it with: pip install 10xscale-agentflow[google-genai]"
                ) from exc

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set "
                    "for Google provider"
                )

            logger.info("Creating Google GenAI Client with async support")
            return genai.Client(api_key=api_key)

        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: 'openai', 'google'"
        )
