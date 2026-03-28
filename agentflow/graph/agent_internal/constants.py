"""Shared constants for Agent internals."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for LLM call retry and fallback behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts for the *primary* model
            before moving to the next fallback (default ``3``).
        initial_delay: Delay in seconds before the first retry (default ``1.0``).
        max_delay: Upper-bound cap on exponential back-off delay (default ``30.0``).
        backoff_factor: Multiplier applied after each retry (default ``2.0``).
        retryable_status_codes: HTTP status codes considered transient/retryable.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 529}),
    )


DEFAULT_RETRY_CONFIG = RetryConfig()

CLIENT_CONSTRUCTOR_KWARGS = frozenset(
    {
        "organization",
        "project",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
    }
)

# Keys that must never be forwarded to request calls.
CALL_EXCLUDED_KWARGS = CLIENT_CONSTRUCTOR_KWARGS | frozenset(
    {
        "api_key",
        "base_url",
    }
)

VALID_OUTPUT_TYPES = ("text", "image", "video", "audio")
GOOGLE_OUTPUT_TYPES = ("text", "image", "video", "audio")
OPENAI_OUTPUT_TYPES = ("text", "image", "audio")

GOOGLE_THINKING_BUDGET_BY_EFFORT = {
    "low": 512,
    "medium": 8192,
    "high": 24576,
}

# Sentinel that distinguishes the default reasoning config from an explicit None.
REASONING_DEFAULT: object = object()
