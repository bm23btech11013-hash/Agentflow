"""Shared constants for Agent internals."""

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
