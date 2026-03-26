"""Integration adapters for optional third-party LLM SDKs.

This module exposes universal converter APIs to normalize responses and
streaming outputs from popular LLM SDKs (for example OpenAI and Google
GenAI) for use in agentflow agent graphs. Adapters provide unified
conversion, streaming, and message normalization for agent workflows.

Exports:
    BaseConverter: Abstract base class for LLM response converters.
    ConverterType: Enum of supported converter types.
    GoogleGenAIConverter: Converter for Google Generative AI responses and streams.
    OpenAIConverter: Converter for OpenAI responses and streams.
"""

from .base_converter import BaseConverter, ConverterType
from .google_genai_converter import GoogleGenAIConverter
from .openai_converter import OpenAIConverter
from .openai_responses_converter import OpenAIResponsesConverter


__all__ = [
    "BaseConverter",
    "ConverterType",
    "GoogleGenAIConverter",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
]
