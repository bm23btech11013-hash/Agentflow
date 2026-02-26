"""
Shared LLM calling utilities for criteria.

Provides LLMCallerMixin with _call_llm_score() used by
LLMJudgeCriterion, RubricBasedCriterion, and others.
"""

from __future__ import annotations

import json
import logging


logger = logging.getLogger("agentflow.evaluation")


def _parse_model_provider(model: str) -> tuple[str, str]:
    """Derive (provider, clean_model) from a model string.

    Supports:
      - ``"gemini-2.5-flash"``          → ``("google", "gemini-2.5-flash")``
      - ``"gemini/gemini-2.5-flash"``    → ``("google", "gemini-2.5-flash")``
        (legacy LiteLLM syntax)
      - ``"gpt-4o"``                     → ``("openai", "gpt-4o")``
      - ``"openai/gpt-4o"``              → ``("openai", "gpt-4o")``

    Returns:
        Tuple of (provider, model_name).
    """
    if "/" in model:
        prefix, name = model.split("/", 1)
        prefix_lower = prefix.lower()
        if prefix_lower in ("gemini", "google"):
            return "google", name
        if prefix_lower in ("openai", "gpt"):
            return "openai", name
        # Unknown prefix — treat the whole string as model name, guess provider
        model = name

    lower = model.lower()
    if lower.startswith("gemini"):
        return "google", model
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", model
    # Default to google
    return "google", model


class LLMCallerMixin:
    """Mixin providing shared LLM calling logic for criteria.

    Uses Google GenAI SDK as the primary path, then litellm, then OpenAI
    as fallback.  All LLM-based criteria inherit from this.
    """

    async def _call_llm_json(self, prompt: str) -> dict:
        """Call LLM and return full parsed JSON dict.

        Tries Google GenAI first, then litellm, then OpenAI. If none
        are available, returns a default dict with score 0.5.

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Parsed JSON dict from LLM response.
        """
        provider, model_name = _parse_model_provider(self.config.judge_model)

        if provider == "google":
            result = await self._call_google_json(model_name, prompt)
            if result is not None:
                return result

        # litellm path — works with any provider litellm supports
        result = await self._call_litellm_json(self.config.judge_model, prompt)
        if result is not None:
            return result

        # OpenAI path (primary for OpenAI models, fallback for Google failures)
        result = await self._call_openai_json(
            self.config.judge_model if provider == "openai" else model_name,
            prompt,
        )
        if result is not None:
            return result

        # Last resort: try Google if we haven't yet
        if provider != "google":
            result = await self._call_google_json(model_name, prompt)
            if result is not None:
                return result

        logger.warning("No LLM library available, returning default score")
        return {"score": 0.5, "reasoning": "No LLM available"}

    async def _call_google_json(self, model: str, prompt: str) -> dict | None:
        """Call Google GenAI and return parsed JSON dict, or None on failure."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client()
            config = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Google GenAI returned empty content")
            return json.loads(text)
        except ImportError:
            return None
        except Exception as e:
            logger.warning("Google GenAI call failed: %s", e)
            return None

    async def _call_litellm_json(self, model: str, prompt: str) -> dict | None:
        """Call LLM via litellm and return parsed JSON dict, or None on failure.

        litellm supports Google Gemini (``gemini/model``), OpenAI, Anthropic,
        and many other providers with a unified interface.  It uses the
        ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` environment variables
        automatically.
        """
        try:
            import litellm

            # Ensure model uses litellm provider prefix for Gemini
            litellm_model = model
            if "/" not in model and model.lower().startswith("gemini"):
                litellm_model = f"gemini/{model}"

            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("litellm returned empty content")
            return json.loads(text)
        except ImportError:
            return None
        except Exception as e:
            logger.warning("litellm call failed: %s", e)
            return None

    async def _call_openai_json(self, model: str, prompt: str) -> dict | None:
        """Call OpenAI and return parsed JSON dict, or None on failure."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("OpenAI returned empty content")
            return json.loads(text)
        except ImportError:
            return None
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)
            return None

    async def _call_llm_score(self, prompt: str) -> tuple[float, str]:
        """Call LLM and return (score, reasoning).

        Convenience wrapper around :meth:`_call_llm_json` that extracts
        the ``score`` and ``reasoning`` fields from the response dict.

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Tuple of (score float 0-1, reasoning string).
        """
        result = await self._call_llm_json(prompt)
        return float(result.get("score", 0.0)), result.get("reasoning", "")
