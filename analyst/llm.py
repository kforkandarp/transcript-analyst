"""LLM client layer with multi-key failover, retries, and structured JSON parsing."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Dict, Tuple, Type, TypeVar
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq
from pydantic import BaseModel, ValidationError

from analyst.config import settings

logger = logging.getLogger("analyst.llm")

T = TypeVar("T", bound=BaseModel)

# In-memory counter of calls per tier
CALL_COUNTS: Dict[str, int] = {
    "small": 0,
    "large": 0,
}

_CURRENT_KEY_INDEX = 0


class LLMError(Exception):
    """Raised when an LLM call fails completely across all retries and keys."""


def _get_next_client() -> Tuple[Groq, str]:
    """Rotate to and return the next active Groq client instance."""
    global _CURRENT_KEY_INDEX
    keys = settings.groq_api_keys
    if not keys:
        raise LLMError("No Groq API keys configured in settings. Cannot make LLM call.")

    key = keys[_CURRENT_KEY_INDEX % len(keys)]
    _CURRENT_KEY_INDEX = (_CURRENT_KEY_INDEX + 1) % len(keys)
    return Groq(api_key=key, timeout=60.0), key


def _extract_json_substring(text: str) -> str:
    """Extract the first valid JSON object substring from raw text."""
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)

    # Find first '{' and matching '}'
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1]

    return clean.strip()


def call_json(
    system: str,
    user: str,
    schema: Type[T],
    tier: str = "large",
    max_retries: int = 3,
) -> Tuple[T, Dict[str, Any]]:
    """Execute LLM call enforcing valid JSON output conforming to a Pydantic schema.

    Args:
        system: System instructions.
        user: User prompt.
        schema: Target Pydantic model class for validation.
        tier: "small" or "large" (resolves model name from settings).
        max_retries: Maximum validation/network attempts before raising LLMError.

    Returns:
        Tuple of (validated_pydantic_instance, metadata_dict).

    Raises:
        LLMError: If all attempts and keys fail.
    """
    model_name = (
        settings.llm_large_model if tier == "large" else settings.llm_small_model
    )

    current_user_prompt = user
    last_error_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    start_time = time.perf_counter()

    for attempt in range(1, max_retries + 1):
        client, key = _get_next_client()
        masked_key = f"...{key[-4:]}" if len(key) >= 4 else "key"

        try:
            CALL_COUNTS[tier] = CALL_COUNTS.get(tier, 0) + 1

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": current_user_prompt},
                ],
                temperature=0.0,
            )

            # Extract usage metrics if returned
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            raw_reply = response.choices[0].message.content or ""
            json_str = _extract_json_substring(raw_reply)

            parsed_dict = json.loads(json_str)
            validated_obj = schema.model_validate(parsed_dict)

            latency_s = time.perf_counter() - start_time
            meta = {
                "model": model_name,
                "latency_s": round(latency_s, 3),
                "attempts": attempt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            return validated_obj, meta

        except (json.JSONDecodeError, ValidationError) as parse_err:
            last_error_reason = f"Schema validation failed: {parse_err}"
            logger.warning(
                "Attempt %d/%d with model %s failed JSON validation (%s). Retrying...",
                attempt,
                max_retries,
                model_name,
                parse_err,
            )
            # Append validation error feedback to user prompt for subsequent turn
            current_user_prompt = (
                f"{user}\n\n"
                f"Your previous reply was invalid: {parse_err}. "
                f"Return ONLY valid JSON matching the schema."
            )

        except (APIStatusError, APIConnectionError, APITimeoutError) as net_err:
            last_error_reason = f"Network or API status error: {net_err}"
            logger.warning(
                "Attempt %d/%d failed on key %s (%s). Falling back with backoff...",
                attempt,
                max_retries,
                masked_key,
                net_err,
            )

        except Exception as exc:
            last_error_reason = f"Unexpected error: {exc}"
            logger.warning(
                "Attempt %d/%d failed unexpectedly (%s).",
                attempt,
                max_retries,
                exc,
            )

        # Exponential backoff + jitter before retrying
        if attempt < max_retries:
            sleep_time = (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)

    total_latency = time.perf_counter() - start_time
    raise LLMError(
        f"Failed to obtain valid response from tier '{tier}' ({model_name}) "
        f"after {max_retries} attempts ({total_latency:.2f}s elapsed). "
        f"Last error: {last_error_reason}"
    )