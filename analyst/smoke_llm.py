"""Smoke test script for verifying LLM connectivity across tiers."""

from __future__ import annotations

import logging
from pydantic import BaseModel
from analyst.llm import CALL_COUNTS, call_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.smoke_llm")


class SimplePing(BaseModel):
    status: str
    message: str


def main() -> None:
    system_prompt = (
        "You are an assistant. Respond only with valid JSON conforming to the requested schema. "
        "Schema: {\"status\": str, \"message\": str}"
    )
    user_prompt = "Generate a JSON response confirming status is 'ok' and a brief greeting."

    print("\n--- Testing Small Tier ---")
    try:
        small_result, small_meta = call_json(
            system=system_prompt,
            user=user_prompt,
            schema=SimplePing,
            tier="small",
            max_retries=3,
        )
        print("Small Tier Response:", small_result.model_dump())
        print("Small Tier Meta:    ", small_meta)
    except Exception as exc:
        print("Small Tier Failed:  ", exc)

    print("\n--- Testing Large Tier ---")
    try:
        large_result, large_meta = call_json(
            system=system_prompt,
            user=user_prompt,
            schema=SimplePing,
            tier="large",
            max_retries=3,
        )
        print("Large Tier Response:", large_result.model_dump())
        print("Large Tier Meta:    ", large_meta)
    except Exception as exc:
        print("Large Tier Failed:  ", exc)

    print("\nTotal Call Counts:", CALL_COUNTS)


if __name__ == "__main__":
    main()