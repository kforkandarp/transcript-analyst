"""Synthesis and verified answer generation over retrieved transcript exchanges."""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

from analyst.llm import LLMError, call_json
from analyst.models import AnswerResult, Exchange, RawAnswer
from analyst.prompts import ANSWER_SYSTEM, CHAT_EXTRA
from analyst.verify import verify_claims

logger = logging.getLogger("analyst.answer")


def _format_user_prompt(
    question: str,
    sub_question: Optional[str],
    exchanges_by_tid: Dict[str, List[Exchange]],
    mode: Literal["guide", "chat"],
) -> str:
    """Build the user prompt containing question context and grouped exchange chunks."""
    lines: List[str] = [f"QUESTION: {question}"]
    if mode == "guide" and sub_question:
        lines.append(f"FOCUS: {sub_question}")

    lines.append("CHUNKS:")

    # Sort transcript IDs numerically to guarantee transcript order
    sorted_tids = sorted(
        exchanges_by_tid.keys(),
        key=lambda tid: int(tid[1:]) if tid.startswith("T") and tid[1:].isdigit() else tid,
    )

    for tid in sorted_tids:
        exchanges = exchanges_by_tid[tid]
        for ex in exchanges:
            lines.append(
                f"[CHUNK {ex.chunk_id} | {ex.expert_name}, {ex.expert_role}, {ex.market}]\n"
                f"INTERVIEWER (context only, never quote): {ex.question_text}\n"
                f"EXPERT (evidence, quote from here only): {ex.answer_text}\n"
            )

    return "\n".join(lines).strip()


def generate_verified_answer(
    question: str,
    sub_question: Optional[str],
    exchanges_by_tid: Dict[str, List[Exchange]],
    mode: Literal["guide", "chat"] = "guide",
) -> AnswerResult:
    """Generate and deterministically verify an answer from retrieved chunks.

    Args:
        question: Primary user or guide question.
        sub_question: Optional sub-focus query (used only in 'guide' mode).
        exchanges_by_tid: Mapping of transcript ID to retrieved exchanges.
        mode: 'guide' (single or scoped) or 'chat' (cross-transcript).

    Returns:
        An AnswerResult instance containing verified claims and stats.
    """
    # Build dictionary of all available exchanges keyed by chunk_id
    all_chunks: Dict[str, Exchange] = {}
    for ex_list in exchanges_by_tid.values():
        for ex in ex_list:
            all_chunks[ex.chunk_id] = ex

    if not all_chunks:
        return AnswerResult(
            coverage="not_discussed",
            claims=[],
            note="No relevant transcript chunks available to answer the question.",
            stats={
                "raw_quotes": 0,
                "verified_quotes": 0,
                "dropped_quotes": 0,
                "retries": 0,
                "latency_s": 0.0,
                "model": "",
            },
        )

    # Determine system prompt
    system = ANSWER_SYSTEM if mode == "guide" else f"{ANSWER_SYSTEM}\n\n{CHAT_EXTRA}"
    base_user_prompt = _format_user_prompt(question, sub_question, exchanges_by_tid, mode)

    total_latency = 0.0
    used_model = ""
    retries_done = 0

    # Attempt 1
    try:
        raw_ans, meta1 = call_json(
            system=system,
            user=base_user_prompt,
            schema=RawAnswer,
            tier="large",
            max_retries=3,
        )
        total_latency += meta1.get("latency_s", 0.0)
        used_model = meta1.get("model", "")
    except LLMError as exc:
        logger.error("Attempt 1 failed with LLMError: %s", exc)
        return AnswerResult(
            coverage="unverified",
            claims=[],
            note="Could not generate a verified answer",
            stats={
                "raw_quotes": 0,
                "verified_quotes": 0,
                "dropped_quotes": 0,
                "retries": 0,
                "latency_s": total_latency,
                "model": used_model,
            },
        )

    verified_claims, stats = verify_claims(raw_ans.claims, all_chunks)

    # If any evidence was dropped and claims had quotes, retry once with targeted feedback
    if stats["dropped_quotes"] > 0:
        retries_done = 1
        failed_items = [
            f"- [{item['chunk_id']}] \"{item['quote']}\""
            for item in stats["failed"]
        ]
        failed_msg = "\n".join(failed_items)
        retry_prompt = (
            f"{base_user_prompt}\n\n"
            f"These quotes were not found verbatim in the EXPERT text:\n{failed_msg}\n"
            f"Re-copy them exactly or remove the claim."
        )

        try:
            raw_ans_retry, meta2 = call_json(
                system=system,
                user=retry_prompt,
                schema=RawAnswer,
                tier="large",
                max_retries=3,
            )
            total_latency += meta2.get("latency_s", 0.0)
            used_model = meta2.get("model", used_model)
            raw_ans = raw_ans_retry
            verified_claims, stats = verify_claims(raw_ans.claims, all_chunks)
        except LLMError as exc:
            logger.warning("Retry attempt failed with LLMError: %s. Keeping first attempt results.", exc)

    # Assemble combined stats
    final_stats = {
        "raw_quotes": stats["raw_quotes"],
        "verified_quotes": stats["verified_quotes"],
        "dropped_quotes": stats["dropped_quotes"],
        "retries": retries_done,
        "latency_s": round(total_latency, 3),
        "model": used_model,
    }

    # Coverage handling rules
    model_coverage = raw_ans.coverage
    if model_coverage == "not_discussed":
        return AnswerResult(
            coverage="not_discussed",
            claims=[],
            note=raw_ans.note,
            stats=final_stats,
        )

    if model_coverage in ("direct", "indirect"):
        if not verified_claims:
            return AnswerResult(
                coverage="unverified",
                claims=[],
                note=raw_ans.note or "Could not verify quotes for generated claims.",
                stats=final_stats,
            )
        return AnswerResult(
            coverage=model_coverage,
            claims=verified_claims,
            note=raw_ans.note,
            stats=final_stats,
        )

    return AnswerResult(
        coverage="unverified",
        claims=[],
        note=raw_ans.note or "Unrecognized coverage status.",
        stats=final_stats,
    )