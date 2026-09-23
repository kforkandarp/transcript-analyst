"""Cross-transcript interactive chat with query rewriting and targeted retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from analyst.answer import generate_verified_answer
from analyst.index import IndexStore, retrieve
from analyst.llm import LLMError, call_json
from analyst.models import AnswerResult, Guide, Transcript
from analyst.prompts import QUERY_ANALYSE_SYSTEM

logger = logging.getLogger("analyst.chat")


class QueryAnalysisResult(BaseModel):
    intent: Literal["greeting", "question", "out_of_scope"] = Field(
        default="question",
        description="Type of user input: 'greeting' for pure pleasantries, 'question' for transcript inquiries, 'out_of_scope' for off-topic queries."
    )
    standalone_question: str
    target_transcript_ids: List[str] = Field(default_factory=list)
    unknown_expert_mentioned: Optional[str] = None


def topics_covered(guide: Guide) -> List[str]:
    """Extract and return the text of all guide questions for refusal or guidance messages."""
    return [q.text for q in guide.questions]


def _format_known_experts(transcripts: Sequence[Transcript]) -> str:
    """Format known transcripts as 'T1: Dr. Jean Martin (France)' lines."""
    lines = []
    for t in transcripts:
        lines.append(f"{t.id}: {t.expert_name} ({t.market})")
    return "\n".join(lines)


def _format_chat_history(history: Sequence[Dict[str, str]]) -> str:
    """Format the last 4 turns of chat history."""
    if not history:
        return "None"
    recent = history[-4:]
    formatted = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        formatted.append(f"{role}: {turn_content}" if (turn_content := content) else "")
    return "\n".join(filter(None, formatted))


def ask(
    store: IndexStore,
    transcripts: Sequence[Transcript],
    guide: Guide,
    question: str,
    history: Sequence[Dict[str, str]],
) -> Tuple[AnswerResult, Dict[str, Any]]:
    """Answer a user query with history disambiguation and targeted cross-transcript synthesis.

    Args:
        store: Initialized vector index.
        transcripts: All loaded transcripts.
        guide: Interview guide instance.
        question: User query string.
        history: List of past turns formatted as {"role": str, "content": str}.

    Returns:
        A tuple of (AnswerResult, debug_dict).
    """
    all_tids = [t.id for t in transcripts]
    known_experts_str = _format_known_experts(transcripts)

    # 1. Query Analysis Step (always runs on small tier)
    standalone_question = question
    target_ids = list(all_tids)
    unknown_expert: Optional[str] = None
    query_analysis_stats: Dict[str, Any] = {}
    analysis: Optional[QueryAnalysisResult] = None

    history_str = _format_chat_history(history)
    user_prompt = (
        f"KNOWN EXPERTS:\n{known_experts_str}\n\n"
        f"RECENT CHAT HISTORY:\n{history_str}\n\n"
        f"LATEST QUESTION:\n{question}"
    )

    try:
        analysis, meta = call_json(
            system=QUERY_ANALYSE_SYSTEM,
            user=user_prompt,
            schema=QueryAnalysisResult,
            tier="small",
            max_retries=2,
        )
        query_analysis_stats = meta
        standalone_question = analysis.standalone_question or question
        unknown_expert = analysis.unknown_expert_mentioned

        valid_targets = [tid for tid in analysis.target_transcript_ids if tid in all_tids]
        target_ids = valid_targets if valid_targets else list(all_tids)

    except Exception as exc:
        logger.warning("Query analysis failed (%s). Falling back to raw query and all transcripts.", exc)
        standalone_question = question
        target_ids = list(all_tids)

    # 1.5 Intercept Greeting Intent
    if analysis and getattr(analysis, "intent", "question") == "greeting":
        return (
            AnswerResult(
                coverage="direct",
                claims=[],
                note="Hi! Ask me anything about the three expert interviews on robotic surgery adoption in Europe.",
                stats={
                    "raw_quotes": 0,
                    "verified_quotes": 0,
                    "dropped_quotes": 0,
                    "retries": 0,
                    "latency_s": query_analysis_stats.get("latency_s", 0.0),
                    "model": query_analysis_stats.get("model", ""),
                },
            ),
            {
                "standalone_question": standalone_question,
                "target_ids": target_ids,
                "retrieved_chunk_ids": [],
                "model_stats": query_analysis_stats,
            },
        )

    # 1.6 Intercept Out of Scope Intent
    if analysis and getattr(analysis, "intent", "question") == "out_of_scope":
        return (
            AnswerResult(
                coverage="not_discussed",
                claims=[],
                note="This question falls outside the scope of the three expert interview transcripts. I can only answer questions related to robotic surgery adoption, hospital procurement, and clinical practices discussed in the interviews.",
                stats={
                    "raw_quotes": 0,
                    "verified_quotes": 0,
                    "dropped_quotes": 0,
                    "retries": 0,
                    "latency_s": query_analysis_stats.get("latency_s", 0.0),
                    "model": query_analysis_stats.get("model", ""),
                },
            ),
            {
                "standalone_question": standalone_question,
                "target_ids": target_ids,
                "retrieved_chunk_ids": [],
                "model_stats": query_analysis_stats,
            },
        )

    # 2. Intercept Unknown Expert Mention
    if unknown_expert:
        known_list = ", ".join(f"{t.expert_name} ({t.market})" for t in transcripts)
        return (
            AnswerResult(
                coverage="not_discussed",
                claims=[],
                note=f"I could not find information regarding '{unknown_expert}'. Available experts are: {known_list}.",
                stats={
                    "raw_quotes": 0,
                    "verified_quotes": 0,
                    "dropped_quotes": 0,
                    "retries": 0,
                    "latency_s": query_analysis_stats.get("latency_s", 0.0),
                    "model": query_analysis_stats.get("model", ""),
                },
            ),
            {
                "standalone_question": standalone_question,
                "target_ids": target_ids,
                "unknown_expert_mentioned": unknown_expert,
                "retrieved_chunk_ids": [],
                "model_stats": query_analysis_stats,
            },
        )

    # 3. Targeted Retrieval & 4. Verified Synthesis
    try:
        retrieved = retrieve(
            store=store,
            queries=[standalone_question],
            transcript_ids=target_ids,
            k_per_query=3,
            neighbours=1,
            max_chunks=6,
        )

        retrieved_chunk_ids = [
            ex.chunk_id for ex_list in retrieved.values() for ex in ex_list
        ]

        result = generate_verified_answer(
            question=standalone_question,
            sub_question="",
            exchanges_by_tid=retrieved,
            mode="chat",
        )

        debug = {
            "standalone_question": standalone_question,
            "target_ids": target_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "model_stats": {
                "query_analysis": query_analysis_stats,
                "answer": result.stats,
            },
        }
        return result, debug

    except Exception as exc:
        logger.error("Chat execution failed: %s", exc)
        return (
            AnswerResult(
                coverage="unverified",
                claims=[],
                note="I encountered an issue generating a verified answer. Please try rephrasing your question.",
                stats={},
            ),
            {
                "standalone_question": standalone_question,
                "target_ids": target_ids,
                "retrieved_chunk_ids": [],
                "error": str(exc),
            },
        )