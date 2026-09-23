"""Interview guide question decomposition and parallel cross-transcript synthesis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional
from pydantic import BaseModel

from analyst.answer import generate_verified_answer
from analyst.index import IndexStore, retrieve
from analyst.llm import call_json
from analyst.models import AnswerResult, Guide, GuideCell, GuidePart, GuideQuestion
from analyst.prompts import SPLIT_SYSTEM

logger = logging.getLogger("analyst.guide")


class SplitResponse(BaseModel):
    sub_questions: List[str]


def split_questions(
    guide: Guide,
    overrides_path: Optional[Path] = None,
) -> Guide:
    """Split guide questions into sub-questions using small tier LLM or gold overrides.

    Args:
        guide: The raw parsed Guide instance.
        overrides_path: Path to optional data/gold/guide_overrides.json.

    Returns:
        A new Guide object populated with validated sub-questions.
    """
    overrides: Dict[str, List[str]] = {}
    if overrides_path and overrides_path.exists():
        try:
            with open(overrides_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            logger.info("Loaded manual guide question overrides from %s", overrides_path)
        except Exception as exc:
            logger.warning("Could not read overrides from %s: %s", overrides_path, exc)

    updated_questions: List[GuideQuestion] = []

    for q in guide.questions:
        q_key = str(q.id)

        # 1. Manual override takes first precedence
        if q_key in overrides and isinstance(overrides[q_key], list) and overrides[q_key]:
            sub_qs = [sq.strip() for sq in overrides[q_key] if sq.strip()]
            updated_questions.append(
                GuideQuestion(id=q.id, text=q.text, sub_questions=sub_qs)
            )
            continue

        # 2. LLM sub-question decomposition (tier: small)
        user_prompt = f"Question to split:\n{q.text}"
        sub_qs = [q.text]

        try:
            split_res, _ = call_json(
                system=SPLIT_SYSTEM,
                user=user_prompt,
                schema=SplitResponse,
                tier="small",
                max_retries=2,
            )

            # Validation rules: at most 3 sub-questions, each 3 to 20 words
            candidates = [sq.strip() for sq in split_res.sub_questions if sq.strip()]
            valid_candidates: List[str] = []

            for cand in candidates[:3]:
                word_count = len(cand.split())
                if 3 <= word_count <= 20:
                    valid_candidates.append(cand)

            if valid_candidates:
                sub_qs = valid_candidates
            else:
                logger.warning("Split output for Q%d failed word-count constraints. Falling back.", q.id)

        except Exception as exc:
            logger.warning("Failed splitting Q%d with LLM (%s). Falling back to original question text.", q.id, exc)

        updated_questions.append(
            GuideQuestion(id=q.id, text=q.text, sub_questions=sub_qs)
        )

    return Guide(
        title=guide.title,
        objective=guide.objective,
        questions=updated_questions,
    )


def _process_single_cell(
    store: IndexStore,
    q: GuideQuestion,
    tid: str,
) -> GuideCell:
    """Execute synthesis and verification for all sub-questions of one (question, transcript) pair."""
    parts: List[GuidePart] = []
    sub_qs = q.sub_questions if q.sub_questions else [q.text]

    for sub_q in sub_qs:
        queries = [q.text, sub_q]
        try:
            retrieved = retrieve(
                store=store,
                queries=queries,
                transcript_ids=[tid],
                k_per_query=3,
                neighbours=1,
                max_chunks=8,
            )
            result = generate_verified_answer(
                question=q.text,
                sub_question=sub_q,
                exchanges_by_tid=retrieved,
                mode="guide",
            )
        except Exception as exc:
            logger.error("Failed synthesizing cell for Q%d, %s, sub-q '%s': %s", q.id, tid, sub_q, exc)
            result = AnswerResult(
                coverage="unverified",
                claims=[],
                note=f"Error generating answer: {exc}",
                stats={
                    "raw_quotes": 0,
                    "verified_quotes": 0,
                    "dropped_quotes": 0,
                    "retries": 0,
                    "latency_s": 0.0,
                    "model": "",
                },
            )

        parts.append(GuidePart(sub_question=sub_q, result=result))

    return GuideCell(question_id=q.id, transcript_id=tid, parts=parts)


def answer_guide(
    store: IndexStore,
    guide: Guide,
    callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 4,
) -> List[GuideCell]:
    """Synthesize verified answers across all guide questions and transcripts using a thread pool.

    Args:
        store: Populated FAISS IndexStore.
        guide: Guide containing questions and sub-questions.
        callback: Optional progress reporter called with (completed_cells, total_cells).
        max_workers: Max concurrent threads for parallel cell generation (default 4).

    Returns:
        List of all GuideCell results sorted by (question_id, transcript_id).
    """
    tids = store.transcript_ids()
    total_cells = len(guide.questions) * len(tids)
    completed_cells = 0

    cells: List[GuideCell] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for q in guide.questions:
            for tid in tids:
                future = executor.submit(_process_single_cell, store, q, tid)
                future_map[future] = (q.id, tid)

        for future in as_completed(future_map):
            qid, tid = future_map[future]
            try:
                cell = future.result()
                cells.append(cell)
            except Exception as exc:
                logger.error("Cell worker crashed for Q%d, %s: %s", qid, tid, exc)
                # Fallback: record unverified cell so downstream consumer does not break
                empty_part = GuidePart(
                    sub_question="",
                    result=AnswerResult(
                        coverage="unverified",
                        claims=[],
                        note=f"Cell crashed: {exc}",
                        stats={},
                    ),
                )
                cells.append(GuideCell(question_id=qid, transcript_id=tid, parts=[empty_part]))

            completed_cells += 1
            if callback:
                callback(completed_cells, total_cells)

    # Sort deterministically by question_id, then numerical transcript ID
    cells.sort(
        key=lambda c: (
            c.question_id,
            int(c.transcript_id[1:]) if c.transcript_id.startswith("T") and c.transcript_id[1:].isdigit() else c.transcript_id,
        )
    )

    return cells