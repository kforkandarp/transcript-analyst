"""Map-Reduce theme extraction and cross-transcript synthesis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from typing import Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field

from analyst.index import IndexStore
from analyst.llm import LLMError, call_json
from analyst.models import (
    Evidence,
    Exchange,
    RawClaim,
    RawEvidence,
    Theme,
    ThemePosition,
    Transcript,
)
from analyst.prompts import THEME_MAP_SYSTEM, THEME_REDUCE_SYSTEM
from analyst.verify import verify_claims

logger = logging.getLogger("analyst.themes")


# --- Internal Pydantic Schemas for Map / Reduce JSON Parsing ---

class RawMapTheme(BaseModel):
    label: str
    position: str
    evidence: List[RawEvidence] = Field(default_factory=list)
    value: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None


class RawMapOutput(BaseModel):
    themes: List[RawMapTheme] = Field(default_factory=list)


class IntermediatePosition(BaseModel):
    ref: str
    transcript_id: str
    expert: str
    role: str
    market: str
    label: str
    position: str
    evidence: List[Evidence]
    value: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None


class RawReduceTheme(BaseModel):
    title: str
    kind: Literal["common", "disagreement", "unique"]
    verdict: Optional[Literal["agree", "partly_agree", "disagree", "not_comparable"]] = None
    summary: str
    position_refs: List[str] = Field(default_factory=list)


class RawReduceOutput(BaseModel):
    themes: List[RawReduceTheme] = Field(default_factory=list)


# --- Helper Functions ---

def _format_transcript_chunks(transcript: Transcript, exchanges: Sequence[Exchange]) -> str:
    """Format all exchanges of a single transcript using standard chunk formatting."""
    lines: List[str] = [f"TRANSCRIPT: {transcript.id} - {transcript.expert_name}"]
    lines.append("CHUNKS:")

    for ex in exchanges:
        lines.append(
            f"[CHUNK {ex.chunk_id} | {ex.expert_name}, {ex.expert_role}, {ex.market}]\n"
            f"INTERVIEWER (context only, never quote): {ex.question_text}\n"
            f"EXPERT (evidence, quote from here only): {ex.answer_text}\n"
        )

    return "\n".join(lines).strip()


def _map_single_transcript(
    transcript: Transcript,
    exchanges: Sequence[Exchange],
) -> List[IntermediatePosition]:
    """Execute theme extraction and citation verification for a single transcript."""
    chunk_map = {ex.chunk_id: ex for ex in exchanges}
    base_user_prompt = _format_transcript_chunks(transcript, exchanges)

    try:
        raw_map_out, _ = call_json(
            system=THEME_MAP_SYSTEM,
            user=base_user_prompt,
            schema=RawMapOutput,
            tier="large",
            max_retries=3,
        )
    except LLMError as exc:
        logger.error("Map step failed on transcript %s: %s", transcript.id, exc)
        return []

    def _verify_theme(raw_theme: RawMapTheme) -> Optional[List[Evidence]]:
        claim = RawClaim(text=raw_theme.position, evidence=raw_theme.evidence)
        verified, _ = verify_claims([claim], chunk_map)
        return verified[0].evidence if verified else None

    surviving_positions: List[IntermediatePosition] = []
    unresolved: List[RawMapTheme] = []
    ref_counter = 1

    for raw_theme in raw_map_out.themes:
        evidence = _verify_theme(raw_theme)
        if evidence is None:
            unresolved.append(raw_theme)
            continue

        surviving_positions.append(
            IntermediatePosition(
                ref=f"{transcript.id}-M{ref_counter}",
                transcript_id=transcript.id,
                expert=transcript.expert_name,
                role=transcript.expert_role,
                market=transcript.market,
                label=raw_theme.label,
                position=raw_theme.position,
                evidence=evidence,
                value=raw_theme.value,
                scope=raw_theme.scope,
                qualifier=raw_theme.qualifier,
            )
        )
        ref_counter += 1

    if unresolved:
        failed_msg = "\n".join(f"- \"{t.label}\": {t.position}" for t in unresolved)
        retry_prompt = (
            f"{base_user_prompt}\n\n"
            f"These themes' quotes were not found verbatim in the EXPERT text:\n"
            f"{failed_msg}\n"
            f"Re-copy their quotes exactly. Return the FULL corrected theme list."
        )

        try:
            raw_retry, _ = call_json(
                system=THEME_MAP_SYSTEM,
                user=retry_prompt,
                schema=RawMapOutput,
                tier="large",
                max_retries=3,
            )
            unresolved_labels = {t.label for t in unresolved}
            for raw_theme in raw_retry.themes:
                if raw_theme.label not in unresolved_labels:
                    continue
                evidence = _verify_theme(raw_theme)
                if evidence is None:
                    continue
                surviving_positions.append(
                    IntermediatePosition(
                        ref=f"{transcript.id}-M{ref_counter}",
                        transcript_id=transcript.id,
                        expert=transcript.expert_name,
                        role=transcript.expert_role,
                        market=transcript.market,
                        label=raw_theme.label,
                        position=raw_theme.position,
                        evidence=evidence,
                        value=raw_theme.value,
                        scope=raw_theme.scope,
                        qualifier=raw_theme.qualifier,
                    )
                )
                ref_counter += 1
        except LLMError as exc:
            logger.warning("Retry map step failed for %s: %s. Keeping first-pass results.", transcript.id, exc)

    return surviving_positions


# --- Main Orchestration Entry Point ---

def build_themes(store: IndexStore, transcripts: Sequence[Transcript]) -> List[Theme]:
    """Extract and synthesize cross-transcript themes via Map-Reduce.

    Args:
        store: Initialized vector store containing indexed chunks.
        transcripts: All parsed transcript models.

    Returns:
        List of verified Theme instances sorted by disagreements first,
        then common by descending expert_count, then unique.
    """
    if not transcripts:
        return []

    # Map transcript ID to exchanges directly from store
    all_exchanges_by_tid: Dict[str, List[Exchange]] = {
        t.id: store.exchanges(t.id) for t in transcripts
    }

    # --- 1. MAP PHASE (Parallel across transcripts, max 3 threads) ---
    all_intermediate_positions: List[IntermediatePosition] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(
                _map_single_transcript,
                t,
                all_exchanges_by_tid.get(t.id, []),
            ): t.id
            for t in transcripts
        }

        for future in as_completed(future_map):
            tid = future_map[future]
            try:
                positions = future.result()
                all_intermediate_positions.extend(positions)
            except Exception as exc:
                logger.error("Failed map stage for transcript %s: %s", tid, exc)

    if not all_intermediate_positions:
        logger.warning("No positions survived the MAP phase.")
        return []

    positions_by_ref: Dict[str, IntermediatePosition] = {
        pos.ref: pos for pos in all_intermediate_positions
    }

    # --- 2. REDUCE PHASE ---
    # Build compact JSON representation containing metadata only (strictly NO quotes)
    compact_positions_payload = [
        {
            "ref": pos.ref,
            "expert": pos.expert,
            "role": pos.role,
            "market": pos.market,
            "label": pos.label,
            "position": pos.position,
            "value": pos.value,
            "scope": pos.scope,
            "qualifier": pos.qualifier,
        }
        for pos in all_intermediate_positions
    ]

    reduce_user_prompt = (
        "POSITIONS:\n"
        f"{json.dumps(compact_positions_payload, indent=2, ensure_ascii=False)}"
    )

    try:
        reduce_out, _ = call_json(
            system=THEME_REDUCE_SYSTEM,
            user=reduce_user_prompt,
            schema=RawReduceOutput,
            tier="large",
            max_retries=3,
        )
    except LLMError as exc:
        logger.error("Reduce step failed with LLMError: %s", exc)
        return []

    # --- 3. ASSEMBLY & POST-PROCESSING ---
    resolved_themes: List[Theme] = []

    for raw_th in reduce_out.themes:
        matched_positions: List[IntermediatePosition] = [
            positions_by_ref[ref]
            for ref in raw_th.position_refs
            if ref in positions_by_ref
        ]

        if not matched_positions:
            continue

        # Verified ThemePositions (role omitted per models.py specification)
        final_positions: List[ThemePosition] = [
            ThemePosition(
                transcript_id=p.transcript_id,
                expert_name=p.expert,
                market=p.market,
                position=p.position,
                evidence=p.evidence,
                value=p.value,
                scope=p.scope,
                qualifier=p.qualifier,
            )
            for p in matched_positions
        ]

        distinct_tids = {p.transcript_id for p in final_positions}
        expert_count = len(distinct_tids)

        kind = raw_th.kind
        if expert_count == 1:
            kind = "unique"
        elif kind == "common" and expert_count < 2:
            kind = "unique"

        verdict = None if kind == "unique" else raw_th.verdict

        resolved_themes.append(
            Theme(
                title=th_title if (th_title := raw_th.title) else "Untitled Theme",
                kind=kind,
                verdict=verdict,
                summary=raw_th.summary,
                expert_count=expert_count,
                positions=final_positions,
            )
        )

    # Sort: disagreements first, then common by expert_count descending, then unique
    def _sort_key(t: Theme):
        kind_order = {"disagreement": 0, "common": 1, "unique": 2}
        return (
            kind_order.get(t.kind, 3),
            -t.expert_count if t.kind == "common" else 0,
            t.title,
        )

    resolved_themes.sort(key=_sort_key)
    return resolved_themes