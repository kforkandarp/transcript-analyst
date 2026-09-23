"""Deterministic citation verification and original-text slice recovery."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence, Tuple

from analyst.models import Claim, Evidence, Exchange, RawClaim


def _normalize_with_mapping(text: str) -> Tuple[str, List[int]]:
    """Normalize text for matching while building a character index map back to the original text.

    Replacements:
      - Unicode NFKC decomposition
      - Curly single/double quotes to straight quotes
      - En-dash/em-dash to '-'
      - Non-breaking spaces and whitespace runs collapsed to a single regular space
      - Strip leading and trailing whitespace
    """
    nfkc_text = unicodedata.normalize("NFKC", text)

    # Step 1: Character substitutions
    subbed_chars: List[str] = []
    for ch in nfkc_text:
        if ch in "‘’‚‛":
            subbed_chars.append("'")
        elif ch in "“”„‟":
            subbed_chars.append('"')
        elif ch in "–—":
            subbed_chars.append("-")
        elif ch == "\u00a0":
            subbed_chars.append(" ")
        else:
            subbed_chars.append(ch)

    # Step 2: Collapse whitespace runs while recording original indices
    norm_chars: List[str] = []
    orig_indices: List[int] = []
    in_whitespace = False

    for orig_idx, ch in enumerate(subbed_chars):
        if ch.isspace():
            if not in_whitespace:
                norm_chars.append(" ")
                orig_indices.append(orig_idx)
                in_whitespace = True
        else:
            norm_chars.append(ch)
            orig_indices.append(orig_idx)
            in_whitespace = False

    # Step 3: Strip leading and trailing whitespace
    start_offset = 0
    while start_offset < len(norm_chars) and norm_chars[start_offset] == " ":
        start_offset += 1

    end_offset = len(norm_chars)
    while end_offset > start_offset and norm_chars[end_offset - 1] == " ":
        end_offset -= 1

    final_norm_chars = norm_chars[start_offset:end_offset]
    final_orig_indices = orig_indices[start_offset:end_offset]

    return "".join(final_norm_chars), final_orig_indices


def _normalize_string_only(text: str) -> str:
    """Normalize a query quote string using the exact same normalization rules."""
    norm, _ = _normalize_with_mapping(text)
    return norm


def verify_quote(quote: str, exchange: Exchange) -> Evidence | None:
    """Verify that a candidate quote exists as a contiguous substring in an expert answer turn.

    Returns an Evidence object containing the literal slice of the original turn text,
    or None if verification fails.
    """
    if not quote or "..." in quote or "…" in quote:
        return None

    words = quote.split()
    if len(words) < 2 or len(words) > 60:
        return None

    norm_quote = _normalize_string_only(quote)
    if not norm_quote:
        return None

    q_len = len(norm_quote)

    # Search strictly inside exchange.answer_turns
    for turn in exchange.answer_turns:
        norm_turn_text, index_map = _normalize_with_mapping(turn.text)
        start_idx = norm_turn_text.find(norm_quote)
        if start_idx == -1:
            continue

        end_idx = start_idx + q_len
        orig_start = index_map[start_idx]
        orig_end = index_map[end_idx - 1] + 1
        original_slice = turn.text[orig_start:orig_end]

        question_ts = (
            exchange.question_turns[0].ts if exchange.question_turns else None
        )

        return Evidence(
            chunk_id=exchange.chunk_id,
            transcript_id=exchange.transcript_id,
            expert_name=exchange.expert_name,
            quote=original_slice,
            ts=turn.ts,
            question_ts=question_ts,
        )

    return None


def verify_claims(
    raw_claims: Sequence[RawClaim],
    exchanges_by_chunk_id: Dict[str, Exchange],
) -> Tuple[List[Claim], dict]:
    """Verify raw claims and their quotes against indexed exchanges.

    Drops failed evidence, drops claims left with zero verified evidence,
    and returns verified claims with verification accounting statistics.
    """
    verified_claims: List[Claim] = []
    total_raw_quotes = 0
    total_verified_quotes = 0
    total_dropped_quotes = 0
    failed_quotes: List[dict] = []

    for raw_claim in raw_claims:
        valid_evidences: List[Evidence] = []

        for raw_ev in raw_claim.evidence:
            total_raw_quotes += 1
            exchange = exchanges_by_chunk_id.get(raw_ev.chunk_id)

            if exchange is None:
                total_dropped_quotes += 1
                failed_quotes.append({"chunk_id": raw_ev.chunk_id, "quote": raw_ev.quote})
                continue

            evidence = verify_quote(raw_ev.quote, exchange)
            if evidence is not None:
                valid_evidences.append(evidence)
                total_verified_quotes += 1
            else:
                total_dropped_quotes += 1
                failed_quotes.append({"chunk_id": raw_ev.chunk_id, "quote": raw_ev.quote})

        if valid_evidences:
            verified_claims.append(
                Claim(text=raw_claim.text, evidence=valid_evidences)
            )

    stats = {
        "raw_quotes": total_raw_quotes,
        "verified_quotes": total_verified_quotes,
        "dropped_quotes": total_dropped_quotes,
        "failed": failed_quotes,
    }

    return verified_claims, stats