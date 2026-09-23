"""Chunker module to group transcript speaker turns into interview exchanges."""

from __future__ import annotations

import logging
from typing import List

from analyst.models import Exchange, Transcript, Turn

logger = logging.getLogger("analyst.chunker")


def build_exchanges(t: Transcript) -> list[Exchange]:
    """Group a transcript's sequential turns into structured Exchange chunks.

    One exchange consists of one or more consecutive interviewer turns followed
    by one or more consecutive expert turns. Expert turns appearing before any
    interviewer turn form an exchange with empty question turns. Trailing interviewer
    turns with no subsequent expert response are skipped with a warning.

    Args:
        t: The parsed Transcript model instance.

    Returns:
        A list of Exchange objects indexed with 1-based chunk_id ("{t.id}-X{n}").
    """
    exchanges: List[Exchange] = []
    exchange_counter = 1

    current_question_turns: List[Turn] = []
    current_answer_turns: List[Turn] = []

    def flush() -> None:
        nonlocal exchange_counter, current_question_turns, current_answer_turns
        if not current_question_turns and not current_answer_turns:
            return

        if not current_answer_turns:
            logger.warning(
                "Skipping trailing exchange without expert answer in transcript %s: %s",
                t.id,
                [turn.text for turn in current_question_turns],
            )
            current_question_turns = []
            return

        chunk = Exchange(
            chunk_id=f"{t.id}-X{exchange_counter}",
            transcript_id=t.id,
            expert_name=t.expert_name,
            expert_role=t.expert_role,
            market=t.market,
            question_turns=list(current_question_turns),
            answer_turns=list(current_answer_turns),
        )
        exchanges.append(chunk)
        exchange_counter += 1

        current_question_turns = []
        current_answer_turns = []

    for turn in t.turns:
        if turn.role == "interviewer":
            if current_answer_turns:
                flush()
            current_question_turns.append(turn)
        elif turn.role == "expert":
            current_answer_turns.append(turn)

    flush()

    return exchanges