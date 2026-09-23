"""Tests for citation verification module against real transcript data and edge cases."""

from pathlib import Path
import pytest

from analyst.chunker import build_exchanges
from analyst.models import Exchange, RawClaim, RawEvidence, Turn
from analyst.parser import load_raw_dir
from analyst.verify import verify_claims, verify_quote

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def real_exchanges() -> dict[str, Exchange]:
    """Load and return exchange map by chunk_id from real data."""
    transcripts, _ = load_raw_dir(DATA_RAW_DIR)
    mapping: dict[str, Exchange] = {}
    for t in transcripts:
        for ex in build_exchanges(t):
            mapping[ex.chunk_id] = ex
    return mapping


def test_quote_exact_and_timestamp(real_exchanges: dict[str, Exchange]) -> None:
    """Exact quote 'Very important.' from T1-X3 verifies with ts '02:18'."""
    t1_x3 = real_exchanges["T1-X3"]
    evidence = verify_quote("Very important.", t1_x3)
    assert evidence is not None
    assert evidence.ts == "02:18"
    assert evidence.quote == "Very important."
    assert evidence.chunk_id == "T1-X3"
    assert evidence.transcript_id == "T1"

def test_quote_normalization_returns_original_text(real_exchanges: dict[str, Exchange]) -> None:
    """Loose whitespace around a real quote still verifies and returns original text."""
    t1_x3 = real_exchanges["T1-X3"]
    loose_quote = "  Very    important.  "
    evidence = verify_quote(loose_quote, t1_x3)
    assert evidence is not None
    assert evidence.quote == "Very important."


def test_curly_apostrophe_normalization() -> None:
    """A curly apostrophe in the candidate quote still matches a straight apostrophe in source text."""
    turn = Turn(
        idx=0,
        ts="02:18",
        seconds=138,
        speaker_label="Dr. Martin",
        role="expert",
        text="The hospital's committee approved the system's rollout last year.",
    )
    exchange = Exchange(
        chunk_id="TX-TEST1",
        transcript_id="TX",
        expert_name="Dr. Test",
        expert_role="Test Role",
        market="Testland",
        question_turns=[],
        answer_turns=[turn],
    )
    curly_quote = "The hospital’s committee approved the system’s rollout"
    evidence = verify_quote(curly_quote, exchange)
    assert evidence is not None
    assert evidence.quote == "The hospital's committee approved the system's rollout"


def test_interviewer_question_rejected(real_exchanges: dict[str, Exchange]) -> None:
    """Quotes taken from interviewer turns must return None."""
    t1_x3 = real_exchanges["T1-X3"]
    # 'So ROI is important?' is an interviewer turn
    assert verify_quote("So ROI is important?", t1_x3) is None


def test_ellipsis_and_length_rejections(real_exchanges: dict[str, Exchange]) -> None:
    """Quotes with ellipsis, <2 words, or >60 words must return None."""
    t1_x3 = real_exchanges["T1-X3"]

    # Contains '...'
    assert verify_quote("Very important... We evaluate", t1_x3) is None
    # Contains unicode ellipsis '…'
    assert verify_quote("Very important… We evaluate", t1_x3) is None
    # Single word
    assert verify_quote("Very", t1_x3) is None
    # Empty string
    assert verify_quote("", t1_x3) is None
    # Over 60 words
    long_quote = "word " * 61
    assert verify_quote(long_quote, t1_x3) is None


def test_mismatched_chunk_or_altered_words(real_exchanges: dict[str, Exchange]) -> None:
    """Quote checked against wrong chunk, changed word, or altered case returns None."""
    t1_x3 = real_exchanges["T1-X3"]
    t2_x4 = real_exchanges["T2-X4"]

    # Real quote from T1-X3 checked against T2-X4
    assert verify_quote("Very important.", t2_x4) is None

    # Changed word
    assert verify_quote("Quite important.", t1_x3) is None

    # Lowercased version (case sensitive match)
    assert verify_quote("very important.", t1_x3) is None


def test_same_phrase_different_turns(real_exchanges: dict[str, Exchange]) -> None:
    """'only one surgeon' verifies in T1-X4 (ts 03:10) and T2-X4 (ts 03:05)."""
    t1_x4 = real_exchanges["T1-X4"]
    t2_x4 = real_exchanges["T2-X4"]

    ev1 = verify_quote("only one surgeon", t1_x4)
    ev2 = verify_quote("only one surgeon", t2_x4)

    assert ev1 is not None
    assert ev1.ts == "03:10"

    assert ev2 is not None
    assert ev2.ts == "03:05"


def test_verify_claims_and_unknown_chunk_accounting(real_exchanges: dict[str, Exchange]) -> None:
    """Unknown chunk_id is dropped and counted in stats; empty claims dropped."""
    raw_claims = [
        RawClaim(
            text="Valid claim",
            evidence=[
                RawEvidence(chunk_id="T1-X3", quote="Very important."),
                RawEvidence(chunk_id="T99-X1", quote="Unknown chunk text"),
            ],
        ),
        RawClaim(
            text="Claim that will be dropped",
            evidence=[
                RawEvidence(chunk_id="T99-X2", quote="Does not exist"),
            ],
        ),
    ]

    claims, stats = verify_claims(raw_claims, real_exchanges)

    # Only 1 claim survived because the second claim had no verified evidence
    assert len(claims) == 1
    assert claims[0].text == "Valid claim"
    assert len(claims[0].evidence) == 1
    assert claims[0].evidence[0].quote == "Very important."

    # Stats assertions
    assert stats["raw_quotes"] == 3
    assert stats["verified_quotes"] == 1
    assert stats["dropped_quotes"] == 2
    assert len(stats["failed"]) == 2
    assert stats["failed"][0]["chunk_id"] == "T99-X1"
    assert stats["failed"][1]["chunk_id"] == "T99-X2"