"""Tests for chunker module, validating exchange grouping and turn fidelity."""

from pathlib import Path

from analyst.chunker import build_exchanges
from analyst.parser import load_raw_dir

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_chunker_on_real_transcripts() -> None:
    """Verify exchange generation against all task constraints and exact data assertions."""
    transcripts, _ = load_raw_dir(DATA_RAW_DIR)
    assert len(transcripts) == 3

    t_map = {t.id: t for t in transcripts}
    t1, t2, t3 = t_map["T1"], t_map["T2"], t_map["T3"]

    exchanges_t1 = build_exchanges(t1)
    exchanges_t2 = build_exchanges(t2)
    exchanges_t3 = build_exchanges(t3)

    # 1. Assert each transcript gives exactly 7 exchanges
    assert len(exchanges_t1) == 7
    assert len(exchanges_t2) == 7
    assert len(exchanges_t3) == 7

    # 2. Assert T1-X3 properties
    t1_x3 = exchanges_t1[2]
    assert t1_x3.chunk_id == "T1-X3"
    assert t1_x3.question_text == "So ROI is important?"
    assert t1_x3.answer_turns[0].ts == "02:18"
    assert t1_x3.embed_text.startswith(
        "Expert: Dr. Jean Martin, Head of Urology, France"
    )

    # 3. Assert T2-X6 properties
    t2_x6 = exchanges_t2[5]
    assert t2_x6.chunk_id == "T2-X6"
    assert t2_x6.answer_turns[0].ts == "05:08"

    # 4. Assert T3-X4 properties
    t3_x4 = exchanges_t3[3]
    assert t3_x4.chunk_id == "T3-X4"
    assert t3_x4.question_text.startswith(
        "So would you say economics are less important in the UK?"
    )
    assert t3_x4.answer_turns[0].ts == "03:10"

    # 5. Assert T3-X7 properties
    t3_x7 = exchanges_t3[6]
    assert t3_x7.chunk_id == "T3-X7"
    assert t3_x7.answer_turns[0].ts == "06:04"

    # 6. Assert concatenating every exchange's turns reproduces all 14 turns in order
    for t, ex_list in [(t1, exchanges_t1), (t2, exchanges_t2), (t3, exchanges_t3)]:
        reconstructed_turns = []
        for ex in ex_list:
            reconstructed_turns.extend(ex.question_turns)
            reconstructed_turns.extend(ex.answer_turns)

        assert len(reconstructed_turns) == 14
        assert [turn.idx for turn in reconstructed_turns] == [
            turn.idx for turn in t.turns
        ]
        assert [turn.text for turn in reconstructed_turns] == [
            turn.text for turn in t.turns
        ]