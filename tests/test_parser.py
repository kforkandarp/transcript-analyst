"""Tests for parser implementation, verifying data contracts and edge cases."""

from pathlib import Path
import pytest

from analyst.parser import ParseError, load_raw_dir, parse_transcript
from analyst.models import Transcript

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_real_files_load_and_contracts() -> None:
    """Verify loading real raw files against the Master Context specifications."""
    transcripts, guide = load_raw_dir(DATA_RAW_DIR)

    # 1. Assert exactly 3 transcripts ordered T1, T2, T3
    assert len(transcripts) == 3
    assert [t.id for t in transcripts] == ["T1", "T2", "T3"]

    t1, t2, t3 = transcripts[0], transcripts[1], transcripts[2]

    # 2. Assert each transcript has 14 turns (7 interviewer, 7 expert)
    for t in (t1, t2, t3):
        assert len(t.turns) == 14
        interviewer_turns = [turn for turn in t.turns if turn.role == "interviewer"]
        expert_turns = [turn for turn in t.turns if turn.role == "expert"]
        assert len(interviewer_turns) == 7
        assert len(expert_turns) == 7

    # 3. Assert specific properties on T1
    assert t1.turns[3].ts == "01:20"
    assert t1.turns[3].role == "expert"
    assert t1.turns[5].text.startswith("Very important.")

    # 4. Assert specific properties on T2 (Anna Keller is NOT a doctor)
    assert t2.expert_name == "Anna Keller"
    assert t2.market == "Germany"

    # 5. Assert specific properties on T3
    assert t3.expert_role == "Consultant Urologist"

    # 6. Assert guide structure: 6 questions, en-dash preserved in Q5
    assert len(guide.questions) == 6
    q5 = next(q for q in guide.questions if q.id == 5)
    assert "3–5 years" in q5.text


def test_synthetic_colon_inside_sentence(tmp_path: Path) -> None:
    """Verify a colon inside an expert's sentence does not break turn parsing."""
    raw_content = (
        "Expert 1 – Dr. Jean Martin\n"
        "Role: Head of Urology\n"
        "Market: France\n\n"
        "00:10\n"
        "Interviewer: Can you explain the ratio?\n\n"
        "00:25\n"
        "Dr. Martin: The ratio is simple: 3 to 1 in public hospitals, and note: costs are high.\n"
    )
    fake_file = tmp_path / "Transcript_1_France.txt"
    fake_file.write_text(raw_content, encoding="utf-8")

    parsed = parse_transcript(fake_file)
    assert len(parsed.turns) == 2
    assert parsed.turns[1].speaker_label == "Dr. Martin"
    assert parsed.turns[1].text == "The ratio is simple: 3 to 1 in public hospitals, and note: costs are high."


def test_out_of_order_timestamps_raise_parse_error(tmp_path: Path) -> None:
    """Verify that a decreasing timestamp triggers ParseError with filename and line number."""
    bad_content = (
        "Expert 1 – Dr. Jean Martin\n"
        "Role: Head of Urology\n"
        "Market: France\n\n"
        "01:00\n"
        "Interviewer: First question?\n\n"
        "00:45\n"
        "Dr. Martin: Out of order response.\n"
    )
    fake_file = tmp_path / "Bad_Transcript.txt"
    fake_file.write_text(bad_content, encoding="utf-8")

    with pytest.raises(ParseError) as exc_info:
        parse_transcript(fake_file)

    assert "Out-of-order timestamp" in str(exc_info.value)
    assert "Bad_Transcript.txt" in str(exc_info.value)