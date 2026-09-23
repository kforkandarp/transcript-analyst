"""Parser for raw interview transcripts and interview guides."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple, Union

from analyst.models import Guide, GuideQuestion, Transcript, Turn

logger = logging.getLogger("analyst.parser")

TIMESTAMP_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


class ParseError(Exception):
    """Raised when parsing fails due to malformed syntax or violated constraints."""

    def __init__(self, message: str, file_name: str = "<unknown>", line_number: int = 0) -> None:
        self.file_name = file_name
        self.line_number = line_number
        formatted_message = f"[{file_name}:{line_number}] {message}"
        super().__init__(formatted_message)


def _timestamp_to_seconds(ts: str, file_name: str, line_number: int) -> int:
    """Convert an MM:SS or HH:MM:SS string to total seconds."""
    match = TIMESTAMP_PATTERN.match(ts.strip())
    if not match:
        raise ParseError(f"Invalid timestamp format: '{ts}'", file_name, line_number)
    parts = [int(p) for p in match.groups() if p is not None]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _read_normalized(path: Union[str, Path]) -> tuple[str, str]:
    """Read a text file with utf-8-sig encoding and normalize CRLF to LF."""
    p = Path(path)
    if not p.is_file():
        raise ParseError(f"File not found: {p}", str(p), 0)
    try:
        content = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(f"File encoding error: {exc}", p.name, 0) from exc
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, p.name


def parse_guide(path: Union[str, Path]) -> Guide:
    """Parse the interview guide text file."""
    content, file_name = _read_normalized(path)
    lines = content.split("\n")

    title = ""
    objective = ""
    questions: List[GuideQuestion] = []

    in_questions = False
    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if not title:
            title = line
            continue

        if line.lower().startswith("project objective:"):
            objective = line.split(":", 1)[1].strip()
            continue

        if line.lower().startswith("questions:"):
            in_questions = True
            continue

        if in_questions:
            q_match = re.match(r"^(\d+)\.\s*(.+)$", line)
            if q_match:
                q_id = int(q_match.group(1))
                q_text = q_match.group(2).strip()
                questions.append(GuideQuestion(id=q_id, text=q_text))

    if not title:
        raise ParseError("Missing guide title", file_name, 1)
    if not questions:
        raise ParseError("No questions found under 'Questions:'", file_name, len(lines))

    return Guide(title=title, objective=objective, questions=questions)


def parse_transcript(path: Union[str, Path], transcript_id: str | None = None) -> Transcript:
    """Parse an interview transcript into structured metadata and turns."""
    content, file_name = _read_normalized(path)
    lines = content.split("\n")

    expert_name = ""
    expert_role = ""
    market = ""
    header_expert_num = ""

    line_idx = 0
    total_lines = len(lines)

    # 1. Parse Header
    while line_idx < total_lines:
        raw_line = lines[line_idx].strip()
        line_idx += 1

        if not raw_line:
            if expert_name and expert_role and market:
                break
            continue

        exp_match = re.match(r"^Expert\s+(\d+)\s*[-–—]\s*(.+)$", raw_line, re.IGNORECASE)
        if exp_match:
            header_expert_num = exp_match.group(1).strip()
            expert_name = exp_match.group(2).strip()
            continue

        role_match = re.match(r"^Role:\s*(.+)$", raw_line, re.IGNORECASE)
        if role_match:
            expert_role = role_match.group(1).strip()
            continue

        mkt_match = re.match(r"^Market:\s*(.+)$", raw_line, re.IGNORECASE)
        if mkt_match:
            market = mkt_match.group(1).strip()
            continue

    if not expert_name:
        raise ParseError("Missing expert name in header", file_name, line_idx)
    if not expert_role:
        raise ParseError("Missing expert role in header", file_name, line_idx)
    if not market:
        raise ParseError("Missing market in header", file_name, line_idx)

    assigned_id = transcript_id or (f"T{header_expert_num}" if header_expert_num else "T1")

    # 2. Parse Timestamped Blocks
    turns: List[Turn] = []
    current_ts: str | None = None
    current_speaker: str | None = None
    current_text_lines: List[str] = []
    current_ts_line_num = 0
    last_seconds = -1

    def flush_turn() -> None:
        nonlocal current_ts, current_speaker, current_text_lines, current_ts_line_num, last_seconds
        if current_ts is None:
            return

        text = " ".join(l.strip() for l in current_text_lines if l.strip())
        if not text:
            raise ParseError("Turn text cannot be empty", file_name, current_ts_line_num)
        if not current_speaker:
            raise ParseError("Missing speaker label for turn", file_name, current_ts_line_num)

        seconds = _timestamp_to_seconds(current_ts, file_name, current_ts_line_num)
        if seconds < last_seconds:
            raise ParseError(
                f"Out-of-order timestamp: {current_ts} ({seconds}s) after {last_seconds}s",
                file_name,
                current_ts_line_num,
            )
        last_seconds = seconds

        role = "interviewer" if current_speaker.strip().lower() == "interviewer" else "expert"

        turns.append(
            Turn(
                idx=len(turns),
                ts=current_ts,
                seconds=seconds,
                speaker_label=current_speaker,
                role=role,
                text=text,
            )
        )

        current_ts = None
        current_speaker = None
        current_text_lines = []

    while line_idx < total_lines:
        line_num = line_idx + 1
        raw_line = lines[line_idx]
        line = raw_line.strip()
        line_idx += 1

        if not line:
            continue

        ts_match = TIMESTAMP_PATTERN.match(line)
        if ts_match:
            flush_turn()
            current_ts = line
            current_ts_line_num = line_num
            continue

        # If we have an active timestamp but no speaker yet, extract speaker from start of line
        if current_ts is not None and current_speaker is None:
            if ": " in raw_line:
                candidate_label, rest = raw_line.split(": ", 1)
                candidate_words = candidate_label.strip().split()
                # A speaker label must be 'Interviewer' or a plausible name (max 4 words)
                if 1 <= len(candidate_words) <= 4:
                    current_speaker = candidate_label.strip()
                    if rest.strip():
                        current_text_lines.append(rest.strip())
                    continue

            raise ParseError(
                f"Expected speaker label formatted as 'Speaker: text', got '{line}'",
                file_name,
                line_num,
            )

        # Multi-line turn text continuation
        if current_ts is not None:
            current_text_lines.append(line)

    flush_turn()

    # 3. Validate Turn Consistency
    if not turns:
        raise ParseError("No conversation turns found", file_name, line_idx)

    has_interviewer = any(t.role == "interviewer" for t in turns)
    has_expert = any(t.role == "expert" for t in turns)

    if not has_interviewer or not has_expert:
        raise ParseError(
            f"Transcript must have at least one interviewer and one expert turn (interviewer={has_interviewer}, expert={has_expert})",
            file_name,
            line_idx,
        )

    return Transcript(
        id=assigned_id,
        source_file=file_name,
        expert_name=expert_name,
        expert_role=expert_role,
        market=market,
        turns=turns,
    )


def load_raw_dir(raw_dir: Union[str, Path]) -> Tuple[List[Transcript], Guide]:
    """Load all transcripts and the interview guide from the raw data folder."""
    p_dir = Path(raw_dir)
    if not p_dir.is_dir():
        raise ParseError(f"Directory not found: {p_dir}", str(p_dir), 0)

    # 1. Locate and parse guide
    guide_candidates = [
        f for f in p_dir.glob("*.txt") if "guide" in f.name.lower()
    ]
    if not guide_candidates:
        raise ParseError("No interview guide file found matching '*guide*.txt'", str(p_dir), 0)

    guide = parse_guide(guide_candidates[0])

    # 2. Locate and parse transcripts
    transcript_files = [
        f for f in p_dir.glob("*.txt") if "transcript" in f.name.lower()
    ]

    transcripts: List[Transcript] = []
    errors: List[str] = []

    for t_file in transcript_files:
        try:
            transcript = parse_transcript(t_file)
            transcripts.append(transcript)
        except ParseError as pe:
            logger.error("Failed to parse transcript file %s: %s", t_file.name, pe)
            errors.append(str(pe))
        except Exception as exc:
            logger.error("Unexpected error parsing %s: %s", t_file.name, exc)
            errors.append(f"[{t_file.name}] {exc}")

    # Sort numerically by ID (e.g. T1, T2 ... T10, T11) rather than lexicographically
    transcripts.sort(key=lambda t: int(t.id[1:]) if t.id.startswith("T") and t.id[1:].isdigit() else t.id)

    if errors:
        logger.warning("Encountered %d errors while loading raw transcripts:\n%s", len(errors), "\n".join(errors))

    return transcripts, guide