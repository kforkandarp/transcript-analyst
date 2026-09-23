"""Data contracts for transcript parsing, indexing, validation, and synthesis."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single speaker turn in a transcript."""

    idx: int
    ts: str
    seconds: int
    speaker_label: str
    role: Literal["interviewer", "expert"]
    text: str


class Transcript(BaseModel):
    """Complete parsed transcript for one interview."""

    id: str
    source_file: str
    expert_name: str
    expert_role: str
    market: str
    turns: list[Turn]


class Exchange(BaseModel):
    """A single interview exchange combining question turn(s) with expert answer turn(s)."""

    chunk_id: str  # e.g., 'T1-X3'
    transcript_id: str
    expert_name: str
    expert_role: str
    market: str
    question_turns: list[Turn]
    answer_turns: list[Turn]

    @property
    def question_text(self) -> str:
        """Concatenated text of interviewer questions."""
        return " ".join(t.text.strip() for t in self.question_turns if t.text.strip())

    @property
    def answer_text(self) -> str:
        """Concatenated text of expert answers (the quotable surface)."""
        return " ".join(t.text.strip() for t in self.answer_turns if t.text.strip())

    @property
    def embed_text(self) -> str:
        """Passage embedding text representation."""
        return (
            f"Expert: {self.expert_name}, {self.expert_role}, {self.market}\n"
            f"Interviewer asked: {self.question_text}\n"
            f"Expert answered: {self.answer_text}"
        )


class GuideQuestion(BaseModel):
    """Single interview guide question with optional sub-questions."""

    id: int
    text: str
    sub_questions: list[str] = Field(default_factory=list)


class Guide(BaseModel):
    """Interview guide containing metadata and numbered questions."""

    title: str
    objective: str
    questions: list[GuideQuestion]


class RawEvidence(BaseModel):
    """Evidence quote returned by the LLM prior to verification."""

    chunk_id: str
    quote: str


class RawClaim(BaseModel):
    """Claim with raw unverified quotes returned by the LLM."""

    text: str
    evidence: list[RawEvidence]


class RawAnswer(BaseModel):
    """Structured LLM synthesis prior to deterministic quote verification."""

    coverage: Literal["direct", "indirect", "not_discussed"]
    claims: list[RawClaim] = []
    note: str = ""


class Evidence(BaseModel):
    """Verified exact-slice evidence with authoritative timestamps."""

    chunk_id: str
    transcript_id: str
    expert_name: str
    quote: str
    ts: str
    question_ts: Optional[str] = None


class Claim(BaseModel):
    """Verified claim backed by deterministically checked quotes."""

    text: str
    evidence: list[Evidence]


class AnswerResult(BaseModel):
    """Complete synthesized answer verified against transcript turns."""

    coverage: Literal["direct", "indirect", "not_discussed", "unverified"]
    claims: list[Claim] = []
    note: str = ""
    stats: dict = Field(default_factory=dict)


class GuidePart(BaseModel):
    """Result for a specific guide sub-question."""

    sub_question: str
    result: AnswerResult


class GuideCell(BaseModel):
    """Grid matrix cell for a given question and transcript."""

    question_id: int
    transcript_id: str
    parts: list[GuidePart]


class ThemePosition(BaseModel):
    """Individual expert position supporting or dissenting a theme."""

    transcript_id: str
    expert_name: str
    market: str
    position: str
    evidence: list[Evidence]
    value: Optional[str] = None
    scope: Optional[str] = None
    qualifier: Optional[str] = None


class Theme(BaseModel):
    """Cross-transcript synthesis theme (common, disagreement, or unique)."""

    title: str
    kind: Literal["common", "disagreement", "unique"]
    verdict: Optional[Literal["agree", "partly_agree", "disagree", "not_comparable"]] = None
    summary: str
    positions: list[ThemePosition]
    expert_count: int = 0