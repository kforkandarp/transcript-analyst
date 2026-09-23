"""Smoke test script for verifying end-to-end answer generation and verification."""

from __future__ import annotations

import logging
from pathlib import Path

from analyst.answer import generate_verified_answer
from analyst.chunker import build_exchanges
from analyst.index import IndexStore, retrieve
from analyst.parser import load_raw_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.smoke_answer")

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def run() -> None:
    # 1. Load transcripts and guide
    transcripts, guide = load_raw_dir(DATA_RAW_DIR)
    t1 = next(t for t in transcripts if t.id == "T1")
    q6 = next(q for q in guide.questions if q.id == 6)

    # 2. Build index for T1
    store = IndexStore()
    exchanges_t1 = build_exchanges(t1)
    store.add_transcript(exchanges_t1)

    # 3. Retrieve relevant chunks for Question 6
    retrieved = retrieve(
        store=store,
        queries=[q6.text],
        transcript_ids=["T1"],
        k_per_query=3,
        neighbours=1,
        max_chunks=6,
    )

    print(f"\n=======================================================")
    print(f"Guide Question 6: {q6.text}")
    print(f"Retrieved {len(retrieved['T1'])} chunks from T1: {[ex.chunk_id for ex in retrieved['T1']]}")
    print(f"=======================================================\n")

    # 4. Generate verified answer
    answer = generate_verified_answer(
        question=q6.text,
        sub_question=None,
        exchanges_by_tid=retrieved,
        mode="guide",
    )

    print(f"Coverage: {answer.coverage}")
    if answer.note:
        print(f"Note:     {answer.note}")
    print(f"Stats:    {answer.stats}\n")

    print(f"Claims ({len(answer.claims)}):")
    for i, claim in enumerate(answer.claims, start=1):
        print(f"  {i}. {claim.text}")
        for ev in claim.evidence:
            print(f"     -> [{ev.chunk_id} @ {ev.ts}] \"{ev.quote}\"")


if __name__ == "__main__":
    run()