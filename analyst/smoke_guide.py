"""Smoke test script for verifying question splitting and full guide matrix synthesis."""

from __future__ import annotations

import logging
from pathlib import Path

from analyst.chunker import build_exchanges
from analyst.guide import answer_guide, split_questions
from analyst.index import IndexStore
from analyst.parser import load_raw_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.smoke_guide")

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def progress_callback(done: int, total: int) -> None:
    print(f"Progress: {done}/{total} cells completed...")


def main() -> None:
    # 1. Load raw transcripts and guide
    transcripts, guide = load_raw_dir(DATA_RAW_DIR)

    # 2. Build IndexStore and index all 3 transcripts
    store = IndexStore()
    for t in transcripts:
        exchanges = build_exchanges(t)
        store.add_transcript(exchanges)

    print(f"Indexed transcripts: {store.transcript_ids()}")

    # 3. Split questions and display results
    print("\n--- Splitting Guide Questions ---")
    split_guide = split_questions(guide)
    for q in split_guide.questions:
        print(f"\nQ{q.id}: {q.text}")
        print(f"  Sub-questions ({len(q.sub_questions)}):")
        for sq in q.sub_questions:
            print(f"    - {sq}")

    # 4. Answer guide matrix across all questions and transcripts
    print("\n--- Generating Guide Answer Matrix ---")
    cells = answer_guide(store, split_guide, callback=progress_callback, max_workers=4)

    # 5. Report per-cell results and totals
    print("\n--- Synthesis Summary ---")
    unverified_count = 0

    for cell in cells:
        # Determine overall cell coverage & total claim count across all parts
        coverages = [part.result.coverage for part in cell.parts]
        total_claims = sum(len(part.result.claims) for part in cell.parts)

        # A cell is unverified if any part is unverified
        cell_coverage = "unverified" if "unverified" in coverages else coverages[0]
        if cell_coverage == "unverified":
            unverified_count += 1

        print(
            f"Question {cell.question_id:2d} | Transcript {cell.transcript_id} | "
            f"Coverage: {cell_coverage:<12} | Claims: {total_claims}"
        )

    print("\n=======================================================")
    print(f"Total cells produced: {len(cells)} (expected: 18)")
    print(f"Unverified cells:     {unverified_count}")
    print("=======================================================")


if __name__ == "__main__":
    main()