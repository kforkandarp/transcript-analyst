"""CLI build tool for precomputing and persisting pipeline artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict

from analyst.pipeline import cache_is_fresh, run_all
from analyst.store import (
    load_meta,
    save_guide,
    save_guide_cells,
    save_meta,
    save_themes,
    save_transcripts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analyst.build_cache")

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache transcript analysis artifacts.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full pipeline re-run even if cache is fresh.",
    )
    args = parser.parse_args()

    meta = load_meta(DATA_PROCESSED_DIR)
    if not args.force and cache_is_fresh(meta, DATA_RAW_DIR):
        print("\n=======================================================")
        print("Cache is fresh. All artifacts are up to date.")
        print(f"Timestamp:    {meta.get('timestamp')}")
        print(f"Content hash: {meta.get('content_hash')}")
        print("Use --force to rebuild.")
        print("=======================================================\n")
        return

    print("\n=======================================================")
    print("Building pipeline cache from raw data...")
    if args.force:
        print("Flag --force specified: bypassing cache freshness check.")
    print("=======================================================\n")

    start_time = time.time()

    def _cli_progress(stage: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {stage}")

    results = run_all(DATA_RAW_DIR, progress=_cli_progress)

    # 1. Persist all artifacts atomically
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_transcripts(results["transcripts"], DATA_PROCESSED_DIR)
    save_guide(results["guide"], DATA_PROCESSED_DIR)
    save_guide_cells(results["guide_cells"], DATA_PROCESSED_DIR)
    save_themes(results["themes"], DATA_PROCESSED_DIR)
    save_meta(DATA_RAW_DIR, DATA_PROCESSED_DIR)

    elapsed_s = time.time() - start_time

    # 2. Compute summary metrics
    total_raw_quotes = 0
    total_verified_quotes = 0
    total_dropped_quotes = 0
    total_retries = 0

    for cell in results["guide_cells"]:
        for part in cell.parts:
            stats = part.result.stats
            total_raw_quotes += stats.get("raw_quotes", 0)
            total_verified_quotes += stats.get("verified_quotes", 0)
            total_dropped_quotes += stats.get("dropped_quotes", 0)
            total_retries += stats.get("retries", 0)

    # Guide questions use 1 small call each for splitting
    guide_split_calls = len(results["guide"].questions)
    # Each guide part uses 1 large call (plus retries)
    guide_part_count = sum(len(cell.parts) for cell in results["guide_cells"])
    guide_large_calls = guide_part_count + total_retries
    # Themes map phase uses 1 large call per transcript (+ possible retries) + 1 reduce call
    theme_large_calls = len(results["transcripts"]) + 1

    print("\n=======================================================")
    print("CACHE BUILD SUMMARY")
    print("=======================================================")
    print(f"Transcripts processed:   {len(results['transcripts'])}")
    print(f"Guide questions split:   {len(results['guide'].questions)}")
    print(f"Guide cells generated:   {len(results['guide_cells'])}")
    print(f"Themes extracted:        {len(results['themes'])}")
    print(f"Estimated Small calls:   {guide_split_calls}")
    print(f"Estimated Large calls:   {guide_large_calls + theme_large_calls}")
    print(f"Verification Quotes:     {total_verified_quotes}/{total_raw_quotes} verified "
          f"({total_dropped_quotes} dropped, {total_retries} retries)")
    print(f"Total elapsed time:      {elapsed_s:.2f} seconds")
    print(f"Artifacts written to:    {DATA_PROCESSED_DIR.resolve()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()