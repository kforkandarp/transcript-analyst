"""End-to-end ingestion, synthesis pipeline orchestration, and cache validation."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from analyst.chunker import build_exchanges
from analyst.config import settings
from analyst.guide import answer_guide, split_questions
from analyst.index import IndexStore
from analyst.models import Guide, GuideCell, Theme, Transcript
from analyst.parser import load_raw_dir
from analyst.prompts import PROMPT_VERSION
from analyst.store import (
    compute_raw_content_hash,
    load_guide,
    load_guide_cells,
    load_meta,
    load_themes,
    load_transcripts,
)
from analyst.themes import build_themes

logger = logging.getLogger("analyst.pipeline")


def load_corpus(raw_dir: Path) -> Tuple[list[Transcript], Guide]:
    """Parse raw transcripts and interview guide from raw directory."""
    return load_raw_dir(raw_dir)


def build_store(transcripts: Sequence[Transcript]) -> IndexStore:
    """Build, chunk, and embed all exchanges into an in-memory IndexStore."""
    store = IndexStore()
    for t in transcripts:
        exchanges = build_exchanges(t)
        store.add_transcript(exchanges)
    return store


def content_hash(raw_dir: Path) -> str:
    """Calculate the deterministic SHA-256 hash of all text files in the raw directory."""
    return compute_raw_content_hash(raw_dir)


def cache_is_fresh(meta: Optional[Dict[str, Any]], raw_dir: Path) -> bool:
    """Check if cached metadata matches raw content hash, PROMPT_VERSION, and active models."""
    if not meta:
        return False

    current_hash = content_hash(raw_dir)
    if meta.get("content_hash") != current_hash:
        return False

    if meta.get("prompt_version") != PROMPT_VERSION:
        return False

    models = meta.get("models", {})
    if models.get("small") != settings.llm_small_model:
        return False
    if models.get("large") != settings.llm_large_model:
        return False
    if models.get("embed") != settings.embed_model_name:
        return False

    return True


def run_all(
    raw_dir: Path,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Execute the full analytical pipeline from raw transcripts to verified themes.

    Args:
        raw_dir: Path containing raw transcript and guide files.
        progress: Optional callback reporting stage description and fraction completed (0.0 to 1.0).

    Returns:
        Dict containing transcripts, split guide, guide_cells, themes, and metadata.
    """
    def _report(stage: str, frac: float) -> None:
        logger.info("Pipeline progress [%.1f%%]: %s", frac * 100, stage)
        if progress:
            progress(stage, frac)

    _report("Loading raw corpus", 0.05)
    transcripts, guide = load_corpus(raw_dir)

    _report("Building vector index", 0.15)
    store = build_store(transcripts)

    _report("Splitting guide questions", 0.25)
    split_g = split_questions(guide)

    _report("Answering guide matrix", 0.35)

    def _guide_progress(done: int, total: int) -> None:
        if total > 0:
            frac = 0.35 + (done / total) * 0.40  # 35% to 75%
            _report(f"Answering guide cells ({done}/{total})", frac)

    guide_cells = answer_guide(store, split_g, callback=_guide_progress, max_workers=4)

    _report("Extracting themes (Map-Reduce)", 0.75)
    themes = build_themes(store, transcripts)

    _report("Finalizing metadata", 0.95)
    meta = {
        "prompt_version": PROMPT_VERSION,
        "models": {
            "small": settings.llm_small_model,
            "large": settings.llm_large_model,
            "embed": settings.embed_model_name,
        },
        "content_hash": content_hash(raw_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _report("Pipeline completed", 1.0)

    return {
        "transcripts": transcripts,
        "guide": split_g,
        "guide_cells": guide_cells,
        "themes": themes,
        "meta": meta,
        "store": store,
    }


def load_cached(processed_dir: Path, raw_dir: Path) -> Optional[Dict[str, Any]]:
    """Load preprocessed artifacts if cache is present, valid, and fresh.

    Returns:
        Dict with keys (transcripts, guide, guide_cells, themes, meta, store) or None.
    """
    meta = load_meta(processed_dir)
    if not cache_is_fresh(meta, raw_dir):
        return None

    transcripts = load_transcripts(processed_dir)
    guide = load_guide(processed_dir)
    guide_cells = load_guide_cells(processed_dir)
    themes = load_themes(processed_dir)

    if None in (transcripts, guide, guide_cells, themes):
        return None

    store = build_store(transcripts)  # type: ignore[arg-type]

    return {
        "transcripts": transcripts,
        "guide": guide,
        "guide_cells": guide_cells,
        "themes": themes,
        "meta": meta,
        "store": store,
    }