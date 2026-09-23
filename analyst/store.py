"""Processed artifact storage with atomic JSON serialization and integrity checks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from analyst.config import settings
from analyst.models import Guide, GuideCell, Theme, Transcript
from analyst.prompts import PROMPT_VERSION

logger = logging.getLogger("analyst.store")


def compute_raw_content_hash(raw_dir: Path) -> str:
    """Compute a single deterministic SHA256 hash over all files in the raw directory."""
    hasher = hashlib.sha256()
    if not raw_dir.exists():
        return ""

    for file_path in sorted(raw_dir.glob("*.txt")):
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(file_path.read_bytes())

    return hasher.hexdigest()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write data to a temporary file in the target directory and atomically replace destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix="tmp_", suffix=".json")
    try:
        with open(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_path).replace(path)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


def _safe_load_json(path: Path) -> Optional[Any]:
    """Safely load JSON data; return None if the file is missing or corrupted."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed loading %s (corrupt or invalid JSON): %s", path, exc)
        return None


# --- Save Methods ---

def save_transcripts(transcripts: Sequence[Transcript], processed_dir: Path) -> None:
    path = processed_dir / "transcripts.json"
    data = [t.model_dump() for t in transcripts]
    _atomic_write_json(path, data)


def save_guide(guide: Guide, processed_dir: Path) -> None:
    path = processed_dir / "guide.json"
    _atomic_write_json(path, guide.model_dump())


def save_guide_cells(cells: Sequence[GuideCell], processed_dir: Path) -> None:
    path = processed_dir / "guide_cells.json"
    data = [c.model_dump() for c in cells]
    _atomic_write_json(path, data)


def save_themes(themes: Sequence[Theme], processed_dir: Path) -> None:
    path = processed_dir / "themes.json"
    data = [t.model_dump() for t in themes]
    _atomic_write_json(path, data)


def save_meta(raw_dir: Path, processed_dir: Path) -> None:
    path = processed_dir / "meta.json"
    meta_payload = {
        "prompt_version": PROMPT_VERSION,
        "models": {
            "small": settings.llm_small_model,
            "large": settings.llm_large_model,
            "embed": settings.embed_model_name,
        },
        "content_hash": compute_raw_content_hash(raw_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(path, meta_payload)


# --- Load Methods ---

def load_transcripts(processed_dir: Path) -> Optional[List[Transcript]]:
    raw = _safe_load_json(processed_dir / "transcripts.json")
    if raw is None or not isinstance(raw, list):
        return None
    try:
        return [Transcript.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Failed validating transcripts.json: %s", exc)
        return None


def load_guide(processed_dir: Path) -> Optional[Guide]:
    raw = _safe_load_json(processed_dir / "guide.json")
    if raw is None or not isinstance(raw, dict):
        return None
    try:
        return Guide.model_validate(raw)
    except Exception as exc:
        logger.warning("Failed validating guide.json: %s", exc)
        return None


def load_guide_cells(processed_dir: Path) -> Optional[List[GuideCell]]:
    raw = _safe_load_json(processed_dir / "guide_cells.json")
    if raw is None or not isinstance(raw, list):
        return None
    try:
        return [GuideCell.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Failed validating guide_cells.json: %s", exc)
        return None


def load_themes(processed_dir: Path) -> Optional[List[Theme]]:
    raw = _safe_load_json(processed_dir / "themes.json")
    if raw is None or not isinstance(raw, list):
        return None
    try:
        return [Theme.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Failed validating themes.json: %s", exc)
        return None


def load_meta(processed_dir: Path) -> Optional[Dict[str, Any]]:
    raw = _safe_load_json(processed_dir / "meta.json")
    if raw is None or not isinstance(raw, dict):
        return None
    return raw