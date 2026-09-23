"""Application settings and environment loading."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analyst.config")

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Application settings and operational paths."""

    llm_small_model: str = field(
        default_factory=lambda: os.getenv("LLM_SMALL_MODEL", "openai/gpt-oss-20b")
    )
    llm_large_model: str = field(
        default_factory=lambda: os.getenv("LLM_LARGE_MODEL", "openai/gpt-oss-120b")
    )
    embed_model_name: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )

    groq_api_keys: List[str] = field(default_factory=list)
    app_password: str = field(
        default_factory=lambda: os.getenv("APP_PASSWORD", "")
    )

    root_dir: Path = REPO_ROOT
    data_raw_dir: Path = REPO_ROOT / "data" / "raw"
    data_processed_dir: Path = REPO_ROOT / "data" / "processed"
    data_gold_dir: Path = REPO_ROOT / "data" / "gold"
    eval_results_dir: Path = REPO_ROOT / "eval" / "results"

    @property
    def has_groq_keys(self) -> bool:
        """Check if at least one Groq API key is configured."""
        return len(self.groq_api_keys) > 0


def load_settings() -> Settings:
    """Instantiate settings with parsed API keys."""
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    parsed_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not parsed_keys:
        logger.warning("No Groq API keys found in environment.")

    return Settings(groq_api_keys=parsed_keys)


settings = load_settings()