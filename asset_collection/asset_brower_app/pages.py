from __future__ import annotations

from functools import lru_cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"


@lru_cache(maxsize=None)
def page_text(filename: str) -> str:
    return (STATIC_DIR / filename).read_text(encoding="utf-8")
