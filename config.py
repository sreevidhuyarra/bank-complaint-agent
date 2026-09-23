"""Central configuration: paths, dataset scope, and model identifiers.

Everything the rest of the project needs to know about *where* things live and
*what* the default scope is lives here, so the scope can be widened without
touching agent or NLP code.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CACHE_DIR = ROOT / "cache"
INDEX_DIR = ROOT / "index"

RAW_COMPLAINTS_CSV = CACHE_DIR / "complaints_raw.csv"
CLEAN_COMPLAINTS_CSV = INDEX_DIR / "complaints_clean.csv"
FAISS_INDEX_PATH = INDEX_DIR / "complaints.faiss"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

# --- dataset scope -----------------------------------------------------------
# CFPB stores companies under their exact legal names. These are the strings the
# API expects; `python -m data.fetch_cfpb --list-companies "wells"` resolves more.
TARGET_COMPANIES = [
    "WELLS FARGO & COMPANY",
    "AMERICAN EXPRESS COMPANY",
    "CHARLES SCHWAB CORPORATION, THE",
]

# Trailing four quarters, per the build spec. Kept as a function so a stale
# checkout doesn't silently fetch a stale window.
TRAILING_QUARTERS = 4


def default_date_range(today: date | None = None) -> tuple[str, str]:
    """Return (date_min, date_max) as ISO dates covering the trailing 4 quarters."""
    today = today or date.today()
    months_back = 3 * TRAILING_QUARTERS
    year = today.year
    month = today.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    start = date(year, month, 1)
    return start.isoformat(), today.isoformat()


# Complaints shorter than this are dropped: the CFPB narrative field is often a
# one-line stub that neither semantic search nor the risk scorer can use.
MIN_NARRATIVE_WORDS = 50

# Safety cap per company. The fetcher walks months newest-first, so a cap that
# bites truncates the *oldest* months — which would quietly break the
# quarter-over-quarter comparison. Keep it above the real volume of the busiest
# company in scope (Wells Fargo runs ~7k narratives per year) rather than using
# it to shrink the index.
MAX_COMPLAINTS_PER_COMPANY = 8000

# --- models ------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# gemini-2.5-flash-lite was the first default here and was retired for new
# accounts almost immediately — confirmed live: a real key got back
# `404 ... "This model models/gemini-2.5-flash-lite is no longer available to
# new users. Please update your code to use models/gemini-3.5-flash-lite"`,
# straight from Google's own error message. That replacement is now the
# default, but the same caution applies to it in turn — list your account's
# actual available models at https://aistudio.google.com or via
# `client.models.list()` before assuming this hardcoded name stays current.
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash-lite")


def google_api_key() -> str | None:
    return os.environ.get("GOOGLE_AI_API_KEY") or None


for _d in (CACHE_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
