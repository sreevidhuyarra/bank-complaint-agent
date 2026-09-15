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

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq's free-tier lineup changes over time — llama-3.3-70b-versatile has since
# been retired. gpt-oss-120b is the current largest model with solid tool-calling
# support, which the handoff pattern depends on; gpt-oss-20b is a faster/lighter
# fallback if the 120B model gets rate-limited. Check what a given key can
# actually reach with `GET {GROQ_BASE_URL}/models` before assuming either name
# is still current.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def groq_api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


for _d in (CACHE_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
