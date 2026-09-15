"""Clean the raw CFPB pull, fit the corpus LM, embed, and build the FAISS index.

Also hosts `ComplaintIndex`, the read side that every query path uses — the
Retrieval agent's tool, the Trend agent's aggregations, and the dashboard.

Usage
-----
    python -m data.ingest                  # build from cache/complaints_raw.csv
    python -m data.ingest --rebuild        # ignore existing artefacts
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLEAN_COMPLAINTS_CSV,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    MIN_NARRATIVE_WORDS,
    RAW_COMPLAINTS_CSV,
)
from nlp.linguistic_risk import (
    LinguisticRiskScorer,
    NgramLM,
    SeverityBands,
    SlorNorm,
    normalize,
)

LM_PATH = INDEX_DIR / "corpus_lm.json.gz"
META_PATH = INDEX_DIR / "index_meta.json"

# Below this share of the corpus surviving a metadata filter, an exact scan of
# the filtered subset beats over-fetching from FAISS and throwing most of it away.
EXACT_SCAN_THRESHOLD = 0.25


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable narratives and normalise the columns downstream code reads."""
    frame = frame.copy()
    frame["narrative"] = frame["complaint_what_happened"].fillna("").map(normalize)
    frame["word_count"] = frame["narrative"].str.split().str.len().fillna(0).astype(int)

    before = len(frame)
    frame = frame[frame["word_count"] >= MIN_NARRATIVE_WORDS]
    print(f"  dropped {before - len(frame):,} narratives under {MIN_NARRATIVE_WORDS} words")

    frame = frame.drop_duplicates(subset="complaint_id")
    frame = frame.drop_duplicates(subset="narrative")

    frame["date_received"] = pd.to_datetime(
        frame["date_received"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    frame = frame.dropna(subset=["date_received"])
    frame["month"] = frame["date_received"].dt.to_period("M").astype(str)
    frame["quarter"] = frame["date_received"].dt.to_period("Q").astype(str)

    for column in ("company", "product", "sub_product", "issue", "sub_issue", "state"):
        frame[column] = frame[column].fillna("").astype(str)

    frame = frame.sort_values("date_received", ascending=False).reset_index(drop=True)
    return frame.drop(columns=["complaint_what_happened"])


def build(raw_path: Path = RAW_COMPLAINTS_CSV, reuse_embeddings: bool = False) -> pd.DataFrame:
    if not raw_path.exists():
        raise SystemExit(
            f"No raw pull at {raw_path}. Run `python -m data.fetch_cfpb` first."
        )

    print(f"Reading {raw_path}")
    frame = clean(pd.read_csv(raw_path))
    print(f"  {len(frame):,} usable complaints")
    if frame.empty:
        raise SystemExit("Nothing left after cleaning — widen the fetch scope.")

    # 1. Fit the corpus bigram LM that SLOR is measured against.
    print("Fitting corpus n-gram LM ...")
    lm = NgramLM().fit(frame["narrative"])
    lm.save(LM_PATH)
    print(f"  vocab {lm.vocab_size:,} · {lm.total_tokens:,} tokens → {LM_PATH.name}")

    # 2. Calibrate SLOR against this corpus, then pre-score every narrative.
    slor_values = [s for s in (lm.slor(t) for t in frame["narrative"]) if s is not None]
    norm = SlorNorm(
        mean=statistics.fmean(slor_values),
        sd=statistics.pstdev(slor_values) or 1.0,
    )
    print(f"  SLOR mean {norm.mean:.3f} sd {norm.sd:.3f}")

    print("Scoring linguistic risk ...")
    scorer = LinguisticRiskScorer(lm=lm, slor_norm=norm)
    scores = scorer.score_many(frame["narrative"])
    frame["risk_score"] = [s.risk_score for s in scores]
    frame["risk_drivers"] = ["; ".join(s.drivers) for s in scores]

    # Severity bands are calibrated against this corpus — see SeverityBands.
    bands = SeverityBands.from_scores(frame["risk_score"].tolist())
    frame["severity"] = [bands.label(s) for s in frame["risk_score"]]
    frame["risk_percentile"] = [bands.percentile(s) for s in frame["risk_score"]]
    print(f"  bands calibrated: medium ≥ {bands.medium}, high ≥ {bands.high}")
    print(f"  score distribution: "
          f"p50 {frame['risk_score'].median():.1f} · "
          f"p90 {frame['risk_score'].quantile(0.9):.1f} · "
          f"max {frame['risk_score'].max():.1f}")
    print("  " + frame["severity"].value_counts().to_string().replace("\n", "\n  "))

    # 3. Embed and index.
    from nlp.embeddings import embed  # local import keeps torch out of light paths

    if reuse_embeddings and EMBEDDINGS_PATH.exists():
        vectors = np.load(EMBEDDINGS_PATH)
        if len(vectors) != len(frame):
            raise SystemExit(
                f"Cached embeddings hold {len(vectors)} rows but the cleaned corpus has "
                f"{len(frame)} — rerun without --reuse-embeddings."
            )
        print(f"Reusing cached embeddings ({len(vectors):,} vectors)")
    else:
        print("Embedding narratives (first run downloads MiniLM, ~90 MB) ...")
        vectors = embed(frame["narrative"].tolist(), show_progress=True)
        np.save(EMBEDDINGS_PATH, vectors)

    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])  # vectors are L2-normalised → cosine
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  indexed {index.ntotal:,} vectors of dim {vectors.shape[1]}")

    frame.to_csv(CLEAN_COMPLAINTS_CSV, index=False)
    META_PATH.write_text(
        json.dumps(
            {
                "built": date.today().isoformat(),
                "embedding_model": EMBEDDING_MODEL,
                "n_complaints": int(len(frame)),
                "slor_mean": norm.mean,
                "slor_sd": norm.sd,
                "severity_medium_at": bands.medium,
                "severity_high_at": bands.high,
                "severity_cuts": bands.cuts,
                "companies": sorted(frame["company"].unique().tolist()),
                "date_min": frame["date_received"].min().date().isoformat(),
                "date_max": frame["date_received"].max().date().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nIndex written to {INDEX_DIR}")
    return frame


# --------------------------------------------------------------------------- #
# Read side
# --------------------------------------------------------------------------- #

@dataclass
class ComplaintHit:
    complaint_id: str
    score: float
    company: str
    product: str
    sub_product: str
    issue: str
    date_received: str
    narrative: str
    risk_score: float
    risk_percentile: int
    severity: str

    def excerpt(self, chars: int = 320) -> str:
        text = self.narrative
        return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + "…"


class ComplaintIndex:
    """Loaded-once, queryable view over the built artefacts."""

    def __init__(self, frame: pd.DataFrame, vectors: np.ndarray, faiss_index,
                 meta: dict, lm: NgramLM | None, norm: SlorNorm, bands: SeverityBands):
        self.frame = frame
        self.vectors = vectors
        self.faiss_index = faiss_index
        self.meta = meta
        self.bands = bands
        self.scorer = LinguisticRiskScorer(lm=lm, slor_norm=norm, bands=bands)

    # -- loading -------------------------------------------------------------
    @classmethod
    def load(cls) -> "ComplaintIndex":
        missing = [p for p in (CLEAN_COMPLAINTS_CSV, FAISS_INDEX_PATH) if not p.exists()]
        if missing:
            raise SystemExit(
                "Index artefacts missing: "
                + ", ".join(p.name for p in missing)
                + "\nRun `python -m data.fetch_cfpb` then `python -m data.ingest`."
            )
        import faiss

        frame = pd.read_csv(CLEAN_COMPLAINTS_CSV)
        frame["date_received"] = pd.to_datetime(frame["date_received"], errors="coerce")
        frame["complaint_id"] = frame["complaint_id"].astype(str)
        for column in ("narrative", "company", "product", "sub_product", "issue", "sub_issue"):
            frame[column] = frame[column].fillna("").astype(str)

        meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
        lm = NgramLM.load(LM_PATH) if LM_PATH.exists() else None
        norm = SlorNorm(meta.get("slor_mean", 0.0), meta.get("slor_sd", 1.0))
        bands = SeverityBands(
            medium=meta.get("severity_medium_at", SeverityBands.medium),
            high=meta.get("severity_high_at", SeverityBands.high),
            cuts=meta.get("severity_cuts", []),
        )
        # The filtered-search path needs the raw vectors as a matrix. They are
        # already inside the flat index, so reconstruct rather than shipping a
        # second 13 MB copy of the same floats to the Space; the .npy is kept
        # locally only to make `--reuse-embeddings` rebuilds fast.
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        if EMBEDDINGS_PATH.exists():
            vectors = np.load(EMBEDDINGS_PATH)
        else:
            vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)

        return cls(
            frame=frame,
            vectors=vectors,
            faiss_index=faiss_index,
            meta=meta,
            lm=lm,
            norm=norm,
            bands=bands,
        )

    # -- filtering -----------------------------------------------------------
    def _mask(
        self,
        company: str = "",
        product: str = "",
        issue: str = "",
        date_min: str = "",
        date_max: str = "",
        severity: str = "",
    ) -> np.ndarray:
        """Case-insensitive substring filters, so agents can pass loose values.

        The LLM will say "Wells Fargo" and "overdraft"; the data says
        "WELLS FARGO & COMPANY" and "Checking or savings account". Substring
        matching over company + product + sub_product bridges that without
        forcing the model to know CFPB's exact taxonomy.
        """
        frame = self.frame
        mask = np.ones(len(frame), dtype=bool)
        if company:
            mask &= frame["company"].str.contains(company, case=False, regex=False).to_numpy()
        if product:
            haystack = frame["product"] + " " + frame["sub_product"]
            mask &= haystack.str.contains(product, case=False, regex=False).to_numpy()
        if issue:
            haystack = frame["issue"] + " " + frame["sub_issue"]
            mask &= haystack.str.contains(issue, case=False, regex=False).to_numpy()
        if date_min:
            mask &= (frame["date_received"] >= pd.Timestamp(date_min)).to_numpy()
        if date_max:
            mask &= (frame["date_received"] <= pd.Timestamp(date_max)).to_numpy()
        if severity:
            mask &= (frame["severity"] == severity.lower()).to_numpy()
        return mask

    # -- search --------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 8,
        company: str = "",
        product: str = "",
        issue: str = "",
        date_min: str = "",
        date_max: str = "",
        severity: str = "",
    ) -> list[ComplaintHit]:
        """Cosine semantic search with metadata filters.

        Two paths, both exact. When filters keep most of the corpus, FAISS does
        the work and results are post-filtered. When filters are selective,
        over-fetching from FAISS would mostly return rows that fail the filter,
        so the filtered subset is scanned directly instead.
        """
        from nlp.embeddings import embed_one

        if self.frame.empty:
            return []

        mask = self._mask(company, product, issue, date_min, date_max, severity)
        candidates = int(mask.sum())
        if candidates == 0:
            return []

        query_vector = embed_one(query).astype("float32")
        selectivity = candidates / len(self.frame)

        if selectivity >= EXACT_SCAN_THRESHOLD:
            fetch = min(len(self.frame), max(k * 20, 200))
            scores, positions = self.faiss_index.search(query_vector[None, :], fetch)
            pairs = [
                (float(s), int(p))
                for s, p in zip(scores[0], positions[0])
                if p >= 0 and mask[p]
            ][:k]
            if len(pairs) < k:  # over-fetch came up short; fall back to exact
                pairs = self._exact(query_vector, mask, k)
        else:
            pairs = self._exact(query_vector, mask, k)

        return [self._hit(position, score) for score, position in pairs]

    def _exact(self, query_vector: np.ndarray, mask: np.ndarray,
               k: int) -> list[tuple[float, int]]:
        positions = np.flatnonzero(mask)
        similarities = self.vectors[positions] @ query_vector
        top = np.argsort(-similarities)[:k]
        return [(float(similarities[i]), int(positions[i])) for i in top]

    def _hit(self, position: int, score: float) -> ComplaintHit:
        row = self.frame.iloc[position]
        return ComplaintHit(
            complaint_id=str(row["complaint_id"]),
            score=round(score, 4),
            company=row["company"],
            product=row["product"],
            sub_product=row["sub_product"],
            issue=row["issue"],
            date_received=row["date_received"].date().isoformat()
            if pd.notna(row["date_received"]) else "",
            narrative=row["narrative"],
            risk_score=float(row["risk_score"]),
            risk_percentile=int(row["risk_percentile"]),
            severity=str(row["severity"]),
        )

    # -- convenience ---------------------------------------------------------
    def subset(self, **filters) -> pd.DataFrame:
        return self.frame[self._mask(**filters)]

    def by_id(self, complaint_id: str) -> ComplaintHit | None:
        rows = self.frame.index[self.frame["complaint_id"] == str(complaint_id)]
        return self._hit(int(rows[0]), 1.0) if len(rows) else None

    def describe(self) -> str:
        meta = self.meta
        return (
            f"{meta.get('n_complaints', len(self.frame)):,} complaints · "
            f"{meta.get('date_min', '?')} → {meta.get('date_max', '?')} · "
            f"{len(meta.get('companies', self.frame['company'].unique()))} companies"
        )


_INDEX: ComplaintIndex | None = None


def get_index() -> ComplaintIndex:
    """Process-wide singleton — the agents all query the same loaded index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = ComplaintIndex.load()
    return _INDEX


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RAW_COMPLAINTS_CSV))
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild even if index artefacts already exist")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="skip re-embedding when only the scoring changed")
    args = parser.parse_args(argv)

    if FAISS_INDEX_PATH.exists() and not args.rebuild:
        print(f"{FAISS_INDEX_PATH.name} already exists — pass --rebuild to overwrite.")
        return 0
    build(Path(args.raw), reuse_embeddings=args.reuse_embeddings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
