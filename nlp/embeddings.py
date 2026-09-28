"""Sentence-transformer wrapper.

One lazily-loaded, process-wide model. Loading MiniLM costs a few seconds and
~90 MB of RAM, which matters on a free CPU Space, so it is never loaded twice.
"""

from __future__ import annotations

import numpy as np

from config import EMBEDDING_MODEL

_model = None


def get_model(name: str = EMBEDDING_MODEL):
    """Load (once) and return the sentence-transformer."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import, keep local

        _model = SentenceTransformer(name, device="cpu")
    return _model


def embed(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
    name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed texts as L2-normalised float32 vectors.

    Normalising here means FAISS inner-product search is exactly cosine
    similarity, so scores are directly comparable and bounded to [-1, 1].
    """
    if not texts:
        return np.zeros((0, get_model(name).get_sentence_embedding_dimension()), dtype="float32")
    vectors = get_model(name).encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def embed_one(text: str, name: str = EMBEDDING_MODEL) -> np.ndarray:
    return embed([text], name=name)[0]


def dimension(name: str = EMBEDDING_MODEL) -> int:
    return int(get_model(name).get_sentence_embedding_dimension())
