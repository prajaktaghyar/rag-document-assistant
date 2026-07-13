"""Load and cache the local embedding model for Render/production."""

from __future__ import annotations

import os

DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def preload_embedding_model(model_name: str | None = None) -> None:
    """Download/load the model (call during Render build to avoid runtime timeouts)."""
    load_embedding_model(model_name or DEFAULT_EMBEDDING_MODEL)


def load_embedding_model(model_name: str | None = None):
    """Return a sentence-transformers model, using HF cache when available."""
    from sentence_transformers import SentenceTransformer

    name = model_name or DEFAULT_EMBEDDING_MODEL
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    kwargs = {"cache_folder": cache_dir} if cache_dir else {}
    return SentenceTransformer(name, **kwargs)
