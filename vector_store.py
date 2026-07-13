"""
vector_store.py
----------------
Chunks document text, embeds chunks with a local sentence-transformers
model (no API key / network dependency for embeddings), and indexes them
in a FAISS store for similarity search at query time.

Embeddings are done locally (not via Grok) because embedding quality and
availability vary across LLM providers, and keeping retrieval local keeps
the pipeline fast, free, and independent of the chat model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _split_on_separator(text: str, separators: list[str]):
    """Split text on the first separator (from the priority list) present in it.
    Returns (pieces, separator_used) -- separator_used is '' if none matched."""
    for sep in separators:
        if sep and sep in text:
            return text.split(sep), sep
    return list(text), ""


def _merge_pieces(pieces: list[str], sep: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks up to chunk_size, carrying a
    character-based overlap between consecutive chunks."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + sep + piece if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            overlap_tail = current[-chunk_overlap:] if chunk_overlap and current else ""
            current = (overlap_tail + sep + piece) if overlap_tail else piece
    if current.strip():
        chunks.append(current)
    return chunks


def _hard_cut(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= chunk_overlap:
        chunk_overlap = 0
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Dependency-light splitter: tries paragraph/line/sentence boundaries near
    the target chunk_size, falling back to a hard character cut for any
    oversized piece. Keeps an overlap between consecutive chunks so context
    isn't lost at boundaries. Guaranteed to terminate -- every fallback
    strictly shrinks the problem, so there is no unbounded recursion.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        chunk_size = 1000
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]

    pieces, sep = _split_on_separator(text, separators)
    merged = _merge_pieces(pieces, sep, chunk_size, chunk_overlap)

    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            if chunk.strip():
                final.append(chunk)
            continue
        # Oversized piece: try a finer-grained separator, else hard-cut.
        remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else []
        sub_pieces, sub_sep = _split_on_separator(chunk, remaining_seps) if remaining_seps else ([], "")
        if sub_sep:
            final.extend(_merge_pieces(sub_pieces, sub_sep, chunk_size, chunk_overlap))
        else:
            final.extend(_hard_cut(chunk, chunk_size, chunk_overlap))

    return [c.strip() for c in final if c.strip()]


class VectorStore:
    """In-memory FAISS-backed store. Rebuilt fresh each Streamlit session."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self._embedding_model_name = embedding_model_name
        self._model = None  # lazy-loaded, sentence-transformers import is heavy
        self.index = None
        self.chunks: list[Chunk] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
        )
        return embeddings.astype("float32")

    def build(self, documents: dict[str, str], chunk_size: int = 1000, chunk_overlap: int = 150) -> int:
        """
        documents: mapping of filename -> full extracted text.
        Returns the number of chunks indexed.
        """
        import faiss

        self.chunks = []
        for filename, text in documents.items():
            pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, piece in enumerate(pieces):
                self.chunks.append(Chunk(text=piece, source=filename, chunk_index=i))

        if not self.chunks:
            self.index = None
            return 0

        embeddings = self._embed([c.text for c in self.chunks])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self.index.add(embeddings)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None or not self.chunks:
            return []

        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
