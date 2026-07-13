#!/usr/bin/env python3
"""Pre-download the embedding model during Render build."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import preload_embedding_model

if __name__ == "__main__":
    preload_embedding_model()
    print("Embedding model downloaded successfully.")
