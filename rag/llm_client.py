"""
llm_client.py
-------------
Thin wrapper around the Grok (xAI) chat completions API. Grok's API is
OpenAI-compatible, so we reuse the `openai` Python SDK pointed at xAI's
base URL. The API key is read from the environment (GROK_API_KEY), never
hard-coded or passed in from the UI.
"""

from __future__ import annotations

import os

from openai import OpenAI

from .vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a precise document-analysis assistant.
Answer the user's question using ONLY the provided context excerpts from
their uploaded document(s). Follow these rules:

1. Ground every claim in the context. Do not use outside knowledge unless
   the context is silent AND the user explicitly asks for general knowledge.
2. If the context does not contain enough information to answer, say so
   clearly instead of guessing.
3. Cite which source file(s) you drew from inline, like (source: filename.pdf).
4. Be concise and directly answer the question first, then add supporting
   detail if useful.
5. If the question is ambiguous given the context, note the ambiguity and
   answer the most likely interpretation.
"""


class GrokClientError(RuntimeError):
    pass


def _get_client() -> OpenAI:
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        raise GrokClientError(
            "GROK_API_KEY is not set. Add it to your .env file "
            "(see .env.example) or export it in your shell environment."
        )
    base_url = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} | source: {rc.chunk.source} | relevance: {rc.score:.2f}]\n"
            f"{rc.chunk.text}"
        )
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    chat_history: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Calls Grok with the retrieved context + question and returns a refined
    answer grounded in the document excerpts.
    """
    client = _get_client()
    model = model or os.environ.get("GROK_MODEL", "grok-2-latest")

    context_block = _build_context_block(retrieved_chunks)
    if not context_block.strip():
        context_block = "(No relevant excerpts were retrieved from the document.)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include prior turns for conversational follow-ups (kept short to save tokens)
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"CONTEXT EXCERPTS:\n{context_block}\n\n"
        f"QUESTION: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 - surface any API error to the UI
        raise GrokClientError(f"Grok API request failed: {exc}") from exc

    return response.choices[0].message.content
