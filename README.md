# Document Analyzer — RAG + Grok + Streamlit

Upload documents of (almost) any file type, ask questions in a chat interface,
and get answers grounded in your documents' content. Retrieval is done
locally; final answers are generated and refined by **Grok** (xAI).

## How it works

```
Upload file(s) → extract text (rag/loaders.py)
             → chunk text (rag/vector_store.py)
             → embed chunks locally (sentence-transformers, all-MiniLM-L6-v2)
             → index in FAISS
Ask question → embed question → retrieve top-k similar chunks
             → send question + chunks to Grok (rag/llm_client.py)
             → Grok returns a grounded, refined answer
```

Embeddings/retrieval run locally (no API key needed for that part) so the
app stays fast and works even before you've set up Grok. Only the final
answer generation calls the Grok API.

## Supported file types

Explicitly parsed: `.txt` `.md` `.csv` `.tsv` `.json` `.log` `.pdf` `.docx`
`.pptx` `.xlsx` `.xls` `.html` `.htm`

Anything else falls back to a best-effort raw text decode, so the app never
hard-fails on an unrecognized extension — it just does its best with the
bytes it's given, and shows a warning in the UI so you know extraction may
be incomplete.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Grok API key**

   Copy `.env.example` to `.env` and fill in your key:

   ```bash
   cp .env.example .env
   ```

   ```
   GROK_API_KEY=your_grok_api_key_here
   ```

   Get a key from https://console.x.ai. The key is read from the
   environment only — it's never entered in the UI or logged.

   Optional overrides in `.env`:
   - `GROK_BASE_URL` (default `https://api.x.ai/v1`)
   - `GROK_MODEL` (default `grok-2-latest` — set to whichever Grok model
     your account has access to, e.g. a newer `grok-4` variant)
   - `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` (retrieval tuning)

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (usually http://localhost:8501).

## Usage

1. In the sidebar, upload one or more files.
2. Click **Build / Rebuild Index**.
3. Ask questions in the chat box at the bottom. Each answer shows the
   retrieved source excerpts in an expandable panel so you can verify
   grounding.
4. Upload new files and rebuild the index any time — this replaces the
   previous index for the session.

## Project structure

```
rag_app/
├── app.py                 # Streamlit UI
├── rag/
│   ├── loaders.py          # Multi-format text extraction
│   ├── vector_store.py     # Chunking + local embeddings + FAISS search
│   └── llm_client.py       # Grok API call for grounded answer generation
├── requirements.txt
├── .env.example
└── README.md
```

## Notes & tuning

- The first run downloads the local embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`, ~80MB) from Hugging Face —
  needs internet access once, then it's cached.
- Increase `TOP_K` (sidebar slider) if answers seem to be missing relevant
  context; decrease it to keep prompts smaller/cheaper.
- For very large documents, consider raising `CHUNK_SIZE` in `.env` to
  reduce the total chunk count.
- Swap the embedding model in `VectorStore(embedding_model_name=...)` for a
  larger/more accurate one if needed (trade-off: slower indexing).
# rag-document-assistant
A Streamlit RAG application that indexes documents into vector embeddings and answers natural-language questions using Grok, with source-grounded retrieval and chat memory.
