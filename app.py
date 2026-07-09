from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag.llm_client import GrokClientError, answer_question
from rag.loaders import load_document
from rag.vector_store import VectorStore

load_dotenv()

# Support Streamlit Cloud's secrets manager in addition to local .env files.
# Locally: GROK_API_KEY comes from .env via load_dotenv() above.
# On Streamlit Cloud: set it in the app's Settings -> Secrets as
#   GROK_API_KEY = "your_key_here"
# and it will be picked up here and exposed via os.environ as before.
if "GROK_API_KEY" in st.secrets:
    os.environ["GROK_API_KEY"] = st.secrets["GROK_API_KEY"]

st.set_page_config(
    page_title="Document Analyzer (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", 5))


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --accent-1: #6C5CE7;
            --accent-2: #00CEC9;
            --accent-3: #FD79A8;
            --bg-soft: #0f1220;
        }

        /* Hide default streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* App background */
        .stApp {
            background: radial-gradient(circle at 15% 0%, #1a1c2e 0%, #0e0f1a 55%, #0a0b12 100%);
        }

        /* Hero banner */
        .hero-banner {
            padding: 2.2rem 2.4rem;
            border-radius: 20px;
            margin-bottom: 1.6rem;
            background: linear-gradient(120deg, rgba(108,92,231,0.25), rgba(0,206,201,0.18) 55%, rgba(253,121,168,0.18));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(0,206,201,0.35), transparent 70%);
            filter: blur(10px);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #f5f6fa;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: #b7bcd6;
            margin-top: 0.5rem;
            max-width: 680px;
            line-height: 1.5;
        }
        .hero-badges {
            margin-top: 1.1rem;
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }
        .badge-pill {
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            color: #e4e6f5;
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #9a9fc0;
            margin-top: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #14162a 0%, #0d0e18 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stButton button {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-3));
            border: none;
            font-weight: 700;
            border-radius: 10px;
            padding: 0.6rem 0;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(108,92,231,0.4);
        }

        /* File card in sidebar */
        .file-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        .file-chip-name {
            font-weight: 600;
            color: #eef0fa;
        }
        .file-chip-meta {
            color: #8d92b3;
            font-size: 0.75rem;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.3rem 0.4rem;
        }

        /* Retrieved context expander */
        .context-chunk {
            background: rgba(255,255,255,0.035);
            border-left: 3px solid var(--accent-2);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.6rem;
            font-size: 0.85rem;
            color: #c9cce0;
        }
        .context-chunk-header {
            font-weight: 700;
            color: #f0f1fa;
            margin-bottom: 0.25rem;
            display: flex;
            justify-content: space-between;
        }
        .score-tag {
            background: rgba(0,206,201,0.15);
            color: #00cec9;
            border-radius: 999px;
            padding: 0.05rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 700;
        }

        /* Status pill */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .status-ok { background: rgba(46, 213, 115, 0.14); color: #2ed573; }
        .status-bad { background: rgba(255, 71, 87, 0.14); color: #ff4757; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # list of dicts: filename, chars, warning
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []


# --------------------------------------------------------------------------
# Sidebar: upload + index
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Document Vault")
    st.caption("Any file type is accepted. Text is best-effort extracted for indexing.")

    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    top_k = st.slider("🔍 Chunks to retrieve per question", min_value=1, max_value=15, value=DEFAULT_TOP_K)

    process_clicked = st.button("⚡ Build / Rebuild Index", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            documents: dict[str, str] = {}
            processed_meta = []
            with st.spinner("Extracting text from documents..."):
                for f in uploaded_files:
                    data = f.getvalue()
                    loaded = load_document(f.name, data)
                    if loaded.text:
                        documents[loaded.filename] = loaded.text
                    processed_meta.append(
                        {
                            "filename": loaded.filename,
                            "chars": loaded.num_chars,
                            "warning": loaded.warning,
                        }
                    )

            if not documents:
                st.error("No extractable text was found in the uploaded file(s).")
            else:
                with st.spinner("Embedding and indexing chunks..."):
                    n_chunks = st.session_state.vector_store.build(
                        documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                    )
                st.session_state.processed_files = processed_meta
                st.session_state.chat_history = []
                st.success(f"Indexed {len(documents)} file(s) into {n_chunks} chunks.")
                st.balloons()

    if st.session_state.processed_files:
        st.divider()
        st.markdown("#### 📚 Indexed files")
        for meta in st.session_state.processed_files:
            warning_html = (
                f'<div class="file-chip-meta">⚠️ {meta["warning"]}</div>' if meta["warning"] else ""
            )
            st.markdown(
                f"""
                <div class="file-chip">
                    <div class="file-chip-name">📄 {meta['filename']}</div>
                    <div class="file-chip-meta">{meta['chars']:,} characters extracted</div>
                    {warning_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    api_key_present = bool(os.environ.get("GROK_API_KEY"))
    if api_key_present:
        st.markdown('<span class="status-pill status-ok">🟢 GROK_API_KEY found</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-bad">🔴 GROK_API_KEY missing</span>', unsafe_allow_html=True)
        st.caption("Set it in a `.env` file (see `.env.example`) or your shell environment.")


# --------------------------------------------------------------------------
# Main: hero + stats
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">📄 Document Analyzer</p>
        <p class="hero-subtitle">
            Upload documents, build a semantic index, and ask questions in natural language.
            Answers are grounded in your files via retrieval-augmented generation, refined by Grok.
        </p>
        <div class="hero-badges">
            <span class="badge-pill">🔎 Semantic Search</span>
            <span class="badge-pill">🧩 Chunked Embeddings</span>
            <span class="badge-pill">🤖 Grok-Powered Answers</span>
            <span class="badge-pill">💬 Conversational Memory</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

n_files = len(st.session_state.processed_files)
n_chunks = getattr(st.session_state.vector_store, "num_chunks", None)
if n_chunks is None:
    try:
        n_chunks = len(st.session_state.vector_store.chunks)  # best-effort fallback
    except Exception:
        n_chunks = 0
n_questions = sum(1 for t in st.session_state.chat_history if t["role"] == "user")

col1, col2, col3, col4 = st.columns(4)
for col, value, label in zip(
    [col1, col2, col3, col4],
    [n_files, n_chunks, n_questions, top_k],
    ["Files Indexed", "Chunks Stored", "Questions Asked", "Top-K Retrieval"],
):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

if not st.session_state.vector_store.is_ready:
    st.info("👈 Upload one or more files and click **Build / Rebuild Index** to get started.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your document(s)...")

if question:
    if not st.session_state.vector_store.is_ready:
        st.chat_message("assistant").markdown(
            "I don't have any indexed documents yet — upload files and build the index first."
        )
    else:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        retrieved = st.session_state.vector_store.search(question, top_k=top_k)
        st.session_state.last_retrieved = retrieved

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and asking Grok..."):
                try:
                    answer = answer_question(
                        question=question,
                        retrieved_chunks=retrieved,
                        chat_history=st.session_state.chat_history[:-1],
                    )
                except GrokClientError as e:
                    answer = f"⚠️ {e}"
            st.markdown(answer)

            with st.expander(f"📚 Retrieved context ({len(retrieved)} chunks)"):
                for i, rc in enumerate(retrieved, start=1):
                    preview = rc.chunk.text[:500] + ("..." if len(rc.chunk.text) > 500 else "")
                    st.markdown(
                        f"""
                        <div class="context-chunk">
                            <div class="context-chunk-header">
                                <span>{i}. {rc.chunk.source}</span>
                                <span class="score-tag">score {rc.score:.3f}</span>
                            </div>
                            {preview}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
