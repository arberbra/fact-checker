import os
import textwrap
import re
from io import BytesIO
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper

# Optional Anthropic summarization
try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

# Lightweight RAG deps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader


# --- Startup: load .env if present ---
load_dotenv(override=False)


def _extract_query_terms(query: str) -> list:
    """Extract simple meaningful terms from the query for matching."""
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    # ignore very short tokens and common fillers
    stop = {
        "the","a","an","and","or","of","to","in","for","on","is","are",
        "what","who","when","where","why","how","does","did","with","about"
    }
    return [t for t in tokens if len(t) >= 3 and t not in stop]


def _chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Simple fixed-size chunker with overlap."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(s for s in out if s)


def build_retriever(docs: List[Tuple[str, str]]):
    """Build a TF-IDF retriever from a list of (source_name, text) docs."""
    sources = []
    chunks = []
    for src, text in docs:
        for chunk in _chunk_text(text):
            sources.append(src)
            chunks.append(chunk)

    if not chunks:
        return None, [], []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return (vectorizer, matrix), chunks, sources


def retrieve_top_k(retriever, chunks: List[str], query: str, k: int = 5) -> List[Tuple[str, float]]:
    vectorizer, matrix = retriever
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in idxs if sims[i] > 0]


def search_web(query: str) -> str:
    """Perform a Serper.dev Google search and join snippets into a single string.

    If there are no snippets or none contain meaningful query terms, return a
    user-friendly 'No results found' message.
    """
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY is not set. Add it to your environment or .env file."
        )

    search = GoogleSerperAPIWrapper()
    results = search.results(query)

    organic = results.get("organic", []) or []
    snippets = [r.get("snippet", "") for r in organic]

    # Early no-result condition
    if not any(s.strip() for s in snippets):
        return f"No results found for: {query}"

    # Heuristic: require at least one meaningful query term to appear
    terms = _extract_query_terms(query)
    combined = "\n\n".join(s.strip() for s in snippets if s.strip())
    text_lc = combined.lower()
    if terms and not any(t in text_lc for t in terms):
        return f"No results found for: {query}"

    return combined


def summarize_with_anthropic(text: str, query: str, rag_context: str = "") -> str:
    """Summarize retrieved text with Anthropic, optionally including RAG context."""
    if not HAS_ANTHROPIC:
        return "Anthropic integration not installed."
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set. Skipping summary."

    context = text[:10000]
    rag = rag_context[:6000] if rag_context else ""

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.2,
    )

    prompt = textwrap.dedent(
        f"""
        You are a precise research assistant. Use the user's query, web snippets, and any retrieved
        document context to produce a concise, well-structured fact-check. Prefer quoted, attributed
        facts from the RAG context when relevant. If information is uncertain, say so.

        Query:
        {query}

        Web snippets:
        {context}

        Retrieved document context:
        {rag or '[none]'}

        Task: Provide a brief answer (<= 250 words) with bullet citations by source name or domain.
        """
    ).strip()

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# --- Streamlit UI ---
st.set_page_config(page_title="Fact Checker", page_icon="🔎", layout="wide")
st.title("🔎 Fact Checker")
st.caption("Web search with optional RAG and Anthropic summarization")

with st.sidebar:
    st.header("Settings")
    st.text_input(
        "SERPER_API_KEY",
        value=os.getenv("SERPER_API_KEY", ""),
        type="password",
        help="Serper.dev API key",
        key="serper_key",
    )
    st.text_input(
        "ANTHROPIC_API_KEY",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Optional Anthropic API key",
        key="anthropic_key",
    )
    st.markdown(
        "- Requires Serper.dev key in `SERPER_API_KEY`.\n- Optional Anthropic key in `ANTHROPIC_API_KEY` for summaries."
    )

# Keep env vars in sync with sidebar updates (no secrets are persisted by the app)
os.environ["SERPER_API_KEY"] = st.session_state.get("serper_key", "")
os.environ["ANTHROPIC_API_KEY"] = st.session_state.get("anthropic_key", "")

# RAG document upload
st.subheader("Knowledge Base (optional)")
uploaded_files = st.file_uploader(
    "Upload PDFs or text files to use for retrieval",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

kb_docs: List[Tuple[str, str]] = []
for f in uploaded_files or []:
    content = ""
    if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
        try:
            content = _read_pdf(f.getvalue())
        except Exception:
            content = ""
    else:
        try:
            content = f.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            content = ""
    if content.strip():
        kb_docs.append((f.name, content))

retriever = None
retriever_chunks: List[str] = []
retriever_sources: List[str] = []
if kb_docs:
    retriever, retriever_chunks, retriever_sources = build_retriever(kb_docs)

query = st.text_input(
    "Enter a query to fact-check",
    placeholder="e.g., What did the latest CPI report say?",
    key="query",
)
col1, col2, _ = st.columns([1, 1, 2])

with col1:
    do_search = st.button("Search Web", type="primary")
with col2:
    use_rag = st.checkbox("Use RAG (retrieve from uploaded docs)", value=False)

if do_search and query.strip():
    try:
        with st.spinner("Searching the web..."):
            web_text = search_web(query.strip())
        st.subheader("Search Snippets")
        st.write(web_text)

        rag_context = ""
        if use_rag and retriever is not None:
            with st.spinner("Retrieving from uploaded documents..."):
                top = retrieve_top_k(retriever, retriever_chunks, query.strip(), k=5)
                # annotate with simple source names when possible
                rag_lines = []
                for chunk, score in top:
                    rag_lines.append(f"[score={score:.2f}] {chunk}")
                rag_context = "\n\n".join(rag_lines)
                if not rag_context:
                    st.info("No relevant RAG chunks found.")
                else:
                    st.subheader("Retrieved Context (top chunks)")
                    st.write(rag_context)
        elif use_rag and retriever is None:
            st.info("Upload documents first to enable RAG.")

        if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY") and not web_text.lower().startswith("no results found"):
            with st.spinner("Summarizing with Anthropic..."):
                summary = summarize_with_anthropic(web_text, query.strip(), rag_context)
            st.subheader("Answer")
            st.write(summary)
    except Exception as e:
        st.error(str(e))
elif do_search:
    st.warning("Please enter a query.")
