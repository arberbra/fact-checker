import os
import textwrap
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper

# Optional Anthropic summarization
try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False


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


def summarize_with_anthropic(text: str, query: str) -> str:
    """Summarize retrieved text with Anthropic, if available and key is set."""
    if not HAS_ANTHROPIC:
        return "Anthropic integration not installed."
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set. Skipping summary."

    # Keep prompt size reasonable
    context = text[:12000]

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.2,
    )

    prompt = textwrap.dedent(
        f"""
        You are a precise research assistant. Given the user's query and extracted web snippets,
        produce a concise, well-structured fact-check style summary. Include citations as bullet
        points if URLs or domains are present in the snippets. If information is uncertain, say so.

        Query: {query}

        Snippets:
        {context}

        Task: Provide a brief summary (<= 200 words) answering the query.
        """
    ).strip()

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# --- Streamlit UI ---
st.set_page_config(page_title="Fact Checker", page_icon="🔎", layout="wide")
st.title("🔎 Fact Checker")
st.caption("Web search with optional Anthropic summarization")

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

query = st.text_input(
    "Enter a query to fact-check",
    placeholder="e.g., What did the latest CPI report say?",
    key="query",
)
col1, _ = st.columns([1, 2])

with col1:
    do_search = st.button("Search", type="primary")
    do_summary = st.checkbox("Summarize with Anthropic", value=False)

if do_search and query.strip():
    try:
        with st.spinner("Searching the web..."):
            results_text = search_web(query.strip())
        st.subheader("Search Snippets")
        st.write(results_text)

        if do_summary and not results_text.lower().startswith("no results found"):
            with st.spinner("Summarizing..."):
                summary = summarize_with_anthropic(results_text, query.strip())
            st.subheader("Summary")
            st.write(summary)
    except Exception as e:
        st.error(str(e))
elif do_search:
    st.warning("Please enter a query.")
