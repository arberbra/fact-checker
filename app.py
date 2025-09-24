import os
import textwrap
import re
import signal
import time
from io import BytesIO
from typing import List, Tuple, Dict, Any

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



# Bill Nye Model Integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    HAS_BILL_NYE_MODEL = True
except Exception:
    HAS_BILL_NYE_MODEL = False

def summarize_with_bill_nye(text: str, query: str, rag_context: str = "", model_name: str = None) -> str:
    """Summarize retrieved text with Bill Nye personality model."""
    if not HAS_BILL_NYE_MODEL:
        return "Bill Nye model integration not available. Please install transformers and torch."
    
    if not model_name:
        model_name = os.getenv("BILL_NYE_MODEL_NAME", "arberbr/bill-nye-science-guy")
    
    try:
        # Check if model is already loaded in session state
        model_key = f"bill_nye_model_{model_name.replace('/', '_')}"
        tokenizer_key = f"bill_nye_tokenizer_{model_name.replace('/', '_')}"
        
        if model_key not in st.session_state or tokenizer_key not in st.session_state:
            # Load model and tokenizer only if not already loaded
            with st.spinner("Loading Bill Nye model (this may take a moment on first use)..."):
                try:
                    # The Bill Nye model is a LoRA adapter, so we need to load the base model first
                    base_model_name = "microsoft/DialoGPT-medium"
                    adapter_name = model_name  # This should be "arberbr/bill-nye-science-guy"
                    
                    # Load tokenizer from base model
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    
                    # Add padding token if it doesn't exist
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    
                    # Check if CUDA is available for optimal loading
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    if device == "cuda":
                        # GPU loading with optimizations
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                    else:
                        # CPU loading with optimizations
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=torch.float32,  # Use float32 for CPU
                            low_cpu_mem_usage=True,
                            use_cache=True  # Enable KV cache for faster inference
                        )
                        # Move to CPU explicitly
                        base_model = base_model.to("cpu")
                    
                    # Load the LoRA adapter on top of the base model
                    model = PeftModel.from_pretrained(base_model, adapter_name)
                    
                    # Cache in session state
                    st.session_state[tokenizer_key] = tokenizer
                    st.session_state[model_key] = model
                    st.session_state[f"{model_key}_device"] = device
                    
                except Exception as e:
                    st.error(f"Failed to load Bill Nye model: {str(e)}")
                    return f"Model loading failed: {str(e)}"
        else:
            # Use cached model and tokenizer
            tokenizer = st.session_state[tokenizer_key]
            model = st.session_state[model_key]
        
        # Prepare context
        context = text[:8000]  # Limit context size
        rag = rag_context[:4000] if rag_context else ""
        
        # Create Bill Nye-style prompt
        prompt = f"""Human: You are Bill Nye, the Science Guy. Answer the following question in your characteristic enthusiastic and educational style. Use scientific evidence and explain things in an engaging way that makes science accessible to everyone.

Query: {query}

Web search results: {context}

Additional context: {rag or '[none]'}

Please provide a fact-checked answer in your signature enthusiastic style!
Bill Nye:"""
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Get device info
        device = st.session_state.get(f"{model_key}_device", "cpu")
        
        # Move inputs to the same device as model
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                early_stopping=True  # Stop when EOS token is generated
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just Bill Nye's response (remove the prompt part)
        if "Bill Nye:" in response:
            response = response.split("Bill Nye:")[-1].strip()
        
        return response
        
    except Exception as e:
        return f"Error generating Bill Nye response: {str(e)}"

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


def search_web(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Perform a Serper.dev Google search and join snippets.

    Returns a tuple of (combined_snippets_text, sources_list).
    Each source has keys: title, link, snippet.
    If there are no meaningful matches, combined text is a 'No results' message and sources can be empty.
    """
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY is not set. Add it to your environment or .env file."
        )

    search = GoogleSerperAPIWrapper()
    results = search.results(query)

    organic = results.get("organic", []) or []
    snippets = [r.get("snippet", "") for r in organic]
    sources: List[Dict[str, Any]] = []
    for r in organic:
        sources.append({
            "title": r.get("title", r.get("link", "")) or "(untitled)",
            "link": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        })

    # Early no-result condition
    if not any(s.strip() for s in snippets):
        return (f"No results found for: {query}", [])

    # Heuristic: require at least one meaningful query term to appear
    terms = _extract_query_terms(query)
    combined = "\n\n".join(s.strip() for s in snippets if s.strip())
    text_lc = combined.lower()
    if terms and not any(t in text_lc for t in terms):
        return (f"No results found for: {query}", sources)

    return (combined, sources)


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
        "BILL_NYE_MODEL_NAME",
        value=os.getenv("BILL_NYE_MODEL_NAME", "arberbr/bill-nye-science-guy"),
        help="Hugging Face model name for Bill Nye personality",
        key="bill_nye_model",
    )
    
    # Model management buttons
    col_clear, col_preload = st.columns(2)
    
    with col_clear:
        if st.button("Clear Cache", help="Clear cached model to force reload"):
            # Clear all Bill Nye model related session state
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("bill_nye_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Model cache cleared!")
    
    with col_preload:
        if st.button("Preload Model", help="Load model now to avoid delay later"):
            if HAS_BILL_NYE_MODEL:
                model_name = st.session_state.get("bill_nye_model", "arberbr/bill-nye-science-guy")
                with st.spinner("Preloading Bill Nye model..."):
                    result = summarize_with_bill_nye("test", "test")
                    if not result.startswith("Model loading failed:"):
                        st.success("Model preloaded successfully!")
                    else:
                        st.error("Failed to preload model")
            else:
                st.error("Transformers not available")
    st.text_input(
        "ANTHROPIC_API_KEY",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Optional Anthropic API key",
        key="anthropic_key",
    )
    # Model status indicator
    st.markdown("**Model Status:**")
    model_name = st.session_state.get("bill_nye_model", "arberbr/bill-nye-science-guy")
    model_key = f"bill_nye_model_{model_name.replace('/', '_')}"
    
    if model_key in st.session_state:
        device = st.session_state.get(f"{model_key}_device", "unknown")
        st.success(f"✅ Bill Nye model loaded ({device})")
    else:
        st.info("⏳ Bill Nye model not loaded")
    
    st.markdown(
        """**Setup Notes:**
- Requires Serper.dev key in `SERPER_API_KEY`.
- Optional Anthropic key in `ANTHROPIC_API_KEY` for summaries.
- Bill Nye model: LoRA adapter on DialoGPT-medium (no API key required).
- Model components are cached in memory after first load for faster responses."""
    )

# Keep env vars in sync with sidebar updates (no secrets are persisted by the app)
os.environ["SERPER_API_KEY"] = st.session_state.get("serper_key", "")
os.environ["ANTHROPIC_API_KEY"] = st.session_state.get("anthropic_key", "")
os.environ["BILL_NYE_MODEL_NAME"] = st.session_state.get("bill_nye_model", "")

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
            web_text, web_sources = search_web(query.strip())
        st.subheader("Search Snippets")
        st.write(web_text)

        if web_sources:
            st.subheader("Sources")
            for src in web_sources[:10]:
                title = src.get("title") or src.get("link") or "(untitled)"
                link = src.get("link") or ""
                snippet = src.get("snippet", "")
                if link:
                    st.markdown(f"- [{title}]({link})")
                else:
                    st.markdown(f"- {title}")
                if snippet:
                    st.caption(snippet)

        rag_context = ""
        if use_rag and retriever is not None:
            with st.spinner("Retrieving from uploaded documents..."):
                top = retrieve_top_k(retriever, retriever_chunks, query.strip(), k=5)
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

        if HAS_BILL_NYE_MODEL and not web_text.lower().startswith("no results found"):
            with st.spinner("Getting Bill Nye's scientific perspective..."):
                summary = summarize_with_bill_nye(web_text, query.strip(), rag_context)
            
            # Check if Bill Nye model failed and fallback to Anthropic if available
            if summary.startswith("Model loading failed:") or summary.startswith("Error generating"):
                st.warning("Bill Nye model unavailable. Falling back to Anthropic if available.")
                if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
                    with st.spinner("Summarizing with Anthropic..."):
                        summary = summarize_with_anthropic(web_text, query.strip(), rag_context)
                    st.subheader("Answer")
                else:
                    st.subheader("Search Results Summary")
                    # Provide a basic summary without AI
                    summary = f"**Query:** {query.strip()}\n\n**Key Information Found:**\n{web_text[:1000]}..."
            else:
                st.subheader("Bill Nye's Answer")
            
            st.write(summary)
        elif HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY") and not web_text.lower().startswith("no results found"):
            with st.spinner("Summarizing with Anthropic..."):
                summary = summarize_with_anthropic(web_text, query.strip(), rag_context)
            st.subheader("Answer")
            st.write(summary)
        else:
            # No AI available, just show search results
            st.info("AI summarization unavailable. Showing search results above.")
    except Exception as e:
        st.error(str(e))
elif do_search:
    st.warning("Please enter a query.")
