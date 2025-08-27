## Fact Checker App

Web search via Serper.dev with optional RAG and Anthropic summarization, wrapped in Streamlit.

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (or set environment variables):

```bash
cp .env.example .env
# then edit .env to add your keys
```

Required:
- `SERPER_API_KEY` (get at https://serper.dev)

Optional:
- `ANTHROPIC_API_KEY` (for summaries)

### 2) Run the app

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually http://localhost:8501).

### 3) RAG (Retrieval-Augmented Generation)
- Upload PDFs or plain text files in the "Knowledge Base (optional)" section.
- Toggle "Use RAG" to enable retrieval from your uploaded docs.
- The app chunks documents and builds a lightweight TF‑IDF index for similarity search.
- Retrieved chunks are shown and also injected into the Anthropic summary when enabled.
A
### 4) Links and Sources
- The "Sources" section lists top web results as clickable links, including a short snippet.
- Uploaded documents are not linked directly; retrieved text is labeled and shown inline.

### Notes
- The sidebar lets you paste keys at runtime; the app does not persist secrets.
- Summarization uses `claude-3-5-sonnet-20240620` via `langchain-anthropic`.
