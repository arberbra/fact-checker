## Fact Checker App

Web search via Serper.dev with optional RAG, Bill Nye AI personality, and Anthropic summarization, wrapped in Streamlit.

### 1) Quick Setup (Recommended)

**Windows:**
```bash
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### 2) Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_model.py  # Pre-download Bill Nye model
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
- `BILL_NYE_MODEL_NAME` (defaults to "arberbr/bill-nye-science-guy")

### 3) Run the app

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually http://localhost:8501).

### 4) RAG (Retrieval-Augmented Generation)
- Upload PDFs or plain text files in the "Knowledge Base (optional)" section.
- Toggle "Use RAG" to enable retrieval from your uploaded docs.
- The app chunks documents and builds a lightweight TF‑IDF index for similarity search.
- Retrieved chunks are shown and also injected into the Bill Nye or Anthropic summary when enabled.

### 5) Links and Sources
- The "Sources" section lists top web results as clickable links, including a short snippet.
- Uploaded documents are not linked directly; retrieved text is labeled and shown inline.

### 6) Bill Nye AI Personality
- The app includes a fine-tuned Bill Nye personality model for scientific fact-checking
- Model is automatically downloaded during setup and cached for fast loading
- Use the "Clear Model Cache" button in the sidebar to force reload the model
- Model responses are cached in memory for faster subsequent queries

### Notes
- The sidebar lets you paste keys at runtime; the app does not persist secrets.
- Summarization uses `claude-3-5-sonnet-20240620` via `langchain-anthropic`.
- Bill Nye model is cached locally after first download for faster loading.
