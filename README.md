## Fact Checker App

Web search via Serper.dev with optional Anthropic summarization, wrapped in Streamlit.

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

### Notes
- The sidebar lets you paste keys at runtime; the app does not persist secrets.
- Summarization uses `claude-3-5-sonnet-20240620` via `langchain-anthropic`.
