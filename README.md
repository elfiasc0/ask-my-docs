# ask-my-docs

A local RAG (Retrieval-Augmented Generation) pipeline for asking questions
about your own documents. Point it at a folder of text, and it builds a
searchable index you can query in plain English.

Built as a learning project — it demonstrates the full RAG lifecycle:
loading, chunking, embedding, storing, retrieving, and generating.

## Stack

| Component | Purpose |
|---|---|
| [LlamaIndex](https://www.llamaindex.ai/) | Orchestration (loading, chunking, retrieval) |
| [Chroma](https://www.trychroma.com/) | Persistent vector store (embedded, no server) |
| [Voyage AI](https://www.voyageai.com/) | Embeddings (`voyage-3.5-lite`) |
| [Google Gemini](https://ai.google.dev/) | Answer synthesis (`gemini-3.8-flash`) |

## How it works

```
 ./data/*.txt
      │
      ▼
 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │  Load    │───▶│  Chunk   │───▶│  Embed   │───▶│  Store   │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       │
                                                       ▼
                                                 ./chroma_db/
                                                       │
                                                       ▼
 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │  Answer  │◀───│ Generate │◀───│ Retrieve │◀───│  Query   │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

1. **Load** — `SimpleDirectoryReader` reads text from `./data/`
2. **Chunk** — `SentenceSplitter` splits it into ~500-token chunks with 50-token overlap
3. **Embed** — Voyage AI converts each chunk into a vector
4. **Store** — Chroma persists vectors to `./chroma_db/` on disk
5. **Retrieve** — your question is embedded and matched against the top-5 chunks
6. **Generate** — Gemini reads the top chunks and writes the answer

Indexing runs **once**. Every subsequent run reuses `./chroma_db/` and skips
straight to querying.

## Setup

### 1. Install dependencies

```zsh
pip install -r requirements.txt
```

### 2. Get API keys

Both have free tiers:

- **Voyage AI** — https://www.voyageai.com (embeddings)
- **Google Gemini** — https://aistudio.google.com/apikey (LLM)

### 3. Export the keys

```zsh
export VOYAGE_API_KEY="pa-..."
export GOOGLE_API_KEY="AIza..."
```

To make them permanent, add the same lines to `~/.zshrc` and run
`source ~/.zshrc`.

### 4. Add your documents

Drop `.txt`, `.pdf`, or `.md` files into `./data/`. A sample file is
included so you can run the project immediately.

### 5. Run

```zsh
python main.py
```

First run builds the index (see *Performance notes* below). Subsequent
runs skip indexing and go straight to the prompt:

```
Ask questions (type 'quit' to exit):

You: who is midas's father?

Answer: Midas's father is Gordias.

Sources:
  [1] score=0.357 | Skipping and shouting with joy Midas beheld what had once...
  [2] score=0.352 | ...imprecation to the heav...
  [3] score=0.351 | The great Gordian knot lay unsolved for more than a thousand...
------------------------------------------------------------
```

## Configuration

All tunables live at the top of `main.py`:

```python
CHUNK_SIZE = 500         # tokens per chunk
CHUNK_OVERLAP = 50       # tokens shared between consecutive chunks
TOP_K = 5                # chunks fetched from the vector store
RERANK_TOP = 3           # chunks shown in the "Sources" output

EMBED_MODEL = "voyage-3.5-lite"
LLM_MODEL = "gemini-3.8-flash"
```

## Performance notes

**First index: ~8 minutes for a small book.**

Voyage AI's free (unverified) tier is throttled to **3 requests per
minute**. The script enforces this with a `ThrottledVoyage` subclass that
sleeps 20 seconds between embedding batches. Once the index is built, this
delay disappears.

**Subsequent runs: instant.**

The Chroma DB at `./chroma_db/` is reused. You'll see:

```
Reusing existing index (165 chunks).
```

**To force a rebuild** (after changing chunk size, embed model, or adding
new documents):

```zsh
rm -rf ./chroma_db
python main.py
```

## Troubleshooting

### Garbage in the retrieved sources

The loader couldn't extract real text from your PDF. This happens with
scanned documents or PDFs with unusual encoding — `SimpleDirectoryReader`
falls back to dumping raw bytes.

Fix: convert the PDF to text yourself first:

```zsh
pip install pymupdf
python -c "
import fitz
doc = fitz.open('data/yourfile.pdf')
with open('data/yourfile.txt', 'w') as f:
    for page in doc:
        f.write(page.get_text())
"
rm data/yourfile.pdf
rm -rf ./chroma_db
python main.py
```

### `404 NOT_FOUND: This model ... is no longer available`

Google retires Gemini model names frequently. The error message usually
includes the replacement name. Update `LLM_MODEL` in `main.py` and rerun —
no need to rebuild the index, since the vector store is independent of the
LLM.

To list every model your key can currently use:

```zsh
python -c "
import os
from google import genai
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for m in client.models.list():
    if 'generateContent' in (m.supported_actions or []):
        print(m.name)
"
```

### `429 RESOURCE_EXHAUSTED`

You've hit an API rate limit. Sources:

- **Voyage** — the free tier allows 3 RPM. The script already throttles
  for this; if you still hit it, increase the sleep in `ThrottledVoyage`
  from 20 to 25 seconds.
- **Gemini** — the chat free tier has its own quota, separate from
  embeddings. Wait until the daily reset, or add a payment method at
  https://aistudio.google.com/apikey.

### `503 UNAVAILABLE`

Transient server overload on Google's side. The script retries up to 3
times with increasing delays. If all attempts fail, it prints a warning
and returns to the prompt — the session continues.

## Project layout

```
ask-my-docs/
├── main.py            # the whole pipeline
├── data/              # your source documents
├── chroma_db/         # vector store (git-ignored, regenerable)
├── requirements.txt
├── .gitignore
└── README.md
```

## What this project demonstrates

- End-to-end RAG pipeline using three independent services
- Chunking strategy (sentence-aware splitting with overlap)
- Persistent vector storage with graceful reuse
- Provider-specific constraints (rate limits, model deprecation) handled
  in code
- Retry logic and error recovery for transient API failures