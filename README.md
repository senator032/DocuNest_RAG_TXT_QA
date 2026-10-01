# DocuNest

DocuNest is a private, evidence-first document assistant: ask questions in
plain language and get concise answers with the source chunks that informed
them.

A minimal, working Retrieval-Augmented Generation (RAG) pipeline that answers
questions over a set of local documents, with source-chunk citations, served
through a REST API. Everything runs locally on CPU — no API keys required.

## What it does

- Chunks and embeds source documents with sentence-transformer embeddings
  (`all-MiniLM-L6-v2`) into a FAISS vector index.
- Uses LangChain to orchestrate retrieval, prompt construction, and
  generation with a local Hugging Face LLM (`google/flan-t5-base`).
- Returns the source chunks used for each answer, so responses are
  traceable back to the original document text.
- Serves the whole pipeline through a small FastAPI REST API, with
  interactive docs auto-generated at `/docs`.
- Ships with a small web UI (served at `/`) for asking questions and
  seeing cited sources, plus uploading new `.txt` documents that are
  chunked, embedded, and merged into the FAISS index on the fly — no
  restart required.

The included starter corpus is a small backend engineering wiki — five
reference docs on REST API design, caching strategies, database
indexing, authentication/authorization, and distributed systems basics —
the kind of internal knowledge base RAG is actually used for in practice,
with enough real technical depth to exercise retrieval across multiple
documents.

## Architecture

```
                     ingest.py (offline, run once)
  data/docs/*.txt ──▶ RecursiveCharacterTextSplitter ──▶ chunks
                                                            │
                                              all-MiniLM-L6-v2 embeddings
                                                            │
                                                            ▼
                                                    FAISS index
                                                 (saved to data/index/)

                     app.py / rag_chain.py (online, per request)

  HTTP POST /query ──▶ FastAPI ──▶ answer_question(question)
                                        │
                                        ▼
                              FAISS similarity search
                              (top-k relevant chunks)
                                        │
                                        ▼
                      LangChain prompt: question + retrieved context
                                        │
                                        ▼
                          flan-t5-base (local HF model)
                                        │
                                        ▼
                        {"answer": "...", "sources": [...]}

  HTTP POST /documents ──▶ FastAPI ──▶ save .txt to data/docs/
                                              │
                                              ▼
                                   re-run ingest.py's pipeline
                                   (chunk -> embed -> FAISS)
                                              │
                                              ▼
                              hot-swap the running retriever
                              (no server restart, no model reload)
```

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Build the index

Run this once (and again any time `data/docs/` changes):

```bash
python src/ingest.py
```

This reads every `.txt` file in `data/docs/`, splits it into ~500-character
chunks with 50-character overlap, embeds the chunks with
`all-MiniLM-L6-v2`, and saves a FAISS index to `data/index/`.

## Run the API

```bash
python src/app.py
```

or, equivalently:

```bash
uvicorn app:app --app-dir src --host 0.0.0.0 --port 5000
```

The first request will be slow (models load lazily on first use); the
server listens on `http://localhost:5000`. Interactive OpenAPI docs are
available at `http://localhost:5000/docs`.

### Web UI

Open `http://localhost:5000/` for a small chat interface: ask a question,
see the generated answer, and expand the "sources" panel to see exactly
which document chunks it was drawn from. The sidebar lists the currently
ingested documents and lets you upload a new `.txt` file, which is
immediately chunked, embedded, and merged into the FAISS index — no
restart needed.

### Health check

```bash
curl http://localhost:5000/health
```

```json
{"status": "ok"}
```

### Ask a question

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What status code should a REST API return when a resource is successfully created?"}'
```

```json
{
  "answer": "201 Created",
  "sources": [
    {
      "chunk_text": "Status codes communicate outcome precisely. 200 OK is a generic success response; 201 Created should be returned when a POST successfully creates a new resource, typically along with a Location header pointing to it...",
      "source_file": "rest_api_design.txt"
    }
  ]
}
```

### Upload a new document

```bash
curl -X POST http://localhost:5000/documents \
  -F "file=@my_new_doc.txt"
```

```json
{
  "documents": ["auth_and_authorization.txt", "caching_strategies.txt", "database_indexing.txt", "distributed_systems_basics.txt", "my_new_doc.txt", "rest_api_design.txt"],
  "chunks_indexed": 44
}
```

The new document is immediately searchable — no need to re-run `ingest.py`
or restart the server.

## Run the tests

```bash
python src/ingest.py   # build the index first, if you haven't already
pytest
```

## Deploying

The app is a standard Python web service — no Dockerfile required. A
`Procfile` is included for platforms that use one (Railway, Heroku-style
buildpacks):

```
web: python src/ingest.py && uvicorn app:app --app-dir src --host 0.0.0.0 --port $PORT
```

`data/index/` is a build artifact, not source, so it isn't committed — the
start command rebuilds it from `data/docs/` on every deploy before the
server starts.

Notes for hosting this:
- **Memory**: `torch` + `transformers` + the embedding and generation
  models need roughly 1–1.5GB of RAM. Free tiers capped at 512MB (e.g.
  Render's free/starter tiers) can OOM — pick a plan with at least 1GB.
- **Cold start**: the first request after a deploy or idle period is slow,
  since models load lazily and (on a fresh instance) get downloaded from
  Hugging Face.
- **Uploads aren't persistent**: documents added through the web UI's
  upload feature are written to the container's local filesystem, which
  most PaaS platforms reset on every redeploy. For permanent additions,
  add the file to `data/docs/` and redeploy.

## Project structure

```
DocuNest/
  Procfile             deploy start command (rebuild index, then run uvicorn)
  data/
    docs/            sample source documents (.txt)
    index/            generated FAISS index (not committed)
  src/
    ingest.py         chunk -> embed -> build & save FAISS index
    rag_chain.py       LangChain retrieval + prompt + generation
    app.py              FastAPI REST API (+ serves the web UI)
    static/
      index.html         chat UI: ask questions, view citations, upload docs
  tests/
    test_rag_chain.py smoke test against the sample corpus
    test_app.py        API tests: health, document listing, upload -> query
```
