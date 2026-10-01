"""
FastAPI REST API exposing the RAG pipeline, plus a small static frontend.

Endpoints:
    GET  /                  -> chat UI (static/index.html)
    GET  /health             -> {"status": "ok"}
    POST /query               -> {"question": "...", "history": [...]} -> {"answer": "...", "sources": [...]}
    GET  /documents            -> list of currently ingested source documents
    POST /documents             -> upload a new .txt document and re-ingest it
    GET  /docs                   -> interactive OpenAPI docs (auto-generated)

Run with:
    uvicorn app:app --app-dir src
or:
    python src/app.py
"""

import os

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ingest
from rag_chain import answer_question, get_pipeline

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SRC_DIR, "static")

# The upload endpoint is unauthenticated so the public demo stays clickable.
MAX_UPLOAD_BYTES = 256 * 1024

app = FastAPI(
    title="DocuNest",
    description="Evidence-first question answering over a local document workspace.",
)


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    history: list[HistoryTurn] = Field(
        default_factory=list,
        description="Recent question/answer turns, oldest first, for follow-up questions.",
    )


class Source(BaseModel):
    chunk_text: str
    source_file: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentsResponse(BaseModel):
    documents: list[str]


class UploadResponse(BaseModel):
    documents: list[str]
    chunks_indexed: int


def _list_documents() -> list[str]:
    if not os.path.isdir(ingest.DOCS_DIR):
        return []
    return sorted(f for f in os.listdir(ingest.DOCS_DIR) if f.endswith(".txt"))


def _index_document(path: str) -> int:
    """Merge one uploaded document into the live index, returning the new chunk total."""
    chunks = ingest.chunk_documents(ingest.load_document(path))
    return get_pipeline().add_documents(chunks)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required.")

    history = [turn.model_dump() for turn in request.history]
    try:
        return answer_question(question, history=history)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/documents", response_model=DocumentsResponse)
def list_documents():
    return {"documents": _list_documents()}


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile):
    # basename() keeps a crafted filename from escaping data/docs/.
    filename = os.path.basename(file.filename or "")
    if not filename.endswith(".txt") or filename == ".txt":
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    os.makedirs(ingest.DOCS_DIR, exist_ok=True)
    dest_path = os.path.join(ingest.DOCS_DIR, filename)
    if os.path.exists(dest_path):
        raise HTTPException(
            status_code=409,
            detail=f"A document named {filename} already exists. Rename the file and try again.",
        )

    # One byte past the limit is enough to detect an oversized body.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large (limit {MAX_UPLOAD_BYTES // 1024} KB).",
        )
    if not contents.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        chunks_indexed = _index_document(dest_path)
    except Exception:
        # Don't leave a document on disk that isn't in the index.
        os.remove(dest_path)
        raise

    return {"documents": _list_documents(), "chunks_indexed": chunks_indexed}


# Mounted last so it only catches requests not matched by the API routes above.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
