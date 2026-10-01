"""
Tests for the FastAPI layer: health check, document listing, and the
upload -> re-index -> query round trip.
"""

import os

import pytest
from fastapi.testclient import TestClient

import ingest
from app import MAX_UPLOAD_BYTES, app
from rag_chain import INDEX_DIR, get_pipeline

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INDEX_DIR),
    reason="FAISS index not found; run `python src/ingest.py` first.",
)

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_list_documents_includes_sample_corpus():
    res = client.get("/documents")
    assert res.status_code == 200
    documents = res.json()["documents"]
    assert "rest_api_design.txt" in documents


def test_upload_reindexes_and_is_queryable():
    upload_name = "test_rate_limiting.txt"
    upload_path = os.path.join(ingest.DOCS_DIR, upload_name)
    content = (
        "Rate Limiting\n\n"
        "A token bucket rate limiter allows up to 100 requests per minute "
        "per API key. Requests beyond that limit receive a 429 Too Many "
        "Requests response."
    )

    try:
        res = client.post(
            "/documents",
            files={"file": (upload_name, content, "text/plain")},
        )
        assert res.status_code == 200
        body = res.json()
        assert upload_name in body["documents"]
        assert body["chunks_indexed"] > 0

        query_res = client.post(
            "/query",
            json={"question": "How many requests per minute does the rate limiter allow per API key?"},
        )
        assert query_res.status_code == 200
        query_body = query_res.json()
        assert "100" in query_body["answer"]
        assert any(
            s["source_file"] == upload_name for s in query_body["sources"]
        )
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        _restore_index()


def _restore_index():
    """Rebuild the index from whatever is currently on disk in data/docs/."""
    pipeline = get_pipeline()
    chunks = ingest.chunk_documents(ingest.load_documents())
    vectorstore = ingest.build_index(chunks, embeddings=pipeline.embeddings)
    vectorstore.save_local(ingest.INDEX_DIR)
    pipeline.reload_index()


def test_upload_rejects_non_txt():
    res = client.post(
        "/documents",
        files={"file": ("notes.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert res.status_code == 400


def test_upload_rejects_duplicate_filename():
    res = client.post(
        "/documents",
        files={"file": ("rest_api_design.txt", "overwritten", "text/plain")},
    )
    assert res.status_code == 409

    # The original document must be untouched by the rejected upload.
    original = os.path.join(ingest.DOCS_DIR, "rest_api_design.txt")
    with open(original, encoding="utf-8") as f:
        assert "overwritten" not in f.read()


def test_upload_rejects_oversized_file():
    oversized = "x" * (MAX_UPLOAD_BYTES + 1)
    res = client.post(
        "/documents",
        files={"file": ("huge.txt", oversized, "text/plain")},
    )
    assert res.status_code == 413
    assert not os.path.exists(os.path.join(ingest.DOCS_DIR, "huge.txt"))


def test_upload_rejects_empty_file():
    res = client.post(
        "/documents",
        files={"file": ("blank.txt", "   \n", "text/plain")},
    )
    assert res.status_code == 400
    assert not os.path.exists(os.path.join(ingest.DOCS_DIR, "blank.txt"))
