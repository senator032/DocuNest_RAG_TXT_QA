"""
Smoke test for the RAG pipeline. Requires the FAISS index to already exist
(run `python src/ingest.py` before running these tests).
"""

import os

import pytest

from rag_chain import (
    INDEX_DIR,
    MAX_HISTORY_TURNS,
    MAX_INPUT_TOKENS,
    _is_vague_followup,
    answer_question,
    get_pipeline,
)

KNOWN_QUESTION = "What status code should a REST API return when a resource is successfully created?"

needs_index = pytest.mark.skipif(
    not os.path.isdir(INDEX_DIR),
    reason="FAISS index not found; run `python src/ingest.py` first.",
)


@pytest.mark.parametrize(
    "question",
    [
        "can you explain",
        "why",
        "go on",
        "tell me more",
        "what about it",
        "more",
        "how does it work",
        "why is that",
    ],
)
def test_detects_vague_followups(question):
    assert _is_vague_followup(question)


@pytest.mark.parametrize(
    "question",
    [
        "explain indexing",
        "define caching",
        "what is a JWT",
        KNOWN_QUESTION,
    ],
)
def test_topical_questions_are_not_treated_as_followups(question):
    # These carry their own topic, so retrieval must not be diluted with
    # conversation history.
    assert not _is_vague_followup(question)


@needs_index
def test_answer_has_expected_shape():
    result = answer_question(KNOWN_QUESTION)

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str) and result["answer"] != ""
    assert isinstance(result["sources"], list) and len(result["sources"]) > 0

    for source in result["sources"]:
        assert "chunk_text" in source
        assert "source_file" in source


@needs_index
def test_retrieves_relevant_source_document():
    result = answer_question(KNOWN_QUESTION)

    source_files = {s["source_file"] for s in result["sources"]}
    assert "rest_api_design.txt" in source_files


@needs_index
def test_answer_mentions_known_fact():
    result = answer_question(KNOWN_QUESTION)

    assert "201" in result["answer"]


@needs_index
def test_prompt_stays_within_model_input_limit():
    """A long conversation must not push the prompt past flan-t5's 512-token
    encoder limit, which would silently truncate the retrieved context."""
    pipeline = get_pipeline()
    history = [
        {"question": "What is caching? " * 20, "answer": "A stored result. " * 40}
        for _ in range(MAX_HISTORY_TURNS + 2)
    ]

    history_text = pipeline._format_history(history)
    docs = pipeline.vectorstore.similarity_search("caching", k=4)
    selected = pipeline._select_docs_within_budget(docs, history_text, KNOWN_QUESTION)

    prompt = pipeline.prompt.format(
        context="\n\n".join(d.page_content for d in selected),
        history=history_text,
        input=KNOWN_QUESTION,
    )
    assert selected
    assert pipeline._token_count(prompt) <= MAX_INPUT_TOKENS
