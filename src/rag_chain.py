import os
import re
import sys
import threading

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline as hf_pipeline_factory

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"
RETRIEVER_K = 4

# flan-t5-base's encoder caps input at 512 tokens; budget the prompt
# explicitly so the tokenizer never silently truncates the context.
MAX_INPUT_TOKENS = 512
CONTEXT_TOKEN_BUDGET_MARGIN = 24
MIN_CONTEXT_TOKENS = 96

MAX_HISTORY_TURNS = 3
MAX_HISTORY_TOKENS = 120
MAX_HISTORY_TEXT_CHARS = 160

NO_ANSWER = "I don't know based on the provided documents."

PROMPT_TEMPLATE = """You are the research assistant for a document workspace. \
Answer using ONLY the evidence in the context below. Always answer in a \
complete, clear sentence — never a single word. If the context does not \
contain the answer, say "I don't know based on the provided documents." If the \
question alone is unclear (e.g. "can you explain?"), use the conversation \
so far to figure out what the user is asking about.

Context:
{context}

Conversation so far:
{history}

Question: {input}

Answer in a full sentence:"""

CONTENTLESS_WORDS = {
    "a", "about", "again", "an", "and", "are", "be", "can", "clarify",
    "continue", "could", "did", "do", "does", "elaborate", "expand",
    "explain", "go", "how", "i", "in", "is", "it", "me", "mean", "more",
    "of", "ok", "on", "please", "really", "so", "sure", "tell", "that",
    "the", "them", "then", "these", "they", "this", "those", "to", "was",
    "what", "why", "you",
}

REFERENCE_WORDS = {"it", "that", "them", "these", "they", "this", "those"}

MAX_REFERENTIAL_QUESTION_WORDS = 5


def _is_vague_followup(question: str) -> bool:
    """True when a question names no topic of its own, so it must be about the previous turn."""
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return True

    if all(word in CONTENTLESS_WORDS for word in words):
        return True

    # Short questions leaning on a bare pronoun ("how does it work?").
    return len(words) <= MAX_REFERENTIAL_QUESTION_WORDS and any(
        word in REFERENCE_WORDS for word in words
    )


def _clip(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


class RAGPipeline:
    """FAISS retrieval plus a local flan-t5 generator, returning answers with their source chunks."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        text_generation_pipeline = hf_pipeline_factory(
            "text2text-generation",
            model=GENERATION_MODEL_NAME,
            max_new_tokens=200,
        )
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        self.tokenizer = text_generation_pipeline.tokenizer
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # add_documents() mutates the vectorstore in place while request
        # threads may be searching it.
        self._index_lock = threading.RLock()
        self.reload_index()

    def reload_index(self):
        if not os.path.isdir(INDEX_DIR):
            raise FileNotFoundError(
                f"No FAISS index found at {INDEX_DIR}. "
                "Run `python src/ingest.py` first."
            )

        vectorstore = FAISS.load_local(
            INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True
        )
        with self._index_lock:
            self.vectorstore = vectorstore

    def add_documents(self, chunks: list[Document]) -> int:
        """Embed only the new chunks and merge them into the live index."""
        with self._index_lock:
            self.vectorstore.add_documents(chunks)
            os.makedirs(INDEX_DIR, exist_ok=True)
            self.vectorstore.save_local(INDEX_DIR)
            return self.vectorstore.index.ntotal

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _format_history(self, history: list[dict]) -> str:
        # Refusals carry no information and prime the model to refuse again.
        usable = [turn for turn in history if not turn["answer"].startswith("I don't know")]
        if not usable:
            return "(none)"

        blocks = []
        used = 0
        for turn in reversed(usable[-MAX_HISTORY_TURNS:]):
            question = _clip(turn["question"], MAX_HISTORY_TEXT_CHARS)
            answer = _clip(turn["answer"], MAX_HISTORY_TEXT_CHARS)
            block = f"Q: {question}\nA: {answer}"

            cost = self._token_count(block)
            if blocks and used + cost > MAX_HISTORY_TOKENS:
                break
            blocks.append(block)
            used += cost

        return "\n".join(reversed(blocks))

    def _truncate_doc(self, doc: Document, max_tokens: int) -> Document:
        token_ids = self.tokenizer.encode(doc.page_content, add_special_tokens=False)
        text = self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
        return Document(page_content=text, metadata=doc.metadata)

    def _select_docs_within_budget(
        self, docs: list[Document], history_text: str, question: str
    ) -> list[Document]:
        """Keep the most relevant chunks that fit in what's left of the input window."""
        if not docs:
            return []

        scaffold_cost = self._token_count(
            self.prompt.format(context="", history=history_text, input=question)
        )
        budget = max(
            MAX_INPUT_TOKENS - scaffold_cost - CONTEXT_TOKEN_BUDGET_MARGIN,
            MIN_CONTEXT_TOKENS,
        )

        kept: list[Document] = []
        used = 0
        for doc in docs:
            cost = self._token_count(doc.page_content)
            if used + cost > budget:
                # Truncate rather than emit an over-budget prompt.
                if not kept:
                    kept.append(self._truncate_doc(doc, budget))
                break
            kept.append(doc)
            used += cost

        return kept

    def answer_question(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []

        # History is only worth its token cost when the question can't stand
        # alone; on a topical question it just crowds out retrieved context.
        needs_history = bool(history) and _is_vague_followup(question)
        history_text = self._format_history(history) if needs_history else "(none)"

        if needs_history:
            retrieval_query = f"{history[-1]['question']} {question}"
        else:
            retrieval_query = question

        with self._index_lock:
            retrieved_docs = self.vectorstore.similarity_search(
                retrieval_query, k=RETRIEVER_K
            )

        docs = self._select_docs_within_budget(retrieved_docs, history_text, question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt_text = self.prompt.format(
            context=context, history=history_text, input=question
        )
        answer = self.llm.invoke(prompt_text).strip()

        sources = [
            {
                "chunk_text": doc.page_content,
                "source_file": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ]

        return {"answer": answer, "sources": sources}


_pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    # FastAPI runs sync endpoints in a threadpool, so concurrent first
    # requests would otherwise each load their own copy of both models.
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    return get_pipeline().answer_question(question, history=history)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or (
        "What status code should a REST API return when a resource is created?"
    )
    response = answer_question(query)
    print("Q:", query)
    print("A:", response["answer"])
    print("Sources:")
    for source in response["sources"]:
        print(f"  - {source['source_file']}: {source['chunk_text'][:80]}...")
