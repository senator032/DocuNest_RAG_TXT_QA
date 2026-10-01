import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DOCS_DIR = os.path.join(ROOT_DIR, "data", "docs")
INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _normalize_sources(documents):
    # Trim the full path to a filename so citations stay readable.
    for doc in documents:
        doc.metadata["source"] = os.path.basename(doc.metadata["source"])
    return documents


def load_documents():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return _normalize_sources(loader.load())


def load_document(path):
    return _normalize_sources(TextLoader(path, encoding="utf-8").load())


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_index(chunks, embeddings=None):
    # Callers with the model already loaded should pass it in.
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(chunks, embeddings)


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}")

    chunks = chunk_documents(documents)
    print(
        f"Split into {len(chunks)} chunk(s) "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    vectorstore = build_index(chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
