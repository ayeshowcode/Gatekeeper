"""
Retriever: loads documents from data/docs/, chunks them, embeds them with OpenAI,
and stores them in a ChromaDB vector store. Exposes retrieve(question, k=3) which
returns the k most semantically similar chunks for a given question.
"""

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
client = OpenAI()
COLLECTION_NAME = "gatekeeper"
CHUNK_SIZE = 500
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")

_chroma = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
)


def _embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [r.embedding for r in response.data]


def _chunk(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            for i in range(0, len(para), CHUNK_SIZE):
                chunks.append(para[i : i + CHUNK_SIZE])
    return chunks


def build_index() -> None:
    """Read all .txt files in data/docs/, chunk, embed, and store in ChromaDB."""
    try:
        _chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = _chroma.create_collection(COLLECTION_NAME)

    docs, ids, metas = [], [], []
    chunk_id = 0
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in _chunk(text):
            docs.append(chunk)
            ids.append(f"chunk_{chunk_id}")
            metas.append({"source": fname})
            chunk_id += 1

    embeddings = _embed(docs)
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
    print(f"Indexed {chunk_id} chunks from {DOCS_DIR}")


def retrieve(question: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant chunks for a question."""
    collection = _chroma.get_collection(COLLECTION_NAME)
    embedding = _embed([question])[0]
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]


if __name__ == "__main__":
    build_index()
    test_question = "How many moons does Jupiter have?"
    chunks = retrieve(test_question)
    print(f"\nQuery: {test_question}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk}")
