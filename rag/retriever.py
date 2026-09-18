"""
Retrieval layer.

- Policies: semantic search against the ChromaDB collection built by ingest.py,
  using a relevance-score threshold to decide whether a policy is "relevant enough".
- Tickets: simple keyword/category filter over the existing ticket_queue table
  (no vector search needed for 10 short rows - see project brief).

TOP_K and RELEVANCE_THRESHOLD are configurable via .env (see .env.example),
not hardcoded, so they can be tuned without touching code.
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

from rag.ingest import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, build_index
from database.db import search_ticket_queue

load_dotenv()

# Chroma's relevance score is roughly 0 (unrelated) to 1 (identical). Below this,
# we treat the request as having "no relevant policy" rather than risk generating
# an answer from a weakly-related document.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))
TOP_K = int(os.getenv("TOP_K", "4"))

_vectorstore = None


def get_vectorstore():
    """Load the persisted Chroma collection, building it on first use if missing."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

    if not os.path.isdir(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        # Index hasn't been built yet - build it now so the app works out of the box.
        _vectorstore, _ = build_index()
        return _vectorstore

    _vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return _vectorstore


def retrieve_policies(query: str, k: int = None):
    """
    Search ChromaDB for the top-k policy chunks relevant to `query`.
    Returns a list of dicts: {source, content, score} sorted by relevance (desc).
    Only chunks with score >= RELEVANCE_THRESHOLD are considered "relevant".
    """
    k = k if k is not None else TOP_K
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    policies = []
    for doc, score in results:
        policies.append({
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
            "score": round(float(score), 3),
        })
    return policies


def has_relevant_policy(policies: list) -> bool:
    """A policy list is 'relevant enough' if at least one chunk clears the threshold."""
    return any(p["score"] >= RELEVANCE_THRESHOLD for p in policies)


def relevant_policies_only(policies: list) -> list:
    """Filter down to only the chunks that clear the relevance threshold."""
    return [p for p in policies if p["score"] >= RELEVANCE_THRESHOLD]


def retrieve_tickets(keywords: list):
    """Keyword search over the existing ticket queue in SQLite. Returns list of dicts."""
    return search_ticket_queue(keywords)
