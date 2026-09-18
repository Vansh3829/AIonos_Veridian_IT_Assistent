"""
Ingests the Veridian Corp policy documents into a local ChromaDB collection
using FastEmbed for embeddings.

All of the following are configurable via .env (see .env.example), not hardcoded:
  EMBEDDING_MODEL        - FastEmbed model name
  CHROMA_PERSIST_DIR      - where the vector store is persisted on disk
  CHROMA_COLLECTION_NAME  - Chroma collection name
  CORPUS_DIR              - folder containing the policy .md files
  CHUNK_SIZE / CHUNK_OVERLAP - text splitter settings

Each policy file is short and self-contained (one KB article each). Most files
are well under CHUNK_SIZE and come through as a single chunk; the splitter only
kicks in for a file that happens to exceed it, so a chunk always stays traceable
back to its KB source via metadata.

Run directly to (re)build the index:
    python -m rag.ingest
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "veridian_policies")
CORPUS_DIR = os.getenv("CORPUS_DIR", os.path.join(BASE_DIR, "data", "policies"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Normalize a relative CORPUS_DIR/CHROMA_PERSIST_DIR (e.g. "./data/policies") against
# the project root, so `python -m rag.ingest` works the same from any cwd.
if not os.path.isabs(CORPUS_DIR):
    CORPUS_DIR = os.path.join(BASE_DIR, CORPUS_DIR)
if not os.path.isabs(CHROMA_PERSIST_DIR):
    CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, CHROMA_PERSIST_DIR)

# Kept for backwards-compat with existing imports elsewhere in the project
PERSIST_DIR = CHROMA_PERSIST_DIR
COLLECTION_NAME = CHROMA_COLLECTION_NAME


def load_policy_documents():
    """Read every .md file in CORPUS_DIR into one LangChain Document per file,
    then split any file that exceeds CHUNK_SIZE into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md"))):
        source_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "KB-01-password-reset"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        base_doc = Document(page_content=content, metadata={"source": source_id})

        if len(content) <= CHUNK_SIZE:
            docs.append(base_doc)
        else:
            chunks = splitter.split_documents([base_doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = source_id
                chunk.metadata["chunk"] = i
            docs.extend(chunks)

    return docs


def build_index():
    """Build (or rebuild) the ChromaDB collection from the policy documents."""
    docs = load_policy_documents()
    if not docs:
        raise RuntimeError(f"No policy documents found in {CORPUS_DIR}")

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore, len(docs)


if __name__ == "__main__":
    _, n = build_index()
    print(f"Ingested {n} policy chunks into ChromaDB at {CHROMA_PERSIST_DIR} "
          f"(collection: {CHROMA_COLLECTION_NAME}, model: {EMBEDDING_MODEL})")
