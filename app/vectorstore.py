"""
ChromaDB client + collection access.

Depends on: app/config.py, app/embeddings.py
Depended on by: scripts/seed_kb.py, app/main.py, and (next session) the
two-step extraction/verification module, which will call get_collection()
to pull evidence before calling Groq/Gemini.

Singletons here mean the embedding model loads once per process, not once
per request — important since model load is the slow part.
"""
import chromadb

from app.config import CHROMA_PATH, COLLECTION_NAME
from app.embeddings import MultilingualEmbeddingFunction

_client = None
_collection = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        ef = MultilingualEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection():
    """Drops and recreates the collection. Dev/testing convenience only —
    the real selective-removal path (next piece) deletes by metadata
    filter, not by wiping the whole collection."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    return get_collection()
