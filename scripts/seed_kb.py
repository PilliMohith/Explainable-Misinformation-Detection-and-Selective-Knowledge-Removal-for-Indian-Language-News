"""
One-off / re-runnable script to (re)load data/seed_claims.json into the
ChromaDB collection. Safe to re-run — uses upsert, so re-running just
overwrites existing ids instead of duplicating them.

Run from the project root:
    python -m scripts.seed_kb

Depends on: app/vectorstore.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import get_collection

SEED_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "seed_claims.json",
)


def seed():
    with open(SEED_FILE, encoding="utf-8") as f:
        docs = json.load(f)

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    print(f"Seeded {len(ids)} documents into collection '{collection.name}'.")
    print(f"Collection now has {collection.count()} total documents.")


if __name__ == "__main__":
    seed()
