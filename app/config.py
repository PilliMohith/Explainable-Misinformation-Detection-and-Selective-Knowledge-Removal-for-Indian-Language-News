"""
Central config for the retrieval skeleton.

Everything here is overridable via environment variables so the same code
works locally and on a free-tier host (Render/Railway/HF Spaces) without
code changes.
"""
import os

# Where ChromaDB persists its local files. On free hosts with ephemeral
# disks, point this at a mounted volume if one is available, otherwise
# the KB resets on redeploy (fine for a prototype, not for production).
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# Single collection for now. If we later split by language or by
# claim-domain, this becomes a prefix instead of a fixed name.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "misinfo_kb")

# paraphrase-multilingual-MiniLM-L12-v2: chosen in the project scope doc
# for local, free, multilingual embeddings. Do not swap without updating
# the project context doc.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)
