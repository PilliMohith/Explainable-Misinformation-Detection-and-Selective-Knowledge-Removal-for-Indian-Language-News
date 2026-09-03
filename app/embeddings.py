"""
Wraps paraphrase-multilingual-MiniLM-L12-v2 as a Chroma-compatible
embedding function.

Depends on: app/config.py (EMBEDDING_MODEL_NAME)
Depended on by: app/vectorstore.py

Note: the first time this runs, sentence-transformers downloads the model
from the Hugging Face Hub (~470MB) and caches it locally. That download
needs outbound internet access to huggingface.co — if you're running this
in a network-restricted sandbox, do the first run somewhere with open
network access, then the local HF cache can be reused.
"""
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


class MultilingualEmbeddingFunction(EmbeddingFunction):
    """Lazy-loads the model once, reuses it for every embed call."""

    _model = None  # class-level cache so we don't reload per-instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        if MultilingualEmbeddingFunction._model is None:
            MultilingualEmbeddingFunction._model = SentenceTransformer(model_name)
        self.model = MultilingualEmbeddingFunction._model

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine distance behaves well
        )
        return vectors.tolist()
