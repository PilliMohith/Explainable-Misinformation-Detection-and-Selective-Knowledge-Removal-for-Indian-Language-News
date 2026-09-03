# Retrieval Skeleton (v0)

First module of the misinformation-detection pipeline. Embeds a claim,
retrieves the nearest evidence docs from ChromaDB, and returns them —
no LLM verification yet. That's the next piece.

## What's in here

```
app/
  config.py       # env-driven settings (paths, model name, collection name)
  embeddings.py   # multilingual embedding function (Chroma-compatible)
  vectorstore.py  # ChromaDB client/collection singletons
  main.py         # FastAPI app: /health, /retrieve
scripts/
  seed_kb.py      # loads data/seed_claims.json into ChromaDB
data/
  seed_claims.json  # 10 SYNTHETIC fact-check-style docs (en/hi/te) for testing
```

## Setup (run this locally, not in a network-restricted sandbox)

```bash
pip install -r requirements.txt
python -m scripts.seed_kb          # populates ./chroma_data
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** — that's your UI for now. Try
`POST /retrieve` with a claim like `"drinking hot water cures covid"` and
you should get back the matching seed doc plus its metadata.

## What's tested vs. what isn't (read this before trusting the code blindly)

- ✅ All files syntax-checked clean.
- ✅ ChromaDB usage (`upsert`, `query`, `$and` metadata filters) verified
  end-to-end against the actual installed `chromadb==1.5.9` API, using a
  stub embedding function with the same class shape as the real one.
- ⚠️ **Not tested here:** the real `paraphrase-multilingual-MiniLM-L12-v2`
  model. This sandbox has no route to `huggingface.co` (network
  allowlist doesn't include it) and ran low on disk space installing
  `torch`. First run on your machine will download the model
  (~470MB) — make sure you have normal internet access and a few GB
  free, then it's cached locally for every run after.

If `python -m scripts.seed_kb` fails on first run, it's almost certainly
that model download — check your connection, not the code logic.

## Sample data caveat

`data/seed_claims.json` is **synthetic** — I wrote these as realistic
placeholder fact-checks, not scraped from real fact-checking sites.
Good enough to test retrieval and filters; swap in real data once the
Google Fact Check Explorer API ingestion is built.

## Dependency map for next session

- **Two-step extraction/verification module** (next piece) will `from
  app.vectorstore import get_collection`, call `.query()` the same way
  `main.py` does, then feed results into Groq/Gemini via
  Pydantic/Instructor. It's a new file, doesn't need to touch this one
  except to import from it.
- **Selective deletion endpoint** (also queued) adds `POST
  /forget-source` to `app/main.py`, calling
  `collection.delete(where={"source_id": ...})`. `reset_collection()` in
  `vectorstore.py` is already there as a dev-only full-wipe helper —
  don't use it for the real selective-removal demo, that's the whole
  point of doing metadata-filtered delete instead.

## Budget/latency check

Everything here runs locally and free: ChromaDB (local persistent
client), sentence-transformers (local CPU inference, no external API
calls). No Groq/Gemini calls yet — those get added with the
extraction/verification module next.
