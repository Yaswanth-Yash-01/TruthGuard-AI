import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from pinecone import Pinecone


from config.settings import (
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

_pinecone_index = None


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"
_PREFERRED_EMBED = [
    "models/gemini-embedding-001",    # detected during ingestion — try first
    "models/text-embedding-004",
    "models/text-multilingual-embedding-002",
    "models/embedding-001",
]
_active_embed_model: str | None = None


def _get_active_model() -> str:
    """Try each candidate with a real probe call; cache the first that responds."""
    global _active_embed_model
    if _active_embed_model:
        return _active_embed_model

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to your .env file.")

    for model in _PREFERRED_EMBED:
        try:
            probe = requests.post(
                _EMBED_BASE.format(model=model),
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={"content": {"parts": [{"text": "probe"}]}},
                timeout=15,
            )
            if probe.ok and probe.json().get("embedding"):
                _active_embed_model = model
                return model
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach generativelanguage.googleapis.com.\n"
                "Check your internet connection, then restart the server with:\n"
                "  uvicorn api.main:app --port 8000   (no --reload)"
            )

    raise RuntimeError(
        "No embedding model accessible. Tried: " + ", ".join(_PREFERRED_EMBED)
    )


def embed_query(query: str) -> list[float]:
    model = _get_active_model()
    resp = requests.post(
        _EMBED_BASE.format(model=model),
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json={"content": {"parts": [{"text": query}]}},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:400]}")
    return resp.json()["embedding"]["values"]


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    query_vector = embed_query(query)
    index = _get_index()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    sources = []
    seen_urls: set[str] = set()

    for match in results.matches:
        meta = match.metadata or {}
        text = meta.get("text", "")
        url = meta.get("url", "")
        title = meta.get("page_title", url)

        chunks.append(
            {
                "text": text,
                "url": url,
                "page_title": title,
                "score": round(float(match.score), 4),
            }
        )

        if url and url not in seen_urls:
            sources.append({"url": url, "title": title})
            seen_urls.add(url)

    context_parts = []
    for c in chunks:
        header = f"[Source: {c['url']} | {c['page_title']}]"
        context_parts.append(f"{header}\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return {
        "context": context,
        "sources": sources,
        "chunks": chunks,
    }
