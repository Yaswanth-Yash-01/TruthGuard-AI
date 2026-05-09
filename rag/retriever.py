import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from opentelemetry import trace as _otel_trace

_tracer = _otel_trace.get_tracer(__name__)


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
_embeddings_client: GoogleGenerativeAIEmbeddings | None = None


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


def _get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GoogleGenerativeAIEmbeddings(
            model=_get_active_model(),
            google_api_key=GEMINI_API_KEY,
        )
    return _embeddings_client


def embed_query(query: str) -> list[float]:
    with _tracer.start_as_current_span("embed_query") as span:
        span.set_attribute("query.length", len(query))
        try:
            result = _get_embeddings_client().embed_query(query)
            span.set_attribute("embedding.dimensions", len(result))
            return result
        except Exception as exc:
            span.record_exception(exc)
            raise RuntimeError(f"Embedding failed: {exc}") from exc


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    with _tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("query", query[:200])
        span.set_attribute("top_k", top_k)

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

        span.set_attribute("results.count", len(results.matches))
        if results.matches:
            span.set_attribute("results.top_score", round(float(results.matches[0].score), 4))
        span.set_attribute("context.length", len(context))

        return {
            "context": context,
            "sources": sources,
            "chunks": chunks,
        }
