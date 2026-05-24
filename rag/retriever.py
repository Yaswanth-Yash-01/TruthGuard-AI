import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cohere
from pinecone import Pinecone
from opentelemetry import trace as _otel_trace

from config.settings import COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL, TOP_K

_tracer = _otel_trace.get_tracer(__name__)

_pinecone_index = None
_cohere_client: cohere.Client | None = None


def _get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY is empty — add it to your .env file.")
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


def _get_active_model() -> str:
    return EMBEDDING_MODEL


def embed_query(query: str) -> list[float]:
    with _tracer.start_as_current_span("embed_query") as span:
        span.set_attribute("query.length", len(query))
        try:
            co = _get_cohere_client()
            response = co.embed(
                texts=[query],
                model=EMBEDDING_MODEL,
                input_type="search_query",
            )
            embedding = response.embeddings[0]
            span.set_attribute("embedding.dimensions", len(embedding))
            return embedding
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
            text  = meta.get("text", "")
            url   = meta.get("url", "")
            title = meta.get("page_title", url)

            chunks.append({
                "text": text,
                "url": url,
                "page_title": title,
                "score": round(float(match.score), 4),
            })

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

        return {"context": context, "sources": sources, "chunks": chunks}
