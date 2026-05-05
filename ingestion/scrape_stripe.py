import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pinecone import Pinecone, ServerlessSpec

from config.settings import (
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_PAGES,
)

SEED_URLS = [
    "https://stripe.com/docs/payments",
    "https://stripe.com/docs/api",
    "https://stripe.com/docs/webhooks",
    "https://stripe.com/docs/billing",
    "https://stripe.com/docs/connect",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 50]


_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"
_LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_PREFERRED_EMBED = [
    "models/text-embedding-004",
    "models/text-multilingual-embedding-002",
    "models/embedding-001",
]
_active_embed_model: str | None = None


def _get_active_model() -> str:
    """Use ListModels to discover which embedding model is available, then cache it."""
    global _active_embed_model
    if _active_embed_model:
        return _active_embed_model

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is empty — add it to your .env file.")

    list_resp = requests.get(
        _LIST_MODELS_URL,
        params={"key": GEMINI_API_KEY},
        timeout=30,
    )
    if not list_resp.ok:
        raise RuntimeError(
            f"ListModels failed ({list_resp.status_code}):\n{list_resp.text[:400]}\n"
            "Your API key may be invalid. Check https://aistudio.google.com/app/apikey"
        )

    all_models = list_resp.json().get("models", [])
    embed_capable = [
        m["name"] for m in all_models
        if "embedContent" in m.get("supportedGenerationMethods", [])
    ]

    if not embed_capable:
        names = [m["name"] for m in all_models]
        raise RuntimeError(
            "No embedContent-capable models found for this API key.\n"
            f"All available models ({len(names)}): {names}"
        )

    # Use preferred model if available, otherwise take first capable model.
    for pref in _PREFERRED_EMBED:
        if pref in embed_capable:
            print(f"  Embedding model: {pref}")
            _active_embed_model = pref
            return pref

    model = embed_capable[0]
    print(f"  Embedding model (auto-detected): {model}")
    _active_embed_model = model
    return model


def _embed_one(text: str) -> list[float]:
    model = _get_active_model()
    for attempt in range(6):
        resp = requests.post(
            _EMBED_BASE.format(model=model),
            params={"key": GEMINI_API_KEY},
            headers={"Content-Type": "application/json"},
            json={"content": {"parts": [{"text": text}]}},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 15 * (2 ** attempt)   # 15s → 30s → 60s → 120s → 240s → 480s
            print(f"  Rate limited (429) — waiting {wait}s (attempt {attempt + 1}/6)...")
            time.sleep(wait)
            continue
        if not resp.ok:
            raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:400]}")
        return resp.json()["embedding"]["values"]
    raise RuntimeError("Embedding still rate-limited after 6 retries.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for t in texts:
        vectors.append(_embed_one(t))
        time.sleep(0.7)   # 0.7 s gap ≈ 85 calls/min, safely under the 100 RPM free-tier cap
    return vectors


def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find(class_=lambda c: c and any(x in c for x in ["content", "docs", "article"]))
        or soup.body
    )

    if main:
        text = main.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())


def get_stripe_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if (
            parsed.netloc in ("stripe.com", "www.stripe.com")
            and parsed.path.startswith("/docs")
            and "#" not in full
            and "?" not in full
        ):
            links.append(full.split("#")[0])
    return links


def init_pinecone(dim: int) -> object:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_names = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME in existing_names:
        info = pc.describe_index(PINECONE_INDEX_NAME)
        if info.dimension != dim:
            print(
                f"  Index dimension mismatch "
                f"(existing={info.dimension}, model={dim}). Recreating..."
            )
            pc.delete_index(PINECONE_INDEX_NAME)
            existing_names = []  # force recreation below

    if PINECONE_INDEX_NAME not in existing_names:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim={dim}) ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready.")

    return pc.Index(PINECONE_INDEX_NAME)


def scrape_and_ingest():
    print("Probing embedding models...")
    _test_vec = _embed_one("test string")
    embed_dim = len(_test_vec)
    print(f"  Dimension: {embed_dim}  OK")

    index = init_pinecone(embed_dim)

    visited: set[str] = set()
    to_visit: list[str] = list(SEED_URLS)
    pages_processed = 0
    total_vectors = 0

    while to_visit and pages_processed < MAX_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"[{pages_processed + 1}/{MAX_PAGES}] Scraping {url}")
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  Skipped ({exc})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else url

        text = extract_text(soup)
        if len(text) < 200:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        try:
            embeddings = embed_texts(chunks)
        except Exception as exc:
            print(f"  Embedding failed ({exc}), skipping.")
            continue

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vid = hashlib.md5(f"{url}||{i}".encode()).hexdigest()
            vectors.append(
                {
                    "id": vid,
                    "values": emb,
                    "metadata": {
                        "url": url,
                        "page_title": page_title,
                        "chunk_index": i,
                        "text": chunk,
                    },
                }
            )

        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start : batch_start + 100])

        total_vectors += len(vectors)
        print(f"  Upserted {len(vectors)} chunks (total: {total_vectors})")

        for link in get_stripe_links(soup, url):
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        pages_processed += 1
        time.sleep(1)

    print(f"\nIngestion complete. Pages: {pages_processed}, Vectors: {total_vectors}")


if __name__ == "__main__":
    scrape_and_ingest()
