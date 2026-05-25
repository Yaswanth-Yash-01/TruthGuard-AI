import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import hashlib
import requests
import cohere
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pinecone import Pinecone, ServerlessSpec

from config.settings import (
    COHERE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_PAGES,
)

SEED_URLS = [
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/api",
    "https://docs.stripe.com/webhooks",
    "https://docs.stripe.com/billing",
    "https://docs.stripe.com/connect",
    "https://docs.stripe.com/checkout",
    "https://docs.stripe.com/payment-links",
    "https://docs.stripe.com/invoicing",
    "https://docs.stripe.com/billing/subscriptions/overview",
    "https://docs.stripe.com/tax",
    "https://docs.stripe.com/radar",
    "https://docs.stripe.com/terminal",
    "https://docs.stripe.com/issuing",
    "https://docs.stripe.com/treasury",
    "https://docs.stripe.com/identity",
    "https://docs.stripe.com/disputes",
    "https://docs.stripe.com/refunds",
    "https://docs.stripe.com/reports",
    "https://docs.stripe.com/payouts",
    "https://docs.stripe.com/security",
    "https://docs.stripe.com/testing",
    "https://docs.stripe.com/error-codes",
    "https://docs.stripe.com/currencies",
    "https://docs.stripe.com/keys",
]

# Stripe migrated its docs to docs.stripe.com; the legacy stripe.com/docs/*
# URLs 301-redirect there, so we accept both hosts when crawling.
ALLOWED_HOSTS = ("docs.stripe.com", "stripe.com", "www.stripe.com")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_co: cohere.Client | None = None


def _get_cohere() -> cohere.Client:
    global _co
    if _co is None:
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY is empty — add it to your .env file.")
        _co = cohere.Client(api_key=COHERE_API_KEY)
    return _co


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    co = _get_cohere()
    all_embeddings: list[list[float]] = []
    batch_size = 90  # Cohere limit is 96 per request
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                response = co.embed(
                    texts=batch,
                    model=EMBEDDING_MODEL,
                    input_type="search_document",
                )
                all_embeddings.extend(response.embeddings)
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower() or "too many" in msg.lower():
                    wait = 15 * (2 ** attempt)
                    print(f"  Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Embedding failed: {exc}") from exc
        if i + batch_size < len(texts):
            time.sleep(0.3)
    return all_embeddings


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
    text = main.get_text(separator=" ", strip=True) if main else soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def get_stripe_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        parsed = urlparse(full)
        # docs.stripe.com serves docs at the root (/payments, /api, ...),
        # while the legacy stripe.com host nests them under /docs.
        is_doc = parsed.netloc == "docs.stripe.com" or (
            parsed.netloc in ("stripe.com", "www.stripe.com")
            and parsed.path.startswith("/docs")
        )
        if is_doc and "#" not in full and "?" not in full:
            links.append(full.split("#")[0])
    return links


def init_pinecone(dim: int):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME in existing:
        info = pc.describe_index(PINECONE_INDEX_NAME)
        if info.dimension != dim:
            print(f"  Dimension mismatch (existing={info.dimension}, new={dim}). Recreating index...")
            pc.delete_index(PINECONE_INDEX_NAME)
            existing = []

    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim={dim})...")
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
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Probing embedding API...")
    test_vec = embed_texts(["probe"])
    embed_dim = len(test_vec[0])
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

        # Follow redirects (e.g. stripe.com/docs/* -> docs.stripe.com/*) so
        # link extraction and stored metadata use the canonical final URL.
        final_url = resp.url.split("#")[0]
        visited.add(final_url)

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
            vid = hashlib.md5(f"{final_url}||{i}".encode()).hexdigest()
            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": {
                    "url": final_url,
                    "page_title": page_title,
                    "chunk_index": i,
                    "text": chunk,
                },
            })

        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start : batch_start + 100])

        total_vectors += len(vectors)
        print(f"  Upserted {len(vectors)} chunks (total: {total_vectors})")

        for link in get_stripe_links(soup, final_url):
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        pages_processed += 1
        time.sleep(1)

    print(f"\nIngestion complete. Pages: {pages_processed}, Vectors: {total_vectors}")


if __name__ == "__main__":
    scrape_and_ingest()
