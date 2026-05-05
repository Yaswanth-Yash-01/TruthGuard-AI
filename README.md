# TruthGuard AI

AI-powered Stripe documentation assistant with a two-layer hallucination detection pipeline.
Built with LangGraph, LangChain, Gemini 1.5 Flash, Pinecone, and FastAPI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User / Browser                              │
│                   Dark chat UI  ·  GET /  ·  POST /ask              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ JSON  { question }
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI  (api/main.py)                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LangGraph  StateGraph Pipeline                     │
│                                                                      │
│   ┌───────────┐     ┌────────────┐     ┌──────────────────────┐    │
│   │  Node 1   │     │   Node 2   │     │       Node 3          │    │
│   │ retrieve  │────▶│  generate  │────▶│     fact_check        │    │
│   │           │     │            │     │  (Layer-1 Agent)      │    │
│   └─────┬─────┘     └─────┬──────┘     └──────────┬───────────┘    │
│         │                 │                        │                 │
│   Pinecone DB        Gemini 1.5 Flash        Gemini 1.5 Flash       │
│   vector search      context → answer        verify each claim      │
│         │                                          │                 │
│   Google Embed                                     ▼                 │
│   text-embedding-004                 ┌─────────────────────────┐    │
│                                      │         Node 4           │    │
│                                      │   consistency_guard      │    │
│                                      │   (Layer-2 Agent)        │    │
│                                      └────────────┬────────────┘    │
│                                                   │                  │
│                                             Gemini 1.5 Flash        │
│                                          hallucination_score 0-100  │
│                                                   │                  │
│                                      ┌────────────▼────────────┐    │
│                                      │         Node 5           │    │
│                                      │        compile           │    │
│                                      │  assemble final output   │    │
│                                      └────────────┬────────────┘    │
└───────────────────────────────────────────────────┼─────────────────┘
                                                    │
                              ┌─────────────────────▼───────────────────┐
                              │  { answer, fact_check, consistency,      │
                              │    sources, hallucination_score, label } │
                              └─────────────────────────────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 1.5 Flash (Google AI free tier) |
| Embeddings | Google `text-embedding-004` (768-dim) |
| Vector DB | Pinecone (Serverless, free tier) |
| Pipeline | LangGraph `StateGraph` + LangChain |
| API server | FastAPI + Uvicorn |
| Scraping | requests + BeautifulSoup4 |
| Frontend | Single-page HTML (dark theme) |

---

## Setup

### 1. Create and activate a virtual environment

```bash
cd truthguard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey — free |
| `PINECONE_API_KEY` | https://app.pinecone.io — free Serverless tier |
| `PINECONE_INDEX_NAME` | Any name, e.g. `truthguard` (created automatically) |

### 4. Run the ingestion pipeline

This crawls up to 100 Stripe doc pages, chunks the text, embeds with
`text-embedding-004`, and upserts into Pinecone.

```bash
python ingestion/scrape_stripe.py
```

Expected output:
```
Creating Pinecone index 'truthguard' ...
Index ready.
[1/100] Scraping https://stripe.com/docs/payments
  Upserted 18 chunks (total: 18)
[2/100] Scraping https://stripe.com/docs/payments/accept-a-payment
  Upserted 12 chunks (total: 30)
...
Ingestion complete. Pages: 100, Vectors: 1843
```

> Re-running is safe — vectors are upserted by a stable hash ID so
> duplicate pages are overwritten, not duplicated.

### 5. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## API Reference

### `GET /`

Returns the HTML chat UI.

### `POST /ask`

**Request body:**
```json
{ "question": "How do I create a PaymentIntent?" }
```

**Response:**
```json
{
  "query": "How do I create a PaymentIntent?",
  "answer": "To create a PaymentIntent, call POST /v1/payment_intents ...",
  "sources": [
    { "url": "https://stripe.com/docs/payments/accept-a-payment", "title": "Accept a payment" }
  ],
  "fact_check": {
    "claims": [
      {
        "claim": "PaymentIntents are created with POST /v1/payment_intents",
        "verdict": "SUPPORTED",
        "reason": "Context explicitly states the endpoint."
      }
    ],
    "overall_verdict": "PASS",
    "summary": "All claims are directly supported by the Stripe docs."
  },
  "consistency": {
    "hallucination_score": 8,
    "label": "TRUSTED",
    "final_summary": "The answer is accurate and stays within the bounds of the retrieved context.",
    "additional_flags": []
  }
}
```

### `GET /health`

Returns `{ "status": "ok" }`.

---

## Trust Labels

| Label | Score | Meaning |
|---|---|---|
| **TRUSTED** | 0 – 20 | All claims supported by Stripe docs |
| **SUSPICIOUS** | 21 – 50 | Some claims lack doc support |
| **HALLUCINATED** | 51 – 100 | Significant fabrications or contradictions |

---

## Project Structure

```
truthguard/
├── .env.example                 ← API key template
├── requirements.txt
├── config/
│   └── settings.py              ← central config (loaded from .env)
├── ingestion/
│   └── scrape_stripe.py         ← crawl → chunk → embed → Pinecone
├── rag/
│   ├── retriever.py             ← embed query → Pinecone top-K
│   └── generator.py             ← context + query → Gemini answer
├── agents/
│   ├── fact_checker.py          ← Layer 1: per-claim SUPPORTED/UNSUPPORTED/CONTRADICTED
│   └── consistency_guard.py     ← Layer 2: hallucination_score + TRUSTED/SUSPICIOUS/HALLUCINATED
├── graph/
│   └── pipeline.py              ← LangGraph 5-node StateGraph
├── api/
│   ├── main.py                  ← FastAPI GET / and POST /ask
│   └── templates/
│       └── index.html           ← dark chat UI
└── README.md
```
