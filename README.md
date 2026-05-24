# TruthGuard AI : Live - https://truthguardstrip.netlify.app

AI-powered Stripe documentation assistant with a two-layer hallucination detection pipeline.
Built with LangGraph, Cohere, Pinecone, and FastAPI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User / Browser                              │
│              Chat UI  ·  WebSocket /ws/chat  ·  GET /               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ WebSocket { type: "ask", question }
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI  (api/main.py)                        │
│               WebSocket streaming + Human Review Coordinator         │
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
│   Pinecone DB       Cohere command-r-plus     Cohere command-r-plus │
│   vector search     context → answer          verify each claim     │
│         │                                          │                 │
│   Cohere Embed                                     ▼                 │
│   embed-english-v3.0                 ┌─────────────────────────┐    │
│   (1024-dim)                         │         Node 4           │    │
│                                      │   consistency_guard      │    │
│                                      │   (Layer-2 Agent)        │    │
│                                      └────────────┬────────────┘    │
│                                                   │                  │
│                                        Cohere command-r-plus        │
│                                       hallucination_score 0-100     │
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
| LLM | Cohere `command-r-plus-08-2024` |
| Embeddings | Cohere `embed-english-v3.0` (1024-dim) |
| Vector DB | Pinecone (Serverless, free tier) |
| Pipeline | LangGraph `StateGraph` |
| API server | FastAPI + Uvicorn (WebSocket streaming) |
| Scraping | requests + BeautifulSoup4 |
| Frontend | Single-page HTML (black & white theme) |

---

## Setup

### 1. Create and activate a virtual environment

```bash
cd truthguard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
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
| `COHERE_API_KEY` | https://dashboard.cohere.com — free trial available |
| `PINECONE_API_KEY` | https://app.pinecone.io — free Serverless tier |
| `PINECONE_INDEX_NAME` | Any name, e.g. `truthguard` (created automatically) |

### 4. Run the ingestion pipeline

Crawls up to 100 Stripe doc pages, chunks the text, embeds with
`embed-english-v3.0` (1024-dim), and upserts into Pinecone.

```bash
python ingestion/scrape_stripe.py
```

Expected output:
```
Embedding model: embed-english-v3.0
Probing embedding API...
  Dimension: 1024  OK
Creating Pinecone index 'truthguard' (dim=1024)...
Index ready.
[1/100] Scraping https://stripe.com/docs/payments
  Upserted 5 chunks (total: 5)
...
Ingestion complete. Pages: 100, Vectors: 1800
```

> Re-running is safe — vectors are upserted by a stable hash ID so
> duplicate pages are overwritten, not duplicated.

### 5. Start the API server

```bash
uvicorn api.main:app --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Deployment

### Backend — Render

The `render.yaml` at the root configures the web service automatically.

1. Push to GitHub
2. New Web Service on Render → connect repo
3. Add environment variables in the Render dashboard:
   - `COHERE_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`

### Frontend — Netlify

Drag and drop the `dist/` folder onto Netlify. The `_redirects` file
routes `/agent` to `agent_portal.html`.

---

## API Reference

### `GET /`

Returns the HTML chat UI.

### `GET /agent`

Returns the human agent review portal.

### `WebSocket /ws/chat/{session_id}`

Stream-based chat endpoint. Send:
```json
{ "type": "ask", "question": "How do I create a PaymentIntent?" }
```

Receives a sequence of events: `stream_start` → `token` (repeated) →
`stream_end` → `trust_card` (or `needs_review` if hallucinated).

### `POST /ask`

REST fallback. Request body:
```json
{ "question": "How do I create a PaymentIntent?" }
```

### `GET /health`

Returns `{ "status": "ok", "pending_human_reviews": 0 }`.

---

## Trust Labels

| Label | Score | Meaning |
|---|---|---|
| **TRUSTED** | 0 – 20 | All claims supported by Stripe docs |
| **SUSPICIOUS** | 21 – 50 | Some claims lack doc support |
| **HALLUCINATED** | 51 – 100 | Significant fabrications detected — routed to human review |

---

## Project Structure

```
truthguard/
├── .env.example                 ← API key template
├── requirements.txt
├── render.yaml                  ← Render deployment config
├── config/
│   └── settings.py              ← central config (loaded from .env)
├── ingestion/
│   └── scrape_stripe.py         ← crawl → chunk → embed → Pinecone
├── rag/
│   ├── retriever.py             ← Cohere embed query → Pinecone top-K
│   ├── generator.py             ← context + query → Cohere answer (streamed)
│   └── llm_client.py            ← Cohere client (sync + async streaming)
├── agents/
│   ├── fact_checker.py          ← Layer 1: per-claim SUPPORTED/UNSUPPORTED/CONTRADICTED
│   ├── consistency_guard.py     ← Layer 2: hallucination_score + label
│   └── confidence_scorer.py     ← confidence score 0-100
├── graph/
│   └── pipeline.py              ← LangGraph 5-node StateGraph
├── api/
│   ├── main.py                  ← FastAPI + WebSocket + human review coordinator
│   └── templates/
│       ├── index.html           ← Chat UI (black & white)
│       └── agent_portal.html    ← Human review portal
├── frontend/                    ← Netlify deploy folder
├── dist/                        ← Netlify drag-and-drop folder
└── README.md
```
