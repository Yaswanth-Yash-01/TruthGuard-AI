import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph.pipeline import run_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    try:
        from rag.retriever import _get_active_model
        emb = await loop.run_in_executor(None, _get_active_model)
        print(f"  Embedding model: {emb}")
    except Exception as exc:
        print(f"  WARNING — embedding warmup: {exc}")
    try:
        from rag.llm_client import get_llm_model
        llm = await loop.run_in_executor(None, get_llm_model)
        print(f"  LLM model:       {llm}")
    except Exception as exc:
        print(f"  WARNING — LLM warmup: {exc}")
    yield


app = FastAPI(title="TruthGuard AI", version="1.0.0", lifespan=lifespan)

_HTML = os.path.join(os.path.dirname(__file__), "templates", "index.html")


class AskRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return FileResponse(_HTML, media_type="text/html")


@app.post("/ask")
async def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")

    try:
        result = run_pipeline(question)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}
