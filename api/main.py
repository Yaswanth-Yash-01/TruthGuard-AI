import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
import asyncio
import time as _time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import requests

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

import telemetry
from telemetry import setup_telemetry, get_tracer

from graph.pipeline import run_pipeline
from rag.retriever import retrieve
from rag.generator import astream_answer
from agents.fact_checker import fact_check
from agents.consistency_guard import consistency_guard
from agents.confidence_scorer import score_confidence

setup_telemetry()
RequestsInstrumentor().instrument()


_INSUFFICIENT_INFO_PHRASE = "i don't have enough information in the retrieved stripe docs to answer this"


def _is_insufficient_info(answer: str) -> bool:
    return _INSUFFICIENT_INFO_PHRASE in answer.lower()


@dataclass
class ReviewTask:
    task_id: str
    session_id: str
    question: str
    pipeline_result: dict[str, Any]
    trigger: str = "HALLUCINATED"  # "HALLUCINATED" | "INSUFFICIENT_INFO"
    status: str = "pending"
    reviewer: str | None = None
    human_answer: str | None = None


class ReviewCoordinator:
    def __init__(self) -> None:
        self.client_sockets: dict[str, WebSocket] = {}
        self.agent_sockets: set[WebSocket] = set()
        self.review_tasks: dict[str, ReviewTask] = {}
        self.lock = asyncio.Lock()

    async def connect_client(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.client_sockets[session_id] = websocket
        await websocket.send_json({"type": "connected", "session_id": session_id})

    async def disconnect_client(self, session_id: str) -> None:
        async with self.lock:
            self.client_sockets.pop(session_id, None)

    async def connect_agent(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.agent_sockets.add(websocket)
        await websocket.send_json(
            {"type": "queue_snapshot", "tasks": self._pending_tasks_snapshot()}
        )

    async def disconnect_agent(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.agent_sockets.discard(websocket)

    async def send_to_client(self, session_id: str, payload: dict[str, Any]) -> None:
        async with self.lock:
            websocket = self.client_sockets.get(session_id)
        if websocket is None:
            return
        await websocket.send_json(payload)

    async def broadcast_to_agents(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            sockets = list(self.agent_sockets)
        stale: list[WebSocket] = []
        for websocket in sockets:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        if stale:
            async with self.lock:
                for websocket in stale:
                    self.agent_sockets.discard(websocket)

    def _serialize_task(self, task: ReviewTask) -> dict[str, Any]:
        result = task.pipeline_result
        return {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "question": task.question,
            "trigger": task.trigger,
            "status": task.status,
            "reviewer": task.reviewer,
            "human_answer": task.human_answer,
            "pipeline_result": result,
            "consistency_label": result.get("consistency", {}).get("label", ""),
            "hallucination_score": result.get("consistency", {}).get("hallucination_score", 0),
            "confidence_score": result.get("confidence", {}).get("confidence_score", 0),
        }

    def _pending_tasks_snapshot(self) -> list[dict[str, Any]]:
        pending = [
            self._serialize_task(task)
            for task in self.review_tasks.values()
            if task.status == "pending"
        ]
        pending.sort(key=lambda item: item["task_id"], reverse=True)
        return pending

    async def refresh_agent_queue(self) -> None:
        await self.broadcast_to_agents(
            {"type": "queue_snapshot", "tasks": self._pending_tasks_snapshot()}
        )

    async def queue_review(
        self,
        session_id: str,
        question: str,
        pipeline_result: dict[str, Any],
        trigger: str = "HALLUCINATED",
    ) -> ReviewTask:
        task = ReviewTask(
            task_id=uuid4().hex[:8],
            session_id=session_id,
            question=question,
            pipeline_result=pipeline_result,
            trigger=trigger,
        )
        async with self.lock:
            self.review_tasks[task.task_id] = task

        if trigger == "INSUFFICIENT_INFO":
            message = (
                "The AI couldn't find enough information in the Stripe docs to answer this. "
                "The question has been sent to a human agent for a direct response."
            )
        else:
            message = (
                "This answer was flagged as hallucinated and has been routed to "
                "the human review portal."
            )

        await self.send_to_client(
            session_id,
            {
                "type": "needs_review",
                "task_id": task.task_id,
                "trigger": trigger,
                "message": message,
                "pipeline_result": pipeline_result,
            },
        )
        await self.refresh_agent_queue()
        return task

    async def resolve_task(
        self,
        task_id: str,
        final_answer: str,
        reviewer: str,
    ) -> dict[str, Any]:
        async with self.lock:
            task = self.review_tasks.get(task_id)
            if task is None:
                raise ValueError("Review task not found.")
            task.status = "resolved"
            task.reviewer = reviewer
            task.human_answer = final_answer

        payload = {
            "query": task.question,
            "answer": final_answer,
            "human_intervention": True,
            "trigger": task.trigger,
            "review_task_id": task.task_id,
            "reviewer": reviewer,
            "original_consistency": task.pipeline_result.get("consistency", {}),
            "original_confidence": task.pipeline_result.get("confidence", {}),
            "sources": task.pipeline_result.get("sources", []),
        }
        if telemetry.human_review_counter:
            telemetry.human_review_counter.add(1, {"trigger": task.trigger})
        await self.send_to_client(task.session_id, {"type": "human_response", "payload": payload})
        await self.refresh_agent_queue()
        return payload


coordinator = ReviewCoordinator()


# Free hosting (e.g. Render) spins the instance down after ~15 min idle.
# A periodic self-ping keeps the inactivity timer from ever firing.
# Render auto-provides RENDER_EXTERNAL_URL; KEEP_AWAKE_URL can override it.
_KEEP_AWAKE_URL = os.getenv("KEEP_AWAKE_URL") or os.getenv("RENDER_EXTERNAL_URL", "")
_KEEP_AWAKE_INTERVAL = int(os.getenv("KEEP_AWAKE_INTERVAL", "600"))  # seconds


async def _keep_awake() -> None:
    if not _KEEP_AWAKE_URL:
        print("  keep-awake: no base URL configured — self-ping disabled.")
        return
    ping_url = _KEEP_AWAKE_URL.rstrip("/") + "/health"
    print(f"  keep-awake: pinging {ping_url} every {_KEEP_AWAKE_INTERVAL}s")
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(_KEEP_AWAKE_INTERVAL)
        try:
            await loop.run_in_executor(None, lambda: requests.get(ping_url, timeout=10))
        except Exception as exc:
            print(f"  keep-awake ping failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    try:
        from rag.retriever import _get_active_model
        print(f"  Embedding model: {_get_active_model()}")
    except Exception as exc:
        print(f"  WARNING — embedding warmup: {exc}")
    try:
        from rag.llm_client import get_llm_model
        print(f"  LLM model:       {get_llm_model()}")
    except Exception as exc:
        print(f"  WARNING — LLM warmup: {exc}")
    keep_awake_task = asyncio.create_task(_keep_awake())
    try:
        yield
    finally:
        keep_awake_task.cancel()


app = FastAPI(title="TruthGuard AI", version="1.1.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

_HTML = os.path.join(os.path.dirname(__file__), "templates", "index.html")
_AGENT_HTML = os.path.join(os.path.dirname(__file__), "templates", "agent_portal.html")


class AskRequest(BaseModel):
    question: str


def _validate_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")
    return question


async def _run_pipeline_async(question: str) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_pipeline, question)


async def _process_question(session_id: str, question: str) -> None:
    t0 = _time.time()
    if telemetry.request_counter:
        telemetry.request_counter.add(1)
    with get_tracer().start_as_current_span("process_question") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("question.length", len(question))
        try:
            loop = asyncio.get_running_loop()

            # Phase 1: retrieve context
            retrieval = await loop.run_in_executor(None, retrieve, question)
            context = retrieval["context"]
            sources = retrieval["sources"]

            # Phase 2: stream answer tokens to the client
            await coordinator.send_to_client(session_id, {"type": "stream_start"})
            full_answer = ""
            async for token in astream_answer(context, question):
                full_answer += token
                await coordinator.send_to_client(session_id, {"type": "token", "text": token})
            await coordinator.send_to_client(session_id, {"type": "stream_end"})

            # Phase 3: check for insufficient info before running expensive verification
            if _is_insufficient_info(full_answer):
                span.set_attribute("outcome", "insufficient_info")
                if telemetry.insufficient_counter:
                    telemetry.insufficient_counter.add(1)
                minimal_result = {
                    "query": question,
                    "answer": full_answer,
                    "sources": sources,
                    "fact_check": {},
                    "consistency": {"label": "INSUFFICIENT_INFO", "hallucination_score": 0, "final_summary": "", "additional_flags": []},
                    "confidence": {},
                }
                await coordinator.queue_review(session_id, question, minimal_result, trigger="INSUFFICIENT_INFO")
                return

            # Phase 4: run verification agents in a thread
            def _verify():
                fc   = fact_check(full_answer, context)
                cg   = consistency_guard(question, full_answer, fc, context)
                conf = score_confidence(question, full_answer, context, fc, cg)
                return fc, cg, conf

            fc_result, cg_result, conf_result = await loop.run_in_executor(None, _verify)

            result = {
                "query": question,
                "answer": full_answer,
                "sources": sources,
                "fact_check": fc_result,
                "consistency": cg_result,
                "confidence": conf_result,
            }

            # Phase 5: route or deliver
            if cg_result.get("label", "").upper() == "HALLUCINATED":
                span.set_attribute("outcome", "hallucinated")
                span.set_attribute("hallucination_score", cg_result.get("hallucination_score", 0))
                if telemetry.hallucination_counter:
                    telemetry.hallucination_counter.add(1)
                await coordinator.queue_review(session_id, question, result, trigger="HALLUCINATED")
                return

            span.set_attribute("outcome", "trusted")
            span.set_attribute("confidence_score", conf_result.get("confidence_score", 0))
            await coordinator.send_to_client(session_id, {"type": "trust_card", "payload": result})

        except Exception as exc:
            span.record_exception(exc)
            await coordinator.send_to_client(
                session_id,
                {"type": "error", "detail": str(exc)},
            )
        finally:
            if telemetry.pipeline_duration:
                telemetry.pipeline_duration.record((_time.time() - t0) * 1000)


@app.get("/")
async def root():
    return FileResponse(_HTML, media_type="text/html")


@app.get("/agent")
async def agent_portal():
    return FileResponse(_AGENT_HTML, media_type="text/html")


@app.post("/ask")
async def ask(body: AskRequest):
    question = _validate_question(body.question)
    try:
        result = await _run_pipeline_async(question)
        review_required = result.get("consistency", {}).get("label", "").upper() == "HALLUCINATED"
        return {
            **result,
            "review_required": review_required,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws/chat/{session_id}")
async def chat_socket(websocket: WebSocket, session_id: str):
    await coordinator.connect_client(session_id, websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            event_type = payload.get("type")
            if event_type == "ask":
                try:
                    question = _validate_question(payload.get("question", ""))
                except HTTPException as exc:
                    await websocket.send_json({"type": "error", "detail": exc.detail})
                    continue
                await websocket.send_json({"type": "processing", "question": question})
                asyncio.create_task(_process_question(session_id, question))
            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unsupported chat event '{event_type}'."}
                )
    except WebSocketDisconnect:
        await coordinator.disconnect_client(session_id)


@app.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket):
    await coordinator.connect_agent(websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            event_type = payload.get("type")
            if event_type == "resolve_task":
                task_id = str(payload.get("task_id", "")).strip()
                final_answer = str(payload.get("answer", "")).strip()
                reviewer = str(payload.get("reviewer", "Human Agent")).strip() or "Human Agent"
                if not task_id or not final_answer:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": "Both task_id and answer are required to resolve a task.",
                        }
                    )
                    continue
                try:
                    await coordinator.resolve_task(task_id, final_answer, reviewer)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                await websocket.send_json(
                    {"type": "resolved", "task_id": task_id, "reviewer": reviewer}
                )
            elif event_type == "refresh":
                await websocket.send_json(
                    {"type": "queue_snapshot", "tasks": coordinator._pending_tasks_snapshot()}
                )
            else:
                await websocket.send_json(
                    {"type": "error", "detail": f"Unsupported agent event '{event_type}'."}
                )
    except WebSocketDisconnect:
        await coordinator.disconnect_agent(websocket)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pending_human_reviews": len(coordinator._pending_tasks_snapshot()),
    }
