import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cohere
from opentelemetry import trace as _otel_trace, metrics as _otel_metrics

from config.settings import COHERE_API_KEY, LLM_MODEL

_tracer = _otel_trace.get_tracer(__name__)
_llm_duration_hist = None

_client: cohere.Client | None = None
_async_client: cohere.AsyncClient | None = None


def _get_llm_duration_hist():
    global _llm_duration_hist
    if _llm_duration_hist is None:
        _llm_duration_hist = _otel_metrics.get_meter("truthguard").create_histogram(
            "truthguard.llm.duration_ms", unit="ms", description="LLM call duration"
        )
    return _llm_duration_hist


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        _client = cohere.Client(api_key=COHERE_API_KEY)
    return _client


def _get_async_client() -> cohere.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = cohere.AsyncClient(api_key=COHERE_API_KEY)
    return _async_client


def get_llm_model() -> str:
    return LLM_MODEL


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    _attempt: int = 0,
) -> str:
    if _attempt >= 5:
        raise RuntimeError("Cohere is currently unavailable. Try again in a moment.")

    t0 = time.time()
    with _tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", LLM_MODEL)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.attempt", _attempt)
        span.set_attribute("llm.prompt_chars", len(system_prompt) + len(user_prompt))
        try:
            client = _get_client()
            kwargs: dict = dict(model=LLM_MODEL, message=user_prompt, temperature=temperature)
            if system_prompt:
                kwargs["preamble"] = system_prompt

            response = client.chat(**kwargs)
            text = response.text or ""

            if not text:
                raise RuntimeError("Received an empty response from Cohere.")

            elapsed_ms = (time.time() - t0) * 1000
            span.set_attribute("llm.response_chars", len(text))
            span.set_attribute("llm.duration_ms", round(elapsed_ms))
            _get_llm_duration_hist().record(elapsed_ms, {"llm.model": LLM_MODEL})
            return text

        except Exception as exc:
            msg = str(exc)
            span.record_exception(exc)
            if "429" in msg or "rate" in msg.lower() or "too many" in msg.lower():
                wait = 15 * (2 ** _attempt)
                span.set_attribute("llm.error", "rate_limited")
                print(f"  Cohere rate limited — waiting {wait}s...")
                time.sleep(wait)
                return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)
            raise RuntimeError(f"Cohere LLM call failed: {exc}") from exc


async def astream_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
):
    """Async generator — yields text chunks as they stream from Cohere."""
    with _tracer.start_as_current_span("llm.stream") as span:
        span.set_attribute("llm.model", LLM_MODEL)
        span.set_attribute("llm.temperature", temperature)

        client = _get_async_client()
        kwargs: dict = dict(model=LLM_MODEL, message=user_prompt, temperature=temperature)
        if system_prompt:
            kwargs["preamble"] = system_prompt

        total_chars = 0
        async for event in client.chat_stream(**kwargs):
            if event.event_type == "text-generation":
                text = event.text
                if text:
                    total_chars += len(text)
                    yield text
        span.set_attribute("llm.response_chars", total_chars)
