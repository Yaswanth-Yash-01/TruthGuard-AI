import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from opentelemetry import trace as _otel_trace, metrics as _otel_metrics

from config.settings import GEMINI_API_KEY

_tracer = _otel_trace.get_tracer(__name__)
_llm_duration_hist = None


def _get_llm_duration_hist():
    global _llm_duration_hist
    if _llm_duration_hist is None:
        _llm_duration_hist = _otel_metrics.get_meter("truthguard").create_histogram(
            "truthguard.llm.duration_ms", unit="ms", description="LLM call duration"
        )
    return _llm_duration_hist

_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_PREFERRED_LLM = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-flash-latest",
    "models/gemini-2.5-pro",
]

_active_llm: str | None = None
_overloaded: set[str] = set()   # models returning 503 this session
_gen_capable: list[str] = []    # discovered from ListModels, cached
_llm_clients: dict[str, ChatGoogleGenerativeAI] = {}


def _normalize_model_name(model: str) -> str:
    return model.removeprefix("models/")


def _extract_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _get_llm_client(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    normalized = _normalize_model_name(model)
    cache_key = f"{normalized}|{temperature}"
    if cache_key not in _llm_clients:
        _llm_clients[cache_key] = ChatGoogleGenerativeAI(
            model=normalized,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
        )
    return _llm_clients[cache_key]


def _list_gen_models() -> list[str]:
    """Call ListModels once and return generateContent models in preference order."""
    try:
        resp = requests.get(
            _LIST_URL,
            params={"key": GEMINI_API_KEY},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach generativelanguage.googleapis.com.\n"
            "Run without --reload:  uvicorn api.main:app --port 8000"
        )

    if not resp.ok:
        raise RuntimeError(
            f"ListModels failed ({resp.status_code}):\n{resp.text[:400]}"
        )

    all_models = resp.json().get("models", [])
    capable = [
        m["name"] for m in all_models
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]

    if not capable:
        raise RuntimeError(
            "No generateContent models found.\n"
            f"All models: {[m['name'] for m in all_models]}"
        )

    # Return in preference order: preferred first, then anything else.
    ordered = [p for p in _PREFERRED_LLM if p in capable]
    ordered += [m for m in capable if m not in ordered]
    return ordered


def get_llm_model() -> str:
    """Return the first non-overloaded available model; populate cache on first call."""
    global _active_llm, _gen_capable

    if not _gen_capable:
        _gen_capable = _list_gen_models()

    for model in _gen_capable:
        if model not in _overloaded:
            if _active_llm != model:
                print(f"  LLM model: {model}")
                _active_llm = model
            return model

    # All known models overloaded — clear flags and try again from the top.
    _overloaded.clear()
    _active_llm = _gen_capable[0]
    print(f"  LLM model (retry): {_active_llm}")
    return _active_llm


async def astream_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
):
    """Async generator — yields text chunks as they arrive from the model."""
    model = get_llm_model()
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    client = _get_llm_client(model, temperature)
    with _tracer.start_as_current_span("llm.stream") as span:
        span.set_attribute("llm.model", _normalize_model_name(model))
        span.set_attribute("llm.temperature", temperature)
        total_chars = 0
        async for chunk in client.astream(messages):
            text = _extract_text(chunk)
            if text:
                total_chars += len(text)
                yield text
        span.set_attribute("llm.response_chars", total_chars)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    _attempt: int = 0,
) -> str:
    """Call the active LLM through LangChain; fall back on 503 and retry on 429."""
    if _attempt >= len(_gen_capable) + 1:
        raise RuntimeError("All LLM models are currently overloaded. Try again in a moment.")

    model = get_llm_model()
    t0 = time.time()
    with _tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", _normalize_model_name(model))
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.attempt", _attempt)
        span.set_attribute("llm.prompt_chars", len(system_prompt) + len(user_prompt))
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=user_prompt))

            client = _get_llm_client(model, temperature)
            response = client.invoke(messages)
            text = _extract_text(response)
            if not text:
                raise RuntimeError("Received an empty response from the LLM.")

            elapsed_ms = (time.time() - t0) * 1000
            span.set_attribute("llm.response_chars", len(text))
            span.set_attribute("llm.duration_ms", round(elapsed_ms))
            _get_llm_duration_hist().record(elapsed_ms, {"llm.model": _normalize_model_name(model)})
            return text
        except Exception as exc:
            message = str(exc)
            span.record_exception(exc)
            global _active_llm

            if "503" in message or "overloaded" in message.lower():
                span.set_attribute("llm.error", "overloaded_503")
                print(f"  {model} overloaded (503) — trying next model...")
                _overloaded.add(model)
                _active_llm = None
                time.sleep(1)
                return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)

            if "429" in message or "resource exhausted" in message.lower():
                wait = 15 * (2 ** _attempt)
                span.set_attribute("llm.error", "rate_limited_429")
                print(f"  Rate limited (429) — waiting {wait}s...")
                time.sleep(wait)
                return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)

            raise RuntimeError(f"Error calling model '{model}': {message}") from exc
