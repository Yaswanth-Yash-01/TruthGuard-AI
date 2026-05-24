import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import anthropic
from opentelemetry import trace as _otel_trace, metrics as _otel_metrics

from config.settings import ANTHROPIC_API_KEY, LLM_MODEL

_tracer = _otel_trace.get_tracer(__name__)
_llm_duration_hist = None

_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None


def _get_llm_duration_hist():
    global _llm_duration_hist
    if _llm_duration_hist is None:
        _llm_duration_hist = _otel_metrics.get_meter("truthguard").create_histogram(
            "truthguard.llm.duration_ms", unit="ms", description="LLM call duration"
        )
    return _llm_duration_hist


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
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
        raise RuntimeError("Claude is currently overloaded. Try again in a moment.")

    t0 = time.time()
    with _tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", LLM_MODEL)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.attempt", _attempt)
        span.set_attribute("llm.prompt_chars", len(system_prompt) + len(user_prompt))
        try:
            client = _get_client()

            kwargs: dict = {
                "model": LLM_MODEL,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if system_prompt:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            response = client.messages.create(**kwargs)
            text = response.content[0].text if response.content else ""

            if not text:
                raise RuntimeError("Received an empty response from Claude.")

            elapsed_ms = (time.time() - t0) * 1000
            span.set_attribute("llm.response_chars", len(text))
            span.set_attribute("llm.duration_ms", round(elapsed_ms))
            _get_llm_duration_hist().record(elapsed_ms, {"llm.model": LLM_MODEL})
            return text

        except anthropic.RateLimitError:
            wait = 15 * (2 ** _attempt)
            span.set_attribute("llm.error", "rate_limited_429")
            print(f"  Claude rate limited — waiting {wait}s...")
            time.sleep(wait)
            return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)

        except anthropic.APIStatusError as exc:
            if exc.status_code == 529:
                wait = 5 * (2 ** _attempt)
                span.set_attribute("llm.error", "overloaded_529")
                print(f"  Claude overloaded — waiting {wait}s...")
                time.sleep(wait)
                return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)
            span.record_exception(exc)
            raise RuntimeError(f"Claude API error {exc.status_code}: {exc.message}") from exc

        except Exception as exc:
            span.record_exception(exc)
            raise RuntimeError(f"LLM call failed: {exc}") from exc


async def astream_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
):
    """Async generator — yields text chunks as they stream from Claude."""
    with _tracer.start_as_current_span("llm.stream") as span:
        span.set_attribute("llm.model", LLM_MODEL)
        span.set_attribute("llm.temperature", temperature)

        client = _get_async_client()

        kwargs: dict = {
            "model": LLM_MODEL,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        total_chars = 0
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    total_chars += len(text)
                    yield text
        span.set_attribute("llm.response_chars", total_chars)
