import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
from config.settings import GEMINI_API_KEY

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


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    _attempt: int = 0,
) -> str:
    """Call the active LLM; automatically fall back on 503 overload, retry on 429."""
    if _attempt >= len(_gen_capable) + 1:
        raise RuntimeError("All LLM models are currently overloaded. Try again in a moment.")

    model = get_llm_model()
    body: dict = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    resp = requests.post(
        _GEN_URL.format(model=model),
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=60,
    )

    if resp.status_code == 503:
        print(f"  {model} overloaded (503) — trying next model...")
        _overloaded.add(model)
        global _active_llm
        _active_llm = None
        time.sleep(1)
        return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)

    if resp.status_code == 429:
        wait = 15 * (2 ** _attempt)   # 15s → 30s → 60s  (resets within 1 min window)
        print(f"  Rate limited (429) — waiting {wait}s...")
        time.sleep(wait)
        return call_llm(system_prompt, user_prompt, temperature, _attempt + 1)

    if not resp.ok:
        raise RuntimeError(
            f"Error calling model '{model}' ({resp.status_code}): {resp.text[:400]}"
        )

    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected LLM response structure: {resp.text[:300]}")
