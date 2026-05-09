import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from opentelemetry import trace as _otel_trace

from rag.llm_client import call_llm

_tracer = _otel_trace.get_tracer(__name__)

_PROMPT = """You are a rigorous fact-checking agent for Stripe documentation.

Given an AI-generated answer and the Stripe documentation context it was based on, identify every distinct factual claim in the answer and verify each one against the context.

Verdict options:
- SUPPORTED   → directly and clearly supported by the context
- UNSUPPORTED → not mentioned in the context (may or may not be true in reality)
- CONTRADICTED → explicitly contradicts what the context says

Rules:
- Extract every atomic factual claim (ignore filler/opinion sentences).
- Be strict: if the context only partially covers a claim, mark UNSUPPORTED.
- overall_verdict is PASS only when ALL claims are SUPPORTED.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "claims": [
    {{
      "claim": "<exact claim text from the answer>",
      "verdict": "SUPPORTED|UNSUPPORTED|CONTRADICTED",
      "reason": "<one sentence citing specific context evidence>"
    }}
  ],
  "overall_verdict": "PASS|FAIL",
  "summary": "<two-sentence overall assessment>"
}}

Context from Stripe docs:
{context}

Answer to verify:
{answer}

JSON output:"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "claims": [],
        "overall_verdict": "FAIL",
        "summary": f"Fact-check parsing failed. Raw output: {raw[:300]}",
    }


def fact_check(answer: str, context: str) -> dict:
    with _tracer.start_as_current_span("fact_check") as span:
        span.set_attribute("answer.length", len(answer))
        prompt = _PROMPT.format(context=context[:6000], answer=answer)
        raw = call_llm("", prompt, temperature=0)
        result = _parse_json(raw)
        span.set_attribute("fact_check.verdict", result.get("overall_verdict", ""))
        span.set_attribute("fact_check.claims_count", len(result.get("claims", [])))
        return result
