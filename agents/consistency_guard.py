import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from opentelemetry import trace as _otel_trace

from rag.llm_client import call_llm

_tracer = _otel_trace.get_tracer(__name__)

_PROMPT = """You are a hallucination detection agent performing a second-layer audit.

You receive:
1. The user's original question
2. The AI-generated answer
3. Layer-1 fact-check results (claim verdicts)
4. The original Stripe docs context

Your job: assign an overall hallucination risk score and label.

Scoring guide:
- 0–20   → TRUSTED     (all claims supported, answer stays in-scope)
- 21–50  → SUSPICIOUS  (some unsupported claims or minor overreach)
- 51–100 → HALLUCINATED (contradictions, fabrications, or significant scope violations)

Consider:
- Ratio of SUPPORTED vs UNSUPPORTED/CONTRADICTED claims
- Whether the answer invents numbers, endpoints, or features not in context
- Internal consistency of the answer
- Whether the answer hedges appropriately when context is thin

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "hallucination_score": <integer 0-100>,
  "label": "TRUSTED|SUSPICIOUS|HALLUCINATED",
  "final_summary": "<two or three sentence comprehensive reliability assessment>",
  "additional_flags": ["<specific concern>", ...]
}}

Keep additional_flags empty [] when there are no concerns.

Original question: {query}

AI answer: {answer}

Layer-1 fact-check results:
{fact_check_result}

Stripe docs context (truncated):
{context}

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
        "hallucination_score": 50,
        "label": "SUSPICIOUS",
        "final_summary": f"Consistency guard parsing failed. Raw: {raw[:300]}",
        "additional_flags": ["parse_error"],
    }


def consistency_guard(
    query: str,
    answer: str,
    fact_check_result: dict,
    context: str,
) -> dict:
    with _tracer.start_as_current_span("consistency_guard") as span:
        span.set_attribute("query.length", len(query))
        prompt = _PROMPT.format(
            query=query,
            answer=answer,
            fact_check_result=json.dumps(fact_check_result, indent=2),
            context=context[:4000],
        )
        raw = call_llm("", prompt, temperature=0)
        result = _parse_json(raw)
        span.set_attribute("consistency.label", result.get("label", ""))
        span.set_attribute("consistency.hallucination_score", result.get("hallucination_score", 0))
        return result
