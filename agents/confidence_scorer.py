import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from opentelemetry import trace as _otel_trace

from rag.llm_client import call_llm

_tracer = _otel_trace.get_tracer(__name__)

_PROMPT = """You are a confidence scoring agent for a retrieval-grounded Stripe documentation assistant.

You receive:
1. The user's original question
2. The retrieved Stripe documentation context
3. The AI-generated answer
4. The layer-1 fact-check results
5. The layer-2 hallucination audit

Your job: estimate how confident the system should be in this answer.

Scoring guide:
- 90-100 → Very high confidence: answer is tightly grounded and all claims are supported
- 70-89  → High confidence: mostly grounded, only minor ambiguity
- 40-69  → Medium confidence: partial support, limited context, or notable uncertainty
- 0-39   → Low confidence: unsupported claims, contradictions, or weak retrieval grounding

Consider:
- How directly the context answers the user's question
- Whether the answer stays within the retrieved evidence
- The fact-check verdicts and hallucination audit
- Whether the answer appropriately states uncertainty when context is thin

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "confidence_score": <integer 0-100>,
  "confidence_label": "VERY_HIGH|HIGH|MEDIUM|LOW",
  "reasoning_summary": "<two-sentence explanation of why this confidence score was assigned>"
}}

Original question: {query}

Retrieved Stripe docs context (truncated):
{context}

AI answer:
{answer}

Layer-1 fact-check results:
{fact_check_result}

Layer-2 hallucination audit:
{consistency_result}

JSON output:"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {
                "confidence_score": 35,
                "confidence_label": "LOW",
                "reasoning_summary": f"Confidence scoring parsing failed. Raw: {raw[:300]}",
            }
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {
                "confidence_score": 35,
                "confidence_label": "LOW",
                "reasoning_summary": f"Confidence scoring parsing failed. Raw: {raw[:300]}",
            }

    score = int(max(0, min(100, data.get("confidence_score", 35))))
    label = str(data.get("confidence_label", "LOW")).upper()
    if label not in {"VERY_HIGH", "HIGH", "MEDIUM", "LOW"}:
        label = "LOW"

    return {
        "confidence_score": score,
        "confidence_label": label,
        "reasoning_summary": str(
            data.get(
                "reasoning_summary",
                "Confidence score generated without an explanatory summary.",
            )
        ),
    }


def score_confidence(
    query: str,
    answer: str,
    context: str,
    fact_check_result: dict,
    consistency_result: dict,
) -> dict:
    with _tracer.start_as_current_span("confidence_scorer") as span:
        span.set_attribute("query.length", len(query))
        prompt = _PROMPT.format(
            query=query,
            context=context[:4000],
            answer=answer,
            fact_check_result=json.dumps(fact_check_result, indent=2),
            consistency_result=json.dumps(consistency_result, indent=2),
        )
        raw = call_llm("", prompt, temperature=0)
        result = _parse_json(raw)
        span.set_attribute("confidence.score", result.get("confidence_score", 0))
        span.set_attribute("confidence.label", result.get("confidence_label", ""))
        return result
