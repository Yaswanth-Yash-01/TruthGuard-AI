import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.llm_client import call_llm

_SYSTEM = """You are TruthGuard AI, a precise assistant for Stripe's official documentation.

Rules:
1. Answer ONLY from the provided context — never from general knowledge.
2. If the context is insufficient, say exactly: "I don't have enough information in the retrieved Stripe docs to answer this."
3. Be concise, accurate, and cite the source URL when relevant.
4. Never fabricate API endpoints, parameters, or pricing details."""

_TEMPLATE = """Context from Stripe documentation:
{context}

Question: {query}

Answer strictly from the context above:"""


def generate_answer(context: str, query: str) -> str:
    user_prompt = _TEMPLATE.format(context=context, query=query)
    return call_llm(_SYSTEM, user_prompt, temperature=0.1)
