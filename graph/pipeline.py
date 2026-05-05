import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict

from langgraph.graph import StateGraph, END

from rag.retriever import retrieve
from rag.generator import generate_answer
from agents.fact_checker import fact_check
from agents.consistency_guard import consistency_guard
from agents.confidence_scorer import score_confidence


class PipelineState(TypedDict):
    query: str
    context: str
    sources: list
    answer: str
    fact_check_result: dict
    consistency_result: dict
    confidence_result: dict
    final_output: dict


def retrieve_node(state: PipelineState) -> dict:
    result = retrieve(state["query"])
    return {
        "context": result["context"],
        "sources": result["sources"],
    }


def generate_node(state: PipelineState) -> dict:
    answer = generate_answer(state["context"], state["query"])
    return {"answer": answer}


def fact_check_node(state: PipelineState) -> dict:
    result = fact_check(state["answer"], state["context"])
    return {"fact_check_result": result}


def consistency_guard_node(state: PipelineState) -> dict:
    result = consistency_guard(
        state["query"],
        state["answer"],
        state["fact_check_result"],
        state["context"],
    )
    return {"consistency_result": result}


def confidence_node(state: PipelineState) -> dict:
    result = score_confidence(
        state["query"],
        state["answer"],
        state["context"],
        state["fact_check_result"],
        state["consistency_result"],
    )
    return {"confidence_result": result}


def compile_node(state: PipelineState) -> dict:
    return {
        "final_output": {
            "query": state["query"],
            "answer": state["answer"],
            "sources": state["sources"],
            "fact_check": state["fact_check_result"],
            "consistency": state["consistency_result"],
            "confidence": state["confidence_result"],
        }
    }


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("consistency_guard", consistency_guard_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("compile", compile_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "fact_check")
    graph.add_edge("fact_check", "consistency_guard")
    graph.add_edge("consistency_guard", "confidence")
    graph.add_edge("confidence", "compile")
    graph.add_edge("compile", END)

    return graph.compile()


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_pipeline(query: str) -> dict:
    initial_state: PipelineState = {
        "query": query,
        "context": "",
        "sources": [],
        "answer": "",
        "fact_check_result": {},
        "consistency_result": {},
        "confidence_result": {},
        "final_output": {},
    }
    result = get_pipeline().invoke(initial_state)
    return result["final_output"]
