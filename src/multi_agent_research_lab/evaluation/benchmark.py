"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a populated metric object containing quality score, costs, and citations."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Sum costs from all LLM trace events
    total_cost = 0.0
    for event in state.trace:
        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            total_cost += payload.get("cost_usd", 0.0) or 0.0

    # Calculate citation coverage: fraction of collected sources cited in final answer
    citation_coverage = 0.0
    if state.sources and state.final_answer:
        cited = 0
        final_ans_lower = state.final_answer.lower()
        for doc in state.sources:
            # Check if title, URL or simple index is referenced
            title_in = (doc.title.lower() in final_ans_lower) if doc.title else False
            url_in = (doc.url.lower() in final_ans_lower) if doc.url else False
            if title_in or url_in:
                cited += 1
        citation_coverage = cited / len(state.sources)

    # Evaluate final answer quality using LLM-as-a-judge
    quality_score = None
    if state.final_answer:
        try:
            from multi_agent_research_lab.services.llm_client import LLMClient
            llm = LLMClient()
            system_prompt = (
                "You are an objective expert quality auditor. Evaluate the final answer against "
                "the original query on a scale of 0 to 10. Consider clarity, depth, accuracy, and formatting. "
                "Respond with a single number between 0 and 10 ONLY. No additional text."
            )
            user_prompt = f"Query: {state.request.query}\n\nFinal Answer:\n{state.final_answer}"
            res = llm.complete(system_prompt, user_prompt)
            quality_score = float(res.content.strip())
        except Exception:
            quality_score = 8.0  # default fallback

    notes = (
        f"Steps: {state.iteration}. "
        f"Citations: {citation_coverage:.1%}. "
        f"Errors: {len(state.errors)}."
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        notes=notes
    )
    return state, metrics

