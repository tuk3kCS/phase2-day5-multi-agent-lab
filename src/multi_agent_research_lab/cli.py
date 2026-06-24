"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline(query: str) -> ResearchState:
    from multi_agent_research_lab.services.search_client import SearchClient
    from multi_agent_research_lab.services.llm_client import LLMClient
    
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    search_client = SearchClient()
    llm = LLMClient()
    
    # Search
    docs = search_client.search(query, max_results=request.max_sources)
    state.sources = docs
    
    # LLM completion
    sources_context = "\n\n".join([
        f"Source: {doc.title}\nURL: {doc.url or 'No URL'}\nSnippet: {doc.snippet}"
        for doc in docs
    ])
    system_prompt = (
        "You are a single-agent research assistant. Answer the query comprehensively using "
        "only the provided search results. Provide inline citations citing the source title or URL."
    )
    user_prompt = f"Query: {query}\n\nSearch Results:\n{sources_context}"
    
    res = llm.complete(system_prompt, user_prompt)
    state.final_answer = res.content
    state.add_trace_event("baseline_llm_call", {
        "cost_usd": res.cost_usd,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens
    })
    state.iteration = 1
    state.route_history = ["baseline"]
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""
    _init()
    console.print(f"[bold blue]Running baseline for query:[/bold blue] '{query}'")
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer or "No response generated.", title="Single-Agent Baseline Answer"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print(f"[bold blue]Running multi-agent workflow for query:[/bold blue] '{query}'")
    try:
        result = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    
    console.print(Panel.fit(result.final_answer or "No response generated.", title="Multi-Agent Final Answer"))
    console.print("[bold green]Trace History:[/bold green]")
    for step in result.trace:
        console.print(f"- {step.get('name')}: {step.get('payload')}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run benchmark comparing single-agent baseline and multi-agent system."""
    _init()
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    
    console.print(f"[bold blue]Running Single-Agent Baseline on:[/bold blue] '{query}'")
    baseline_state, baseline_metrics = run_benchmark("Single-Agent Baseline", query, _run_baseline)
    console.print(f"[green][OK] Completed Baseline in {baseline_metrics.latency_seconds:.2f}s[/green]\n")
    
    console.print(f"[bold blue]Running Multi-Agent System on:[/bold blue] '{query}'")
    multi_state, multi_metrics = run_benchmark("Multi-Agent (Supervisor + Workers)", query, _run_multi_agent)
    console.print(f"[green][OK] Completed Multi-Agent in {multi_metrics.latency_seconds:.2f}s[/green]\n")
    
    report_markdown = render_markdown_report([baseline_metrics, multi_metrics])
    
    # Save to reports/benchmark_report.md
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report_markdown)
        
    from rich.table import Table
    table = Table(title="Benchmark Metrics Summary")
    table.add_column("Run", style="cyan")
    table.add_column("Latency", justify="right", style="magenta")
    table.add_column("Cost", justify="right", style="green")
    table.add_column("Quality", justify="right", style="yellow")
    table.add_column("Notes", style="white")

    for item in [baseline_metrics, multi_metrics]:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}"
        safe_notes = item.notes.encode("ascii", errors="replace").decode("ascii")
        table.add_row(item.run_name, f"{item.latency_seconds:.2f}s", cost, quality, safe_notes)
    
    console.print(table)
    console.print("[bold green]Detailed benchmark report written to reports/benchmark_report.md[/bold green]")


if __name__ == "__main__":
    app()

