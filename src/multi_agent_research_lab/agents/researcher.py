"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        from multi_agent_research_lab.services.search_client import SearchClient
        from multi_agent_research_lab.services.llm_client import LLMClient

        search_client = SearchClient()
        llm = LLMClient()

        query = state.request.query
        max_sources = state.request.max_sources

        # Run search
        try:
            results = search_client.search(query, max_results=max_sources)
            for doc in results:
                # Add source avoiding duplicates
                if not any(s.url == doc.url for s in state.sources if s.url):
                    state.sources.append(doc)
        except Exception as e:
            state.errors.append(f"Researcher search failed: {e}")

        # Construct context from sources
        if state.sources:
            context = "\n\n".join([
                f"Source: {doc.title}\nURL: {doc.url or 'No URL'}\nSnippet: {doc.snippet}"
                for doc in state.sources
            ])
        else:
            context = "No source documents retrieved."

        system_prompt = (
            "You are an expert Researcher Agent. Your task is to extract relevant facts, definitions, "
            "and viewpoints from the search results to answer the query.\n"
            "Format your output as organized, professional research notes. Keep inline citations to the sources (by title/URL)."
        )
        user_prompt = f"Query: {query}\n\nSearch Results:\n{context}"

        try:
            res = llm.complete(system_prompt, user_prompt)
            state.research_notes = res.content
            state.add_trace_event("researcher_llm_call", {
                "cost_usd": res.cost_usd,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens
            })
            state.add_trace_event("researcher_execution", {"num_sources": len(state.sources)})
        except Exception as e:

            state.errors.append(f"Researcher LLM summary failed: {e}")

        return state

