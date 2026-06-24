"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        llm = LLMClient()

        query = state.request.query
        research_notes = state.research_notes or "No research notes available."

        system_prompt = (
            "You are an expert Analyst Agent. Your job is to take raw research notes and analyze them critically.\n"
            "1. Extract the main claims and factual arguments.\n"
            "2. Identify any conflicting viewpoints or perspectives.\n"
            "3. Flag potential gaps in the research or areas where evidence is weak.\n"
            "Present your analysis in structured markdown notes."
        )
        user_prompt = f"Query: {query}\n\nResearch Notes:\n{research_notes}"

        try:
            res = llm.complete(system_prompt, user_prompt)
            state.analysis_notes = res.content
            state.add_trace_event("analyst_llm_call", {
                "cost_usd": res.cost_usd,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens
            })
            state.add_trace_event("analyst_execution", {})
        except Exception as e:

            state.errors.append(f"Analyst execution failed: {e}")

        return state

