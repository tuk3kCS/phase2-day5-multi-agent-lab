"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        llm = LLMClient()

        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."

        system_prompt = (
            f"You are an expert technical Writer Agent. Your job is to draft a comprehensive final answer "
            f"addressing the query. The target audience is: '{audience}'.\n"
            "Use the provided research notes and analytical notes to construct a detailed, well-structured output.\n"
            "Ensure you integrate all relevant facts and address potential gaps or logical connections identified.\n"
            "Format the final answer in clean, readable markdown. Include citations/references to the sources collected."
        )
        user_prompt = (
            f"Query: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}"
        )

        try:
            res = llm.complete(system_prompt, user_prompt)
            state.final_answer = res.content
            state.add_trace_event("writer_llm_call", {
                "cost_usd": res.cost_usd,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens
            })
            state.add_trace_event("writer_execution", {})
        except Exception as e:

            state.errors.append(f"Writer execution failed: {e}")

        return state

