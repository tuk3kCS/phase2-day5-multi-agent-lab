"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        from multi_agent_research_lab.services.llm_client import LLMClient
        llm = LLMClient()

        query = state.request.query
        final_answer = state.final_answer or "No final answer drafted."
        research_notes = state.research_notes or "No research notes available."

        system_prompt = (
            "You are an expert Critic Agent. Your job is to verify the final answer against the query "
            "and the raw research notes.\n"
            "Evaluate:\n"
            "1. Alignment: Does it fully answer the original query?\n"
            "2. Factuality: Are all key statements supported by the research notes? Flag any unsupported claims.\n"
            "3. Citation Coverage: Are sources correctly referenced/cited?\n\n"
            "Provide detailed feedback. In the first line, state exactly 'STATUS: APPROVED' if the answer passes, "
            "or 'STATUS: REJECTED' if there are gaps or issues to resolve."
        )
        user_prompt = (
            f"Query: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Draft Answer:\n{final_answer}"
        )

        try:
            res = llm.complete(system_prompt, user_prompt)
            feedback = res.content
            approved = "status: approved" in feedback.lower()
            
            state.add_trace_event("critic_llm_call", {
                "cost_usd": res.cost_usd,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens
            })
            state.add_trace_event("critic_review", {
                "approved": approved,
                "feedback": feedback
            })


            # If rejected, append critic feedback to analysis notes so the next round of writer execution can use it.
            if not approved:
                state.analysis_notes = (
                    (state.analysis_notes or "") + 
                    f"\n\nCRITIC REJECTED FEEDBACK (ITERATION {state.iteration}):\n{feedback}"
                )
        except Exception as e:
            state.errors.append(f"Critic execution failed: {e}")

        return state

