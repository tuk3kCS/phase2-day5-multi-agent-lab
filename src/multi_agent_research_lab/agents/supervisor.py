"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        from multi_agent_research_lab.core.config import get_settings
        from multi_agent_research_lab.services.llm_client import LLMClient

        settings = get_settings()

        # Enforce max iterations to prevent infinite loops
        if state.iteration >= settings.max_iterations:
            state.record_route("done")
            state.add_trace_event("supervisor_routing", {"decision": "done", "reason": "max_iterations_reached"})
            return state

        has_research = "Yes" if state.research_notes else "No"
        has_analysis = "Yes" if state.analysis_notes else "No"
        has_final_answer = "Yes" if state.final_answer else "No"

        system_prompt = (
            "You are the Supervisor of a multi-agent research team. Your goal is to coordinate them "
            "to produce a high-quality final answer for the query.\n"
            "Available agents:\n"
            "- 'researcher': Gather sources and write initial research notes. Always call this first if notes are empty.\n"
            "- 'analyst': Take research notes and extract key claims, identify gaps/viewpoints, and write analysis notes.\n"
            "- 'writer': Take analysis/research notes and write a final comprehensive answer.\n"
            "- 'critic': Evaluate the final answer for accuracy, quality, and gaps. Only call this after writer has produced an answer.\n"
            "- 'done': Final answer is complete, verified, and satisfactory.\n\n"
            "Routing Rules:\n"
            "1. If no research has been done yet, route to 'researcher'.\n"
            "2. If research is done but no analysis is done, route to 'analyst'.\n"
            "3. If analysis is done but no final answer is written, route to 'writer'.\n"
            "4. If final answer is written, route to 'critic'.\n"
            "5. If critic has approved the answer (or if you decide no more edits are needed), route to 'done'.\n"
            "6. If the workflow is stuck or max iterations is near, route to 'done' to output the best available answer."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Iteration: {state.iteration}\n"
            f"Route history: {state.route_history}\n"
            f"Sources gathered: {len(state.sources)}\n"
            f"Research Notes present: {has_research}\n"
            f"Analysis Notes present: {has_analysis}\n"
            f"Final Answer present: {has_final_answer}\n\n"
            "What is the next agent to route to? (Respond with exactly one of: 'researcher', 'analyst', 'writer', 'critic', 'done')"
        )

        try:
            llm = LLMClient()
            response = llm.complete(system_prompt, user_prompt)
            next_step = response.content.strip().lower()
            state.add_trace_event("supervisor_llm_call", {
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens
            })
        except Exception:
            next_step = "done"


        # Match step
        valid_routes = {"researcher", "analyst", "writer", "critic", "done"}
        matched_route = None
        for r in valid_routes:
            if r in next_step:
                matched_route = r
                break

        if not matched_route:
            # Deterministic fallback
            if not state.research_notes:
                matched_route = "researcher"
            elif not state.analysis_notes:
                matched_route = "analyst"
            elif not state.final_answer:
                matched_route = "writer"
            else:
                matched_route = "done"

        state.record_route(matched_route)
        state.add_trace_event("supervisor_routing", {"decision": matched_route, "reasoning": next_step})
        return state

