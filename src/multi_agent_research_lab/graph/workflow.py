"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(ResearchState)

        # Define node wrappers
        def supervisor_node(state: ResearchState) -> ResearchState:
            from multi_agent_research_lab.agents.supervisor import SupervisorAgent
            return SupervisorAgent().run(state)

        def researcher_node(state: ResearchState) -> ResearchState:
            from multi_agent_research_lab.agents.researcher import ResearcherAgent
            return ResearcherAgent().run(state)

        def analyst_node(state: ResearchState) -> ResearchState:
            from multi_agent_research_lab.agents.analyst import AnalystAgent
            return AnalystAgent().run(state)

        def writer_node(state: ResearchState) -> ResearchState:
            from multi_agent_research_lab.agents.writer import WriterAgent
            return WriterAgent().run(state)

        def critic_node(state: ResearchState) -> ResearchState:
            from multi_agent_research_lab.agents.critic import CriticAgent
            return CriticAgent().run(state)

        # Add nodes to graph
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("writer", writer_node)
        workflow.add_node("critic", critic_node)

        # Define conditional routing from supervisor
        def route_decision(state: ResearchState) -> str:
            if not state.route_history:
                return "supervisor"
            return state.route_history[-1]

        workflow.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END
            }
        )

        # Connect workers back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        workflow.set_entry_point("supervisor")
        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        app = self.build()
        
        # Invoke compiled LangGraph workflow
        result = app.invoke(state)
        
        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        else:
            return ResearchState(**dict(result))

