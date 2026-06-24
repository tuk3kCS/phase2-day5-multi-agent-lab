import pytest
from unittest.mock import MagicMock
from multi_agent_research_lab.agents import SupervisorAgent, ResearcherAgent, AnalystAgent, WriterAgent, CriticAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


def test_supervisor_agent_routing(monkeypatch) -> None:
    # Mock LLMClient.complete to return 'researcher'
    mock_complete = MagicMock(return_value=LLMResponse(content="researcher"))
    monkeypatch.setattr(LLMClient, "complete", mock_complete)
    
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()
    updated_state = supervisor.run(state)
    
    assert updated_state.route_history == ["researcher"]
    assert updated_state.iteration == 1


def test_researcher_agent(monkeypatch) -> None:
    # Mock SearchClient.search and LLMClient.complete
    from multi_agent_research_lab.services.search_client import SearchClient
    from multi_agent_research_lab.core.schemas import SourceDocument
    
    mock_search = MagicMock(return_value=[
        SourceDocument(title="Mock Title", url="http://mock.com", snippet="Mock snippet content")
    ])
    monkeypatch.setattr(SearchClient, "search", mock_search)
    
    mock_complete = MagicMock(return_value=LLMResponse(content="Mocked research notes"))
    monkeypatch.setattr(LLMClient, "complete", mock_complete)
    
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    researcher = ResearcherAgent()
    updated_state = researcher.run(state)
    
    assert len(updated_state.sources) == 1
    assert updated_state.sources[0].title == "Mock Title"
    assert updated_state.research_notes == "Mocked research notes"

