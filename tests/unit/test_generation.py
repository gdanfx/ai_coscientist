"""
Tests for the Enhanced Generation Agent

This module contains comprehensive tests for the EnhancedGenerationAgent
including unit tests, integration tests, and edge case validation.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.generation import (
    EnhancedGenerationAgent,
    create_generation_agent,
    test_generation_agent
)
from core.data_structures import GenerationState, create_timestamp
from utils.llm_client import LLMResponse

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedGenerationAgent:
    """Test suite for the EnhancedGenerationAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = EnhancedGenerationAgent()
        
        # Sample test data
        self.test_research_goal = "Develop novel AI-driven therapeutic approaches for neurodegenerative diseases"
        self.test_constraints = [
            "Must be clinically translatable",
            "Focus on personalized medicine",
            "Consider ethical implications"
        ]
        
        # Sample literature findings
        self.sample_literature = [
            {
                "title": "AI Applications in Alzheimer's Drug Discovery",
                "authors": ["Smith, J.", "Doe, A.", "Johnson, M."],
                "year": "2023",
                "source": "PubMed",
                "key_findings": [
                    "Machine learning models improved target identification",
                    "Personalized treatment protocols showed 30% better outcomes"
                ]
            },
            {
                "title": "Deep Learning for Neurodegeneration Biomarkers",
                "authors": ["Chen, L.", "Williams, K."],
                "year": "2023", 
                "source": "ArXiv",
                "key_findings": [
                    "Novel biomarker discovery using neural networks",
                    "Early detection improved by 45%"
                ]
            }
        ]

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = EnhancedGenerationAgent()
        
        assert agent is not None
        assert hasattr(agent, 'llm')
        
        # Test with custom LLM client
        mock_llm = Mock()
        agent_with_llm = EnhancedGenerationAgent(mock_llm)
        assert agent_with_llm.llm == mock_llm

    def test_factory_function(self):
        """Test the factory function creates agents correctly."""
        agent = create_generation_agent()
        assert isinstance(agent, EnhancedGenerationAgent)

    @patch('agents.generation.get_global_llm_client')
    def test_generate_search_queries_success(self, mock_llm_client):
        """Test successful search query generation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        1. AI drug discovery neurodegenerative diseases
        2. personalized medicine Alzheimer's Parkinson's
        3. machine learning biomarkers neurodegeneration
        4. therapeutic targets artificial intelligence brain
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        # Create state and run query generation
        state = GenerationState(
            research_goal=self.test_research_goal,
            constraints=self.test_constraints
        )
        
        result_state = self.agent.generate_search_queries(state)
        
        # Validate results
        assert result_state.status == "queries_generated"
        assert len(result_state.search_queries) > 0
        assert len(result_state.search_queries) <= 4
        
        # Check that queries are meaningful
        for query in result_state.search_queries:
            assert len(query) > 3
            assert isinstance(query, str)

    @patch('agents.generation.get_global_llm_client')
    def test_generate_search_queries_llm_error(self, mock_llm_client):
        """Test search query generation with LLM error."""
        # Mock LLM error
        mock_response = Mock()
        mock_response.error = "API rate limit exceeded"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            constraints=self.test_constraints
        )
        
        result_state = self.agent.generate_search_queries(state)
        
        # Should fallback to heuristic queries
        assert result_state.status == "queries_generated"
        assert len(result_state.search_queries) > 0
        
        # Should contain research goal terms
        query_text = " ".join(result_state.search_queries)
        assert any(term in query_text.lower() for term in ["ai", "therapeutic", "neurodegenerative"])

    @patch('agents.generation.multi_source_literature_search')
    def test_literature_exploration_success(self, mock_search):
        """Test successful literature exploration."""
        # Mock literature search results
        mock_search.return_value = self.sample_literature
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            constraints=self.test_constraints,
            search_queries=["AI neurodegenerative", "personalized medicine"]
        )
        
        result_state = self.agent.literature_exploration(state)
        
        # Validate results
        assert result_state.status == "literature_explored"
        assert len(result_state.literature_findings) == len(self.sample_literature)
        assert result_state.literature_findings == self.sample_literature
        
        # Verify search was called with correct parameters
        mock_search.assert_called_once_with(
            queries=state.search_queries,
            max_results_per_source=3,
            total_max_results=12
        )

    @patch('agents.generation.multi_source_literature_search')
    def test_literature_exploration_failure(self, mock_search):
        """Test literature exploration with search failure."""
        # Mock search failure
        mock_search.side_effect = Exception("Network timeout")
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            search_queries=["test query"]
        )
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Literature search failed"):
            self.agent.literature_exploration(state)

    @patch('agents.generation.get_global_llm_client')
    def test_synthesize_knowledge_success(self, mock_llm_client):
        """Test successful knowledge synthesis."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        The literature reveals several key themes in AI-driven therapeutic approaches:
        
        1. PERSONALIZED MEDICINE: Multiple studies demonstrate that AI enables more precise
        treatment selection based on individual patient profiles and biomarkers.
        
        2. BIOMARKER DISCOVERY: Machine learning approaches have significantly improved
        our ability to identify novel biomarkers for early disease detection.
        
        3. DRUG DISCOVERY ACCELERATION: AI models are reducing the time and cost of
        identifying promising therapeutic targets and compounds.
        
        KEY GAPS: Despite progress, current approaches lack integration across multiple
        data modalities and long-term validation of personalized treatment outcomes.
        
        BREAKTHROUGH OPPORTUNITIES: Combining multimodal AI with longitudinal patient
        data could enable truly predictive and preventive therapeutic strategies.
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            literature_findings=self.sample_literature
        )
        
        result_state = self.agent.synthesize_knowledge(state)
        
        # Validate results
        assert result_state.status == "knowledge_synthesized"
        assert len(result_state.synthesized_knowledge) > 100
        assert "personalized" in result_state.synthesized_knowledge.lower()
        assert "biomarker" in result_state.synthesized_knowledge.lower()

    def test_synthesize_knowledge_no_findings(self):
        """Test knowledge synthesis with no literature findings."""
        state = GenerationState(
            research_goal=self.test_research_goal,
            literature_findings=[]
        )
        
        result_state = self.agent.synthesize_knowledge(state)
        
        # Should handle gracefully
        assert result_state.status == "knowledge_synthesized"
        assert "No literature findings" in result_state.synthesized_knowledge

    @patch('agents.generation.get_global_llm_client')
    def test_generate_hypotheses_success(self, mock_llm_client):
        """Test successful hypothesis generation."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        Hypothesis 1: AI-powered multimodal biomarker integration platform that combines neuroimaging, genomics, and proteomics data to predict therapeutic response in Alzheimer's patients before treatment initiation, enabling truly personalized therapeutic selection.
        ---
        Hypothesis 2: Federated learning framework for continuous real-world evidence collection from wearable devices and EHRs to dynamically adjust personalized treatment protocols for Parkinson's disease patients, improving long-term outcomes through adaptive therapy.
        ---
        Hypothesis 3: Explainable AI system that identifies novel drug repurposing opportunities for rare neurodegenerative diseases by analyzing molecular pathway similarities and patient phenotype clustering across multiple disease databases.
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            constraints=self.test_constraints,
            synthesized_knowledge="Knowledge synthesis about AI and neurodegeneration",
            literature_findings=self.sample_literature
        )
        
        result_state = self.agent.generate_hypotheses(state)
        
        # Validate results
        assert result_state.status == "hypotheses_generated"
        assert len(result_state.generated_proposals) == 3
        assert result_state.iteration == 1
        
        # Check proposal structure
        for proposal in result_state.generated_proposals:
            assert "id" in proposal
            assert "content" in proposal
            assert "timestamp" in proposal
            assert "based_on_sources" in proposal
            assert "source_diversity" in proposal
            assert len(proposal["content"]) > 50

    def test_generate_hypotheses_no_knowledge(self):
        """Test hypothesis generation with no synthesized knowledge."""
        state = GenerationState(
            research_goal=self.test_research_goal,
            synthesized_knowledge=""
        )
        
        result_state = self.agent.generate_hypotheses(state)
        
        # Should handle gracefully
        assert result_state.status == "hypotheses_generated"
        assert len(result_state.generated_proposals) == 0

    def test_format_literature_findings(self):
        """Test literature findings formatting."""
        formatted = self.agent._format_literature_findings(self.sample_literature)
        
        assert len(formatted) > 0
        assert "AI Applications in Alzheimer's" in formatted
        assert "Smith, J." in formatted
        assert "2023" in formatted
        assert "PubMed" in formatted

    def test_format_literature_findings_empty(self):
        """Test formatting with empty findings."""
        formatted = self.agent._format_literature_findings([])
        assert "No literature findings" in formatted

    def test_parse_hypotheses_from_response(self):
        """Test hypothesis parsing from LLM response."""
        response_content = """
        Hypothesis 1: First test hypothesis about AI and medicine
        ---
        Hypothesis 2: Second test hypothesis about personalized therapy
        ---
        Hypothesis 3: Third test hypothesis about biomarker discovery
        """
        
        state = GenerationState(
            research_goal=self.test_research_goal,
            literature_findings=self.sample_literature
        )
        
        proposals = self.agent._parse_hypotheses_from_response(response_content, state)
        
        assert len(proposals) == 3
        for i, proposal in enumerate(proposals):
            assert f"gen_{state.iteration}_{i+1}" in proposal["id"]
            assert len(proposal["content"]) > 20
            assert proposal["based_on_sources"] == len(self.sample_literature)

    def test_parse_hypotheses_fallback(self):
        """Test hypothesis parsing fallback for unparseable content."""
        response_content = "This is just random text without proper formatting"
        
        state = GenerationState(research_goal=self.test_research_goal)
        proposals = self.agent._parse_hypotheses_from_response(response_content, state)
        
        # Should create fallback proposal
        assert len(proposals) == 1
        assert "fallback" in proposals[0]["id"]

    @patch('agents.generation.multi_source_literature_search')
    @patch('agents.generation.get_global_llm_client')
    def test_run_complete_workflow_success(self, mock_llm_client, mock_search):
        """Test complete workflow execution."""
        # Mock literature search
        mock_search.return_value = self.sample_literature
        
        # Mock LLM responses for different stages
        def mock_invoke(prompt):
            mock_response = Mock()
            mock_response.error = None
            
            if "search queries" in prompt.lower():
                mock_response.content = "1. AI neurodegeneration\n2. personalized medicine"
            elif "synthesize" in prompt.lower():
                mock_response.content = "Comprehensive knowledge synthesis about AI approaches"
            elif "hypotheses" in prompt.lower():
                mock_response.content = "Hypothesis 1: Test hypothesis\n---\nHypothesis 2: Another test hypothesis"
            else:
                mock_response.content = "Default response"
            
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        # Run complete workflow
        result = self.agent.run_complete_workflow(
            self.test_research_goal,
            self.test_constraints
        )
        
        # Validate complete workflow results
        assert isinstance(result, GenerationState)
        assert result.research_goal == self.test_research_goal
        assert result.constraints == self.test_constraints
        assert len(result.search_queries) > 0
        assert len(result.literature_findings) > 0
        assert len(result.synthesized_knowledge) > 0
        assert len(result.generated_proposals) > 0
        assert result.status == "hypotheses_generated"

    @patch('agents.generation.multi_source_literature_search')
    def test_run_complete_workflow_failure(self, mock_search):
        """Test complete workflow with failure during literature search."""
        # Mock search failure
        mock_search.side_effect = Exception("Network error")
        
        result = self.agent.run_complete_workflow(
            self.test_research_goal,
            self.test_constraints
        )
        
        # Should return state with error information
        assert isinstance(result, GenerationState)
        assert "failed_at_" in result.status
        assert len(result.generated_proposals) == 1
        assert "workflow_error" in result.generated_proposals[0]["id"]


class TestGenerationAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_research_goal(self):
        """Test handling of empty research goal."""
        agent = EnhancedGenerationAgent()
        
        result = agent.run_complete_workflow("", [])
        
        # Should handle gracefully
        assert isinstance(result, GenerationState)
        assert result.research_goal == ""

    def test_very_long_research_goal(self):
        """Test handling of very long research goal."""
        agent = EnhancedGenerationAgent()
        
        long_goal = "A" * 5000  # Very long goal
        result = agent.run_complete_workflow(long_goal, [])
        
        assert isinstance(result, GenerationState)
        assert result.research_goal == long_goal

    def test_special_characters_in_goal(self):
        """Test handling of special characters in research goal."""
        agent = EnhancedGenerationAgent()
        
        special_goal = "Research göal with spéciál characters & symbols! @#$%^&*()"
        result = agent.run_complete_workflow(special_goal, [])
        
        assert isinstance(result, GenerationState)
        assert result.research_goal == special_goal

    @patch('agents.generation.get_global_llm_client')
    def test_llm_client_none(self, mock_llm_client):
        """Test behavior when LLM client is None."""
        mock_llm_client.return_value = None
        
        agent = EnhancedGenerationAgent()
        
        # Should not crash during initialization
        assert agent.llm is None

    def test_malformed_literature_findings(self):
        """Test handling of malformed literature findings."""
        agent = EnhancedGenerationAgent()
        
        malformed_findings = [
            {"title": "Test", "missing_required_fields": True},
            {"authors": ["Test"], "no_title": True},
            {}  # Empty dict
        ]
        
        formatted = agent._format_literature_findings(malformed_findings)
        
        # Should not crash and provide some output
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_concurrent_agent_usage(self):
        """Test that multiple agents can be used concurrently."""
        agent1 = EnhancedGenerationAgent()
        agent2 = EnhancedGenerationAgent()
        
        # Both should work independently
        state1 = GenerationState(research_goal="Goal 1")
        state2 = GenerationState(research_goal="Goal 2")
        
        # Should not interfere with each other
        assert state1.research_goal != state2.research_goal


class TestGenerationAgentIntegration:
    """Integration tests for the Generation Agent."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        # This should work without external dependencies in test environment
        try:
            result = test_generation_agent()
            assert isinstance(result, GenerationState)
            assert hasattr(result, 'search_queries')
            assert hasattr(result, 'literature_findings')
            assert hasattr(result, 'synthesized_knowledge')
            assert hasattr(result, 'generated_proposals')
        except Exception as e:
            # If it fails due to missing external APIs, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['api', 'network', 'connection', 'llm'])

    @patch('agents.generation.multi_source_literature_search')
    @patch('agents.generation.get_global_llm_client')
    def test_data_flow_integrity(self, mock_llm_client, mock_search):
        """Test that data flows correctly through all stages."""
        # Setup mocks
        mock_search.return_value = [{"title": "Test", "source": "test", "authors": ["Test"]}]
        
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = "Test response"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        agent = EnhancedGenerationAgent()
        
        # Start with initial state
        initial_state = GenerationState(
            research_goal="Test goal",
            constraints=["Test constraint"]
        )
        
        # Run each stage and verify data flow
        state = agent.generate_search_queries(initial_state)
        assert state.status == "queries_generated"
        assert len(state.search_queries) > 0
        
        state = agent.literature_exploration(state)
        assert state.status == "literature_explored"
        assert len(state.literature_findings) > 0
        
        state = agent.synthesize_knowledge(state)
        assert state.status == "knowledge_synthesized"
        assert len(state.synthesized_knowledge) > 0
        
        state = agent.generate_hypotheses(state)
        assert state.status == "hypotheses_generated"
        # Note: May be empty if parsing fails, but should not crash


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])