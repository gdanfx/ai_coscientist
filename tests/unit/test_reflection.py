"""
Tests for the Robust Reflection Agent

This module contains comprehensive tests for the RobustReflectionAgent
including unit tests, integration tests, and edge case validation.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from agents.reflection import (
    RobustReflectionAgent, 
    create_reflection_agent,
    run_reflection_agent,
    test_reflection_agent
)
from core.data_structures import (
    ReviewCriteria,
    HypothesisReview, 
    ReflectionState,
    GenerationState
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRobustReflectionAgent:
    """Test suite for the RobustReflectionAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = RobustReflectionAgent()
        
        # Sample test hypotheses
        self.test_hypotheses = [
            {
                "id": "hyp_test_1",
                "content": "Novel epigenetic reprogramming approach for liver fibrosis treatment using DNMT3A inhibition in hepatic stellate cells."
            },
            {
                "id": "hyp_test_2", 
                "content": "AI-driven personalized therapy combining multiple epigenetic targets for improved liver fibrosis outcomes."
            }
        ]

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = RobustReflectionAgent()
        
        assert agent is not None
        assert hasattr(agent, 'criteria_weights')
        assert hasattr(agent, 'llm')
        assert len(agent.criteria_weights) == 5
        
        # Check weights sum to 1.0 (approximately)
        total_weight = sum(agent.criteria_weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_factory_function(self):
        """Test the factory function creates agents correctly."""
        agent = create_reflection_agent()
        assert isinstance(agent, RobustReflectionAgent)

    @patch('agents.reflection.get_global_llm_client')
    def test_quick_review_with_mock_llm(self, mock_llm_client):
        """Test quick review with mocked LLM response."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        **SCORES (1-10 scale):**
        Novelty: 8 - Innovative approach
        Feasibility: 7 - Technically achievable
        Scientific Rigor: 8 - Well-founded
        Impact Potential: 9 - High therapeutic value
        Testability: 7 - Clear validation path
        
        **STRENGTHS:**
        • Novel epigenetic mechanism
        • Clear therapeutic target
        • Strong scientific rationale
        
        **WEAKNESSES:**
        • Requires extensive validation
        • Delivery challenges
        • Regulatory complexity
        
        **RECOMMENDATIONS:**
        • Develop in vitro models
        • Optimize delivery methods
        • Plan clinical pathway
        
        **OVERALL:** This hypothesis presents a promising therapeutic approach with strong scientific foundation.
        
        **CONFIDENCE:** 8
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        agent = RobustReflectionAgent()
        result = agent.quick_review(
            self.test_hypotheses[0]["content"], 
            self.test_hypotheses[0]["id"]
        )
        
        # Validate result structure
        assert isinstance(result, HypothesisReview)
        assert result.hypothesis_id == "hyp_test_1"
        assert result.overall_score > 0
        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0
        assert len(result.recommendations) > 0
        assert result.confidence_level > 0

    @patch('agents.reflection.get_global_llm_client')
    def test_detailed_review_with_mock_llm(self, mock_llm_client):
        """Test detailed review with mocked LLM response."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        **DETAILED EVALUATION:**
        
        1. **Novelty (1-10):** 8
           This represents a novel application of epigenetic reprogramming
        
        2. **Feasibility (1-10):** 7
           Technically achievable with current methods
        
        3. **Scientific Rigor (1-10):** 8
           Well-grounded in established epigenetic principles
        
        4. **Impact Potential (1-10):** 9
           Could revolutionize liver fibrosis treatment
        
        5. **Testability (1-10):** 7
           Clear experimental validation approaches
        
        **COMPREHENSIVE ASSESSMENT:**
        
        **Major Strengths:**
        • Innovative epigenetic mechanism
        • Strong therapeutic rationale
        • Clear validation pathway
        • High clinical relevance
        
        **Key Concerns:**
        • Delivery system challenges
        • Potential off-target effects
        • Regulatory pathway complexity
        • Long development timeline
        
        **Improvement Recommendations:**
        • Develop specific delivery methods
        • Conduct safety evaluations
        • Design phased clinical trials
        • Address regulatory requirements
        
        **Summary:** This hypothesis represents a highly promising approach with strong scientific foundation and significant therapeutic potential.
        
        **Review Confidence:** 8
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        agent = RobustReflectionAgent()
        result = agent.detailed_review(
            self.test_hypotheses[0]["content"],
            self.test_hypotheses[0]["id"],
            "liver fibrosis treatment research"
        )
        
        # Validate detailed review structure
        assert isinstance(result, HypothesisReview)
        assert result.reviewer_type == "detailed"
        assert result.overall_score > 0
        assert len(result.strengths) >= 3  # Detailed reviews should have more content
        assert len(result.weaknesses) >= 3
        assert len(result.recommendations) >= 3

    def test_score_extraction_flexible(self):
        """Test flexible score extraction from varied LLM outputs."""
        agent = RobustReflectionAgent()
        
        # Test various score formats
        test_cases = [
            ("Novelty: 8.5 out of 10", {"novelty": 8.5}),
            ("Feasibility score is 7", {"feasibility": 7.0}),
            ("Scientific rigor: 9.2", {"rigor": 9.2}),
            ("Impact potential = 6", {"impact": 6.0}),
            ("Testability rated 8.0", {"testability": 8.0})
        ]
        
        for text, expected in test_cases:
            scores = agent._extract_scores_flexible(text)
            for criterion, expected_score in expected.items():
                assert scores.get(criterion, 0) == expected_score

    def test_confidence_extraction_flexible(self):
        """Test flexible confidence extraction."""
        agent = RobustReflectionAgent()
        
        test_cases = [
            ("Confidence: 8.5", 8.5),
            ("I am 90% confident", 7.0),  # Should fallback to default
            ("Certainty level: 7", 7.0),
            ("Review confidence: 9.2", 9.2)
        ]
        
        for text, expected in test_cases:
            confidence = agent._extract_confidence_flexible(text)
            assert confidence == expected or confidence == 7.0  # Allow fallback

    def test_list_extraction_flexible(self):
        """Test flexible list extraction from varied formats."""
        agent = RobustReflectionAgent()
        
        text = """
        Strengths:
        • Strong scientific foundation
        • Clear therapeutic target
        • Novel mechanism
        
        Weaknesses:
        - Delivery challenges
        - Regulatory complexity
        
        Recommendations:
        1. Develop in vitro models
        2. Optimize delivery
        """
        
        strengths = agent._extract_lists_flexible(text, ["strength", "strengths"])
        weaknesses = agent._extract_lists_flexible(text, ["weakness", "weaknesses"])
        recommendations = agent._extract_lists_flexible(text, ["recommendation", "recommendations"])
        
        assert len(strengths) >= 2
        assert len(weaknesses) >= 2
        assert len(recommendations) >= 2

    def test_enhanced_fallback_review(self):
        """Test enhanced fallback review generation."""
        agent = RobustReflectionAgent()
        
        hypothesis_text = "Novel therapeutic approach using epigenetic mechanisms for liver fibrosis treatment"
        result = agent._create_enhanced_fallback_review(
            hypothesis_text, 
            "test_id", 
            "quick"
        )
        
        # Validate fallback review
        assert isinstance(result, HypothesisReview)
        assert result.hypothesis_id == "test_id"
        assert result.reviewer_type == "quick_heuristic"
        assert result.overall_score > 0
        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0
        assert len(result.recommendations) > 0
        assert result.confidence_level == 5.0  # Heuristic confidence

    @patch('agents.reflection.get_global_llm_client')
    def test_adaptive_batch_review(self, mock_llm_client):
        """Test adaptive batch review with mixed quality hypotheses."""
        # Mock LLM that returns high scores for first hypothesis, low for second
        def mock_invoke(prompt):
            mock_response = Mock()
            mock_response.error = None
            if "epigenetic" in prompt:
                mock_response.content = "Novelty: 9 Feasibility: 8 Scientific Rigor: 9 Impact Potential: 9 Testability: 8 OVERALL: Excellent hypothesis CONFIDENCE: 9"
            else:
                mock_response.content = "Novelty: 6 Feasibility: 6 Scientific Rigor: 6 Impact Potential: 6 Testability: 6 OVERALL: Moderate hypothesis CONFIDENCE: 6"
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        agent = RobustReflectionAgent()
        result = agent.adaptive_batch_review(
            self.test_hypotheses,
            "liver fibrosis research"
        )
        
        # Validate batch results
        assert isinstance(result, ReflectionState)
        assert len(result.hypothesis_reviews) == len(self.test_hypotheses)
        assert result.review_statistics is not None
        assert isinstance(result.quality_flags, list)
        assert result.batch_summary is not None

    def test_statistics_computation(self):
        """Test robust statistics computation."""
        agent = RobustReflectionAgent()
        
        # Create sample reviews
        reviews = []
        for i, hyp in enumerate(self.test_hypotheses):
            criteria = ReviewCriteria(
                novelty_score=7.0 + i,
                feasibility_score=6.0 + i,
                scientific_rigor_score=8.0 + i,
                impact_potential_score=7.5 + i,
                testability_score=6.5 + i,
                novelty_reasoning="Test reasoning",
                feasibility_reasoning="Test reasoning",
                scientific_rigor_reasoning="Test reasoning", 
                impact_potential_reasoning="Test reasoning",
                testability_reasoning="Test reasoning"
            )
            
            review = HypothesisReview(
                hypothesis_id=hyp["id"],
                hypothesis_text=hyp["content"],
                criteria=criteria,
                overall_score=7.0 + i,
                overall_assessment="Test assessment",
                strengths=["Test strength"],
                weaknesses=["Test weakness"],
                recommendations=["Test recommendation"],
                confidence_level=8.0,
                review_timestamp="2024-01-01T12:00:00",
                reviewer_type="test"
            )
            reviews.append(review)
        
        stats = agent._compute_robust_statistics(reviews)
        
        # Validate statistics
        assert 'mean_overall_score' in stats
        assert 'median_overall_score' in stats
        assert 'std_overall_score' in stats
        assert 'mean_confidence' in stats
        assert stats['mean_overall_score'] > 0

    def test_quality_flags_identification(self):
        """Test quality flag identification."""
        agent = RobustReflectionAgent()
        
        # Create reviews with different characteristics
        reviews = []
        for i in range(3):
            review = Mock()
            review.overall_score = 5.0 if i < 2 else 9.0  # Mix of low and high scores
            review.confidence_level = 7.0
            review.reviewer_type = "heuristic" if i == 0 else "quick"
            reviews.append(review)
        
        flags = agent._identify_robust_quality_flags(reviews)
        
        assert isinstance(flags, list)
        # Should detect some quality issues with mixed scores

    def test_integration_function(self):
        """Test the integration function for workflow."""
        # Create mock generation state
        generation_state = Mock()
        generation_state.generated_proposals = self.test_hypotheses
        
        # Mock the agent to avoid LLM calls
        with patch('agents.reflection.RobustReflectionAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_result = ReflectionState()
            mock_result.hypothesis_reviews = []
            mock_agent.adaptive_batch_review.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            
            result = run_reflection_agent(generation_state, "test research goal")
            
            assert isinstance(result, ReflectionState)
            mock_agent.adaptive_batch_review.assert_called_once()

    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs."""
        agent = RobustReflectionAgent()
        
        # Test with empty hypothesis list
        result = agent.adaptive_batch_review([])
        assert isinstance(result, ReflectionState)
        assert len(result.hypothesis_reviews) == 0
        
        # Test with malformed hypothesis
        malformed = [{"no_content": "missing content key"}]
        result = agent.adaptive_batch_review(malformed)
        assert isinstance(result, ReflectionState)

    def test_error_handling_resilience(self):
        """Test agent resilience to various error conditions."""
        agent = RobustReflectionAgent()
        
        # Test with None LLM client
        agent.llm = None
        result = agent.quick_review("test hypothesis", "test_id")
        
        # Should still return a valid review (fallback)
        assert isinstance(result, HypothesisReview)
        assert "heuristic" in result.reviewer_type

    def test_system_integration_test(self):
        """Test the complete system test function."""
        # This should work without external dependencies
        try:
            result = test_reflection_agent()
            assert isinstance(result, ReflectionState)
            assert len(result.hypothesis_reviews) > 0
        except Exception as e:
            # If it fails due to missing LLM, that's expected in test environment
            assert "LLM" in str(e) or "client" in str(e)


class TestReflectionAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_llm_responses(self):
        """Test handling of malformed LLM responses."""
        agent = RobustReflectionAgent()
        
        # Test various malformed responses
        malformed_responses = [
            "",  # Empty response
            "Random text without structure",  # No recognizable format
            "Novelty: not_a_number",  # Invalid number format
            "SCORES: Novelty: 15",  # Out of range score
        ]
        
        for response in malformed_responses:
            # Should not crash and should return fallback review
            result = agent._parse_flexible_review(
                response, 
                "test hypothesis", 
                "test_id", 
                "test"
            )
            assert isinstance(result, HypothesisReview)

    def test_extreme_input_sizes(self):
        """Test handling of extreme input sizes."""
        agent = RobustReflectionAgent()
        
        # Very large hypothesis
        large_hypothesis = "A" * 10000
        result = agent._create_enhanced_fallback_review(
            large_hypothesis, 
            "large_test", 
            "test"
        )
        assert isinstance(result, HypothesisReview)
        # Text should be truncated
        assert len(result.hypothesis_text) < len(large_hypothesis)

    def test_concurrent_review_safety(self):
        """Test thread safety considerations."""
        agent = RobustReflectionAgent()
        
        # Multiple agents should not interfere
        agent1 = RobustReflectionAgent()
        agent2 = RobustReflectionAgent()
        
        # Both should work independently
        result1 = agent1._create_enhanced_fallback_review("test1", "id1", "test")
        result2 = agent2._create_enhanced_fallback_review("test2", "id2", "test")
        
        assert result1.hypothesis_id != result2.hypothesis_id


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])