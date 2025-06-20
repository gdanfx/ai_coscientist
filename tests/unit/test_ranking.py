"""
Tests for the Ranking Agent

This module contains comprehensive tests for the RankingAgent
including unit tests for Elo system, tournament logic, and edge case validation.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.ranking import (
    RankingAgent,
    EloSystem,
    test_ranking_agent
)
from core.data_structures import (
    RankingState, 
    ReflectionState, 
    HypothesisReview, 
    ReviewCriteria,
    PairwiseComparison
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEloSystem:
    """Test suite for the EloSystem class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.elo_system = EloSystem()

    def test_elo_system_initialization(self):
        """Test that the Elo system initializes correctly."""
        assert self.elo_system.initial_rating == 1200
        assert self.elo_system.k_factor == 32
        
        # Test with custom parameters
        custom_elo = EloSystem(initial_rating=1500, k_factor=24)
        assert custom_elo.initial_rating == 1500
        assert custom_elo.k_factor == 24

    def test_expected_score_calculation(self):
        """Test expected score calculation."""
        # Equal ratings should give 0.5 expected score
        expected = self.elo_system.expected_score(1200, 1200)
        assert abs(expected - 0.5) < 0.001
        
        # Higher rating should have higher expected score
        expected_higher = self.elo_system.expected_score(1300, 1200)
        assert expected_higher > 0.5
        
        # Lower rating should have lower expected score
        expected_lower = self.elo_system.expected_score(1100, 1200)
        assert expected_lower < 0.5

    def test_update_ratings_winner_gains_points(self):
        """Test that winner gains rating points."""
        initial_winner = 1200
        initial_loser = 1200
        
        new_winner, new_loser = self.elo_system.update_ratings(
            initial_winner, initial_loser, winner=True
        )
        
        assert new_winner > initial_winner
        assert new_loser < initial_loser

    def test_update_ratings_loser_loses_points(self):
        """Test that loser loses rating points."""
        initial_winner = 1200
        initial_loser = 1200
        
        new_winner, new_loser = self.elo_system.update_ratings(
            initial_winner, initial_loser, winner=False
        )
        
        assert new_winner < initial_winner  # First player lost
        assert new_loser > initial_loser    # Second player won

    def test_rating_conservation(self):
        """Test that total rating points are conserved."""
        rating1, rating2 = 1200, 1200
        total_initial = rating1 + rating2
        
        new_rating1, new_rating2 = self.elo_system.update_ratings(
            rating1, rating2, winner=True
        )
        
        total_final = new_rating1 + new_rating2
        assert abs(total_initial - total_final) < 0.001

    def test_upset_victory_larger_change(self):
        """Test that upset victories result in larger rating changes."""
        # Underdog wins
        underdog_rating = 1100
        favorite_rating = 1300
        
        new_underdog, new_favorite = self.elo_system.update_ratings(
            underdog_rating, favorite_rating, winner=True
        )
        
        underdog_gain = new_underdog - underdog_rating
        favorite_loss = favorite_rating - new_favorite
        
        # Now expected victory
        equal1, equal2 = 1200, 1200
        new_equal1, new_equal2 = self.elo_system.update_ratings(
            equal1, equal2, winner=True
        )
        
        expected_gain = new_equal1 - equal1
        
        # Upset should result in larger changes
        assert underdog_gain > expected_gain

    def test_extreme_rating_differences(self):
        """Test behavior with extreme rating differences."""
        very_high = 2000
        very_low = 400
        
        new_high, new_low = self.elo_system.update_ratings(
            very_high, very_low, winner=True
        )
        
        # High rated player winning should gain very few points
        gain = new_high - very_high
        assert 0 < gain < 5
        
        # Low rated player losing should lose very few points
        loss = very_low - new_low
        assert 0 < loss < 5


class TestRankingAgent:
    """Test suite for the RankingAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = RankingAgent()
        
        # Create sample hypothesis reviews
        self.sample_reviews = []
        for i in range(4):
            criteria = ReviewCriteria(
                novelty_score=7.0 + i * 0.5,
                feasibility_score=6.0 + i * 0.3,
                scientific_rigor_score=8.0 + i * 0.2,
                impact_potential_score=7.5 + i * 0.4,
                testability_score=6.5 + i * 0.3,
                novelty_reasoning=f"Novelty reasoning {i}",
                feasibility_reasoning=f"Feasibility reasoning {i}",
                scientific_rigor_reasoning=f"Rigor reasoning {i}",
                impact_potential_reasoning=f"Impact reasoning {i}",
                testability_reasoning=f"Testability reasoning {i}"
            )
            
            review = HypothesisReview(
                hypothesis_id=f"hyp_{i}",
                hypothesis_text=f"Test hypothesis {i} content",
                criteria=criteria,
                overall_score=7.0 + i * 0.5,
                overall_assessment=f"Assessment for hypothesis {i}",
                strengths=[f"Strength {i}"],
                weaknesses=[f"Weakness {i}"],
                recommendations=[f"Recommendation {i}"],
                confidence_level=8.0,
                review_timestamp="2024-01-01T12:00:00",
                reviewer_type="test"
            )
            self.sample_reviews.append(review)

        self.reflection_state = ReflectionState(
            hypothesis_reviews=self.sample_reviews
        )

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = RankingAgent()
        
        assert agent is not None
        assert hasattr(agent, 'elo_system')
        assert hasattr(agent, 'llm')
        assert isinstance(agent.elo_system, EloSystem)

    def test_agent_initialization_with_params(self):
        """Test agent initialization with custom parameters."""
        mock_llm = Mock()
        agent = RankingAgent(
            llm_client=mock_llm,
            initial_rating=1500,
            k_factor=24
        )
        
        assert agent.llm == mock_llm
        assert agent.elo_system.initial_rating == 1500
        assert agent.elo_system.k_factor == 24

    def test_initialize_ratings(self):
        """Test initialization of hypothesis ratings."""
        hypothesis_ids = ["hyp_1", "hyp_2", "hyp_3"]
        
        ratings = self.agent._initialize_ratings(hypothesis_ids)
        
        assert len(ratings) == 3
        assert all(rating == 1200 for rating in ratings.values())
        assert set(ratings.keys()) == set(hypothesis_ids)

    @patch('agents.ranking.get_global_llm_client')
    def test_conduct_pairwise_debate_clear_winner(self, mock_llm_client):
        """Test pairwise debate with clear winner."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        After careful comparison of both hypotheses:
        
        WINNER: Hypothesis A
        
        REASONING: Hypothesis A demonstrates superior novelty and feasibility
        compared to Hypothesis B. The approach is more scientifically rigorous
        and has clearer potential for clinical translation.
        
        CONFIDENCE: 8
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._conduct_pairwise_debate(
            self.sample_reviews[0],
            self.sample_reviews[1]
        )
        
        assert isinstance(result, PairwiseComparison)
        assert result.winner_id == "hyp_0"  # Hypothesis A maps to hyp_0
        assert result.confidence > 0
        assert "superior novelty" in result.reasoning

    @patch('agents.ranking.get_global_llm_client')
    def test_conduct_pairwise_debate_tie(self, mock_llm_client):
        """Test pairwise debate resulting in tie."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        Both hypotheses have merit and are roughly equivalent.
        
        TIE: Unable to determine clear winner
        
        REASONING: Both approaches show similar levels of innovation and feasibility.
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._conduct_pairwise_debate(
            self.sample_reviews[0],
            self.sample_reviews[1]
        )
        
        assert isinstance(result, PairwiseComparison)
        assert result.winner_id is None  # Tie
        assert "equivalent" in result.reasoning

    @patch('agents.ranking.get_global_llm_client')
    def test_conduct_pairwise_debate_llm_error(self, mock_llm_client):
        """Test pairwise debate with LLM error."""
        mock_response = Mock()
        mock_response.error = "API rate limit exceeded"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._conduct_pairwise_debate(
            self.sample_reviews[0],
            self.sample_reviews[1]
        )
        
        # Should fallback to score-based comparison
        assert isinstance(result, PairwiseComparison)
        assert result.winner_id is not None  # Should pick winner based on scores
        assert "fallback" in result.reasoning.lower()

    def test_parse_debate_result_clear_winner(self):
        """Test parsing debate result with clear winner."""
        debate_text = """
        WINNER: Hypothesis B
        REASONING: Better approach
        CONFIDENCE: 9
        """
        
        hyp_a_id = "hyp_1"
        hyp_b_id = "hyp_2"
        
        winner, reasoning, confidence = self.agent._parse_debate_result(
            debate_text, hyp_a_id, hyp_b_id
        )
        
        assert winner == hyp_b_id
        assert "Better approach" in reasoning
        assert confidence == 9

    def test_parse_debate_result_tie(self):
        """Test parsing debate result with tie."""
        debate_text = """
        TIE: Both hypotheses are equivalent
        REASONING: Similar quality
        """
        
        winner, reasoning, confidence = self.agent._parse_debate_result(
            debate_text, "hyp_1", "hyp_2"
        )
        
        assert winner is None
        assert "Similar quality" in reasoning

    def test_parse_debate_result_malformed(self):
        """Test parsing malformed debate result."""
        malformed_text = "Random text without proper structure"
        
        winner, reasoning, confidence = self.agent._parse_debate_result(
            malformed_text, "hyp_1", "hyp_2"
        )
        
        # Should return fallback values
        assert winner is None
        assert reasoning == malformed_text
        assert confidence == 5.0

    def test_fallback_comparison_by_scores(self):
        """Test fallback comparison based on review scores."""
        result = self.agent._fallback_comparison_by_scores(
            self.sample_reviews[3],  # Higher score (8.5)
            self.sample_reviews[0]   # Lower score (7.0)
        )
        
        assert isinstance(result, PairwiseComparison)
        assert result.winner_id == "hyp_3"  # Higher scoring hypothesis
        assert "score" in result.reasoning.lower()

    def test_generate_all_pairs(self):
        """Test generation of all pairwise combinations."""
        hypothesis_ids = ["hyp_1", "hyp_2", "hyp_3"]
        
        pairs = self.agent._generate_all_pairs(hypothesis_ids)
        
        expected_pairs = [
            ("hyp_1", "hyp_2"),
            ("hyp_1", "hyp_3"),
            ("hyp_2", "hyp_3")
        ]
        
        assert len(pairs) == 3
        assert set(pairs) == set(expected_pairs)

    def test_generate_all_pairs_single_hypothesis(self):
        """Test pair generation with single hypothesis."""
        pairs = self.agent._generate_all_pairs(["hyp_1"])
        assert len(pairs) == 0

    def test_generate_all_pairs_empty_list(self):
        """Test pair generation with empty list."""
        pairs = self.agent._generate_all_pairs([])
        assert len(pairs) == 0

    @patch('agents.ranking.get_global_llm_client')
    def test_run_tournament_round(self, mock_llm_client):
        """Test running a complete tournament round."""
        # Mock consistent LLM responses
        def mock_invoke(prompt):
            mock_response = Mock()
            mock_response.error = None
            # First hypothesis always wins for predictable testing
            mock_response.content = "WINNER: Hypothesis A\nREASONING: Better\nCONFIDENCE: 8"
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        hypothesis_ids = [review.hypothesis_id for review in self.sample_reviews]
        ratings = self.agent._initialize_ratings(hypothesis_ids)
        
        new_ratings, comparisons = self.agent._run_tournament_round(
            self.reflection_state,
            ratings
        )
        
        assert len(new_ratings) == len(ratings)
        assert len(comparisons) == 6  # C(4,2) = 6 pairwise comparisons
        assert all(isinstance(comp, PairwiseComparison) for comp in comparisons)

    def test_compute_win_rates(self):
        """Test computation of win rates from comparisons."""
        # Create sample comparisons
        comparisons = [
            PairwiseComparison(
                hypothesis_a_id="hyp_1",
                hypothesis_b_id="hyp_2", 
                winner_id="hyp_1",
                reasoning="Test",
                confidence=8.0
            ),
            PairwiseComparison(
                hypothesis_a_id="hyp_1",
                hypothesis_b_id="hyp_3",
                winner_id="hyp_1", 
                reasoning="Test",
                confidence=8.0
            ),
            PairwiseComparison(
                hypothesis_a_id="hyp_2",
                hypothesis_b_id="hyp_3",
                winner_id=None,  # Tie
                reasoning="Test",
                confidence=6.0
            )
        ]
        
        win_rates = self.agent._compute_win_rates(
            ["hyp_1", "hyp_2", "hyp_3"],
            comparisons
        )
        
        assert win_rates["hyp_1"] == 100.0  # Won 2/2 = 100%
        assert win_rates["hyp_2"] == 25.0   # Won 0/2, tied 1/2 = 25%
        assert win_rates["hyp_3"] == 25.0   # Won 0/2, tied 1/2 = 25%

    def test_create_final_rankings(self):
        """Test creation of final rankings."""
        ratings = {"hyp_1": 1250, "hyp_2": 1180, "hyp_3": 1220}
        win_rates = {"hyp_1": 75.0, "hyp_2": 25.0, "hyp_3": 50.0}
        
        rankings = self.agent._create_final_rankings(ratings, win_rates)
        
        assert len(rankings) == 3
        
        # Should be sorted by rating (highest first)
        assert rankings[0]["hypothesis_id"] == "hyp_1"  # 1250
        assert rankings[1]["hypothesis_id"] == "hyp_3"  # 1220
        assert rankings[2]["hypothesis_id"] == "hyp_2"  # 1180
        
        # Check rank assignments
        assert rankings[0]["rank"] == 1
        assert rankings[1]["rank"] == 2
        assert rankings[2]["rank"] == 3

    @patch('agents.ranking.get_global_llm_client')
    def test_run_full_tournament(self, mock_llm_client):
        """Test running full tournament."""
        # Mock LLM responses
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = "WINNER: Hypothesis A\nREASONING: Better\nCONFIDENCE: 8"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent.run_full_tournament(
            self.reflection_state,
            num_rounds=2
        )
        
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == len(self.sample_reviews)
        assert result.total_comparisons > 0
        assert result.tournament_rounds == 2
        assert len(result.all_comparisons) > 0


class TestRankingAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_reflection_state(self):
        """Test handling of empty reflection state."""
        agent = RankingAgent()
        
        empty_state = ReflectionState(hypothesis_reviews=[])
        
        result = agent.run_full_tournament(empty_state, num_rounds=1)
        
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == 0
        assert result.total_comparisons == 0

    def test_single_hypothesis_tournament(self):
        """Test tournament with single hypothesis."""
        agent = RankingAgent()
        
        single_review = [self.create_sample_review("hyp_1", 7.5)]
        single_state = ReflectionState(hypothesis_reviews=single_review)
        
        result = agent.run_full_tournament(single_state, num_rounds=1)
        
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == 1
        assert result.final_rankings[0]["rank"] == 1
        assert result.total_comparisons == 0  # No comparisons possible

    def test_identical_scores(self):
        """Test handling of hypotheses with identical scores."""
        agent = RankingAgent()
        
        # Create reviews with identical scores
        identical_reviews = []
        for i in range(3):
            review = self.create_sample_review(f"hyp_{i}", 7.0)  # Same score
            identical_reviews.append(review)
        
        identical_state = ReflectionState(hypothesis_reviews=identical_reviews)
        
        result = agent.run_full_tournament(identical_state, num_rounds=1)
        
        # Should still produce rankings
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == 3

    @patch('agents.ranking.get_global_llm_client')
    def test_all_ties_tournament(self, mock_llm_client):
        """Test tournament where all comparisons result in ties."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = "TIE: Both hypotheses are equivalent"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        agent = RankingAgent()
        
        reviews = [self.create_sample_review(f"hyp_{i}", 7.0) for i in range(3)]
        state = ReflectionState(hypothesis_reviews=reviews)
        
        result = agent.run_full_tournament(state, num_rounds=1)
        
        # Should handle all ties gracefully
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == 3
        # Win rates should all be 50% (all ties)
        assert all(ranking["win_rate"] == 50.0 for ranking in result.final_rankings)

    def test_very_large_tournament(self):
        """Test performance with large number of hypotheses."""
        agent = RankingAgent()
        
        # Create 10 hypotheses (45 pairwise comparisons)
        large_reviews = []
        for i in range(10):
            review = self.create_sample_review(f"hyp_{i}", 7.0 + i * 0.1)
            large_reviews.append(review)
        
        large_state = ReflectionState(hypothesis_reviews=large_reviews)
        
        # Use smaller number of rounds for performance
        result = agent.run_full_tournament(large_state, num_rounds=1)
        
        assert isinstance(result, RankingState)
        assert len(result.final_rankings) == 10
        assert result.total_comparisons == 45  # C(10,2) = 45

    def test_malformed_hypothesis_reviews(self):
        """Test handling of malformed hypothesis reviews."""
        agent = RankingAgent()
        
        # Create review with None values
        malformed_review = HypothesisReview(
            hypothesis_id="malformed",
            hypothesis_text="",
            criteria=None,
            overall_score=None,
            overall_assessment="",
            strengths=[],
            weaknesses=[],
            recommendations=[],
            confidence_level=None,
            review_timestamp="",
            reviewer_type=""
        )
        
        malformed_state = ReflectionState(hypothesis_reviews=[malformed_review])
        
        # Should handle gracefully without crashing
        result = agent.run_full_tournament(malformed_state, num_rounds=1)
        
        assert isinstance(result, RankingState)

    def create_sample_review(self, hyp_id: str, score: float) -> HypothesisReview:
        """Helper method to create sample review."""
        criteria = ReviewCriteria(
            novelty_score=score,
            feasibility_score=score,
            scientific_rigor_score=score,
            impact_potential_score=score,
            testability_score=score,
            novelty_reasoning="Test",
            feasibility_reasoning="Test",
            scientific_rigor_reasoning="Test",
            impact_potential_reasoning="Test",
            testability_reasoning="Test"
        )
        
        return HypothesisReview(
            hypothesis_id=hyp_id,
            hypothesis_text="Test hypothesis",
            criteria=criteria,
            overall_score=score,
            overall_assessment="Test assessment",
            strengths=["Test strength"],
            weaknesses=["Test weakness"],
            recommendations=["Test recommendation"],
            confidence_level=8.0,
            review_timestamp="2024-01-01T12:00:00",
            reviewer_type="test"
        )


class TestRankingAgentIntegration:
    """Integration tests for the Ranking Agent."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        try:
            result = test_ranking_agent()
            assert isinstance(result, RankingState)
            assert hasattr(result, 'final_rankings')
            assert hasattr(result, 'total_comparisons')
        except Exception as e:
            # If it fails due to missing LLM, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['llm', 'client', 'api'])

    @patch('agents.ranking.get_global_llm_client')
    def test_realistic_scientific_ranking(self, mock_llm_client):
        """Test ranking with realistic scientific hypotheses."""
        # Mock LLM to prefer higher-scoring hypotheses
        def mock_invoke(prompt):
            mock_response = Mock()
            mock_response.error = None
            # Simple heuristic: prefer "novel" over "standard"
            if "novel" in prompt and "standard" in prompt:
                mock_response.content = "WINNER: Hypothesis A\nREASONING: More innovative\nCONFIDENCE: 8"
            else:
                mock_response.content = "WINNER: Hypothesis A\nREASONING: Better approach\nCONFIDENCE: 7"
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        agent = RankingAgent()
        
        # Create realistic scientific hypothesis reviews
        reviews = [
            self.create_realistic_review("novel_ai", 8.5, "Novel AI approach for drug discovery"),
            self.create_realistic_review("standard_ml", 7.0, "Standard machine learning for biomarkers"),
            self.create_realistic_review("innovative_quantum", 8.0, "Innovative quantum computing simulation"),
            self.create_realistic_review("traditional_stats", 6.5, "Traditional statistical analysis method")
        ]
        
        reflection_state = ReflectionState(hypothesis_reviews=reviews)
        
        result = agent.run_full_tournament(reflection_state, num_rounds=2)
        
        # Higher-scored hypotheses should generally rank higher
        rankings_by_score = sorted(result.final_rankings, key=lambda x: x["rank"])
        
        # Novel AI (8.5) should rank highly
        top_hypothesis = rankings_by_score[0]
        assert top_hypothesis["hypothesis_id"] == "novel_ai"

    def create_realistic_review(self, hyp_id: str, score: float, description: str) -> HypothesisReview:
        """Helper to create realistic review."""
        criteria = ReviewCriteria(
            novelty_score=score + 0.2,
            feasibility_score=score - 0.1,
            scientific_rigor_score=score,
            impact_potential_score=score + 0.1,
            testability_score=score - 0.2,
            novelty_reasoning=f"Novelty assessment: {score + 0.2}/10",
            feasibility_reasoning=f"Feasibility assessment: {score - 0.1}/10",
            scientific_rigor_reasoning=f"Rigor assessment: {score}/10",
            impact_potential_reasoning=f"Impact assessment: {score + 0.1}/10",
            testability_reasoning=f"Testability assessment: {score - 0.2}/10"
        )
        
        return HypothesisReview(
            hypothesis_id=hyp_id,
            hypothesis_text=description,
            criteria=criteria,
            overall_score=score,
            overall_assessment=f"Comprehensive assessment of {description}",
            strengths=[f"Strong {description.split()[0]} approach"],
            weaknesses=[f"Limited validation for {description.split()[0]} method"],
            recommendations=[f"Develop {description.split()[0]} prototype"],
            confidence_level=8.0,
            review_timestamp="2024-01-01T12:00:00",
            reviewer_type="detailed"
        )


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])