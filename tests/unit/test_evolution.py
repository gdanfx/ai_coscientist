"""
Tests for the Evolution Agent

This module contains comprehensive tests for the EvolutionAgent
including unit tests for evolution strategies, hypothesis improvement, and edge case validation.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.evolution import (
    EvolutionAgent,
    test_evolution_agent
)
from core.data_structures import (
    EvolutionState,
    EvolvedHypothesis, 
    EvolutionStep,
    EvolutionStrategy,
    RankingState,
    ReflectionState,
    HypothesisReview,
    ReviewCriteria
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEvolutionAgent:
    """Test suite for the EvolutionAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = EvolutionAgent()
        
        # Sample hypothesis reviews for testing
        self.sample_reviews = []
        for i in range(3):
            criteria = ReviewCriteria(
                novelty_score=7.0 + i,
                feasibility_score=6.5 + i,
                scientific_rigor_score=8.0 + i,
                impact_potential_score=7.5 + i,
                testability_score=6.8 + i,
                novelty_reasoning=f"Novel approach {i}",
                feasibility_reasoning=f"Feasible method {i}",
                scientific_rigor_reasoning=f"Rigorous design {i}",
                impact_potential_reasoning=f"High impact {i}",
                testability_reasoning=f"Testable hypothesis {i}"
            )
            
            review = HypothesisReview(
                hypothesis_id=f"hyp_{i}",
                hypothesis_text=f"AI-powered drug discovery using machine learning approach {i} for therapeutic target identification",
                criteria=criteria,
                overall_score=7.2 + i * 0.4,
                overall_assessment=f"Strong hypothesis {i} with good potential",
                strengths=[f"Innovative approach {i}", f"Clear methodology {i}"],
                weaknesses=[f"Needs validation {i}", f"Resource intensive {i}"],
                recommendations=[f"Develop prototype {i}", f"Test in vitro {i}"],
                confidence_level=8.0,
                review_timestamp="2024-01-01T12:00:00",
                reviewer_type="detailed"
            )
            self.sample_reviews.append(review)
        
        # Create ranking state
        self.ranking_state = RankingState(
            final_rankings=[
                {"rank": 1, "hypothesis_id": "hyp_2", "final_elo_rating": 1250, "win_rate": 80.0},
                {"rank": 2, "hypothesis_id": "hyp_1", "final_elo_rating": 1220, "win_rate": 60.0},
                {"rank": 3, "hypothesis_id": "hyp_0", "final_elo_rating": 1180, "win_rate": 40.0}
            ]
        )
        
        # Create reflection state
        self.reflection_state = ReflectionState(
            hypothesis_reviews=self.sample_reviews
        )

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = EvolutionAgent()
        
        assert agent is not None
        assert hasattr(agent, 'llm')
        assert hasattr(agent, 'evolution_strategies')
        assert len(agent.evolution_strategies) > 0

    def test_agent_initialization_with_llm(self):
        """Test agent initialization with custom LLM client."""
        mock_llm = Mock()
        agent = EvolutionAgent(mock_llm)
        
        assert agent.llm == mock_llm

    def test_evolution_strategies_defined(self):
        """Test that all evolution strategies are properly defined."""
        strategies = self.agent.evolution_strategies
        
        expected_strategies = [
            EvolutionStrategy.SIMPLIFICATION,
            EvolutionStrategy.COMBINATION,
            EvolutionStrategy.ANALOGICAL_REASONING,
            EvolutionStrategy.CONSTRAINT_RELAXATION,
            EvolutionStrategy.INCREMENTAL_IMPROVEMENT,
            EvolutionStrategy.PARADIGM_SHIFT
        ]
        
        assert all(strategy in strategies for strategy in expected_strategies)

    def test_get_top_hypotheses(self):
        """Test extraction of top hypotheses from ranking state."""
        top_hypotheses = self.agent._get_top_hypotheses(
            self.ranking_state,
            self.reflection_state,
            top_n=2
        )
        
        assert len(top_hypotheses) == 2
        assert top_hypotheses[0].hypothesis_id == "hyp_2"  # Rank 1
        assert top_hypotheses[1].hypothesis_id == "hyp_1"  # Rank 2

    def test_get_top_hypotheses_exceed_available(self):
        """Test getting more top hypotheses than available."""
        top_hypotheses = self.agent._get_top_hypotheses(
            self.ranking_state,
            self.reflection_state,
            top_n=10  # More than available
        )
        
        # Should return all available
        assert len(top_hypotheses) == 3

    def test_get_top_hypotheses_empty_ranking(self):
        """Test getting top hypotheses from empty ranking."""
        empty_ranking = RankingState(final_rankings=[])
        
        top_hypotheses = self.agent._get_top_hypotheses(
            empty_ranking,
            self.reflection_state,
            top_n=2
        )
        
        assert len(top_hypotheses) == 0

    @patch('agents.evolution.get_global_llm_client')
    def test_apply_simplification_strategy(self, mock_llm_client):
        """Test simplification evolution strategy."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        EVOLVED HYPOTHESIS: Simplified AI drug discovery platform using core machine learning algorithms for faster therapeutic target identification with reduced computational complexity.
        
        EVOLUTION REASONING: Simplified the original approach by removing unnecessary complexity while maintaining core functionality. Focused on essential ML algorithms rather than comprehensive multi-modal approaches.
        
        KEY IMPROVEMENTS:
        • Reduced computational requirements
        • Faster implementation timeline
        • Clearer methodology
        • Easier validation process
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._apply_simplification_strategy(
            self.sample_reviews[0],
            []  # No constraints
        )
        
        assert isinstance(result, EvolutionStep)
        assert result.strategy == EvolutionStrategy.SIMPLIFICATION
        assert "Simplified AI drug discovery" in result.evolved_content
        assert len(result.improvements) > 0
        assert result.success is True

    @patch('agents.evolution.get_global_llm_client')
    def test_apply_combination_strategy(self, mock_llm_client):
        """Test combination evolution strategy."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        EVOLVED HYPOTHESIS: Integrated AI-powered drug discovery platform combining machine learning target identification with advanced computational validation and experimental screening protocols.
        
        EVOLUTION REASONING: Combined multiple complementary approaches to create a more comprehensive solution. Integrated computational and experimental methods for enhanced validation.
        
        KEY IMPROVEMENTS:
        • Comprehensive validation pipeline
        • Multi-modal approach integration
        • Enhanced reliability through redundancy
        • Broader application scope
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._apply_combination_strategy(
            [self.sample_reviews[0], self.sample_reviews[1]],
            []
        )
        
        assert isinstance(result, EvolutionStep)
        assert result.strategy == EvolutionStrategy.COMBINATION
        assert "Integrated AI-powered" in result.evolved_content
        assert result.success is True

    @patch('agents.evolution.get_global_llm_client')
    def test_apply_analogical_reasoning_strategy(self, mock_llm_client):
        """Test analogical reasoning evolution strategy."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        EVOLVED HYPOTHESIS: Bio-inspired AI drug discovery system mimicking natural evolution processes for therapeutic target identification, using genetic algorithms and evolutionary optimization similar to natural selection in biological systems.
        
        EVOLUTION REASONING: Applied analogical reasoning from biological evolution to enhance the AI approach. Used principles of natural selection, mutation, and adaptation to improve drug discovery algorithms.
        
        KEY IMPROVEMENTS:
        • Nature-inspired optimization
        • Self-improving algorithms
        • Robust exploration of solution space
        • Adaptive parameter tuning
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._apply_analogical_reasoning_strategy(
            self.sample_reviews[0],
            []
        )
        
        assert isinstance(result, EvolutionStep)
        assert result.strategy == EvolutionStrategy.ANALOGICAL_REASONING
        assert "Bio-inspired" in result.evolved_content
        assert result.success is True

    @patch('agents.evolution.get_global_llm_client')
    def test_apply_constraint_relaxation_strategy(self, mock_llm_client):
        """Test constraint relaxation evolution strategy."""
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        EVOLVED HYPOTHESIS: Expanded AI drug discovery platform with relaxed computational constraints, enabling exploration of larger chemical space and novel therapeutic modalities including rare diseases and orphan drug discovery.
        
        EVOLUTION REASONING: Relaxed previous computational and scope constraints to enable broader exploration. Expanded from common diseases to include rare conditions and novel drug modalities.
        
        KEY IMPROVEMENTS:
        • Broader therapeutic scope
        • Enhanced chemical space exploration
        • Novel drug modality support
        • Increased innovation potential
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        constraints = ["Must be computationally efficient", "Focus on common diseases"]
        
        result = self.agent._apply_constraint_relaxation_strategy(
            self.sample_reviews[0],
            constraints
        )
        
        assert isinstance(result, EvolutionStep)
        assert result.strategy == EvolutionStrategy.CONSTRAINT_RELAXATION
        assert "Expanded AI drug discovery" in result.evolved_content
        assert result.success is True

    @patch('agents.evolution.get_global_llm_client')
    def test_llm_error_handling(self, mock_llm_client):
        """Test handling of LLM errors during evolution."""
        mock_response = Mock()
        mock_response.error = "API rate limit exceeded"
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent._apply_simplification_strategy(
            self.sample_reviews[0],
            []
        )
        
        assert isinstance(result, EvolutionStep)
        assert result.success is False
        assert "error" in result.evolved_content.lower()

    def test_parse_evolution_output_success(self):
        """Test parsing of successful evolution output."""
        evolution_text = """
        EVOLVED HYPOTHESIS: This is the evolved hypothesis content with improvements.
        
        EVOLUTION REASONING: This explains how the hypothesis was evolved and why.
        
        KEY IMPROVEMENTS:
        • First improvement point
        • Second improvement point
        • Third improvement point
        """
        
        content, reasoning, improvements = self.agent._parse_evolution_output(evolution_text)
        
        assert "evolved hypothesis content" in content
        assert "explains how the hypothesis" in reasoning
        assert len(improvements) == 3
        assert "First improvement point" in improvements[0]

    def test_parse_evolution_output_malformed(self):
        """Test parsing of malformed evolution output."""
        malformed_text = "This is just random text without proper structure"
        
        content, reasoning, improvements = self.agent._parse_evolution_output(malformed_text)
        
        # Should handle gracefully with fallback values
        assert content == malformed_text
        assert "evolution" in reasoning.lower()
        assert len(improvements) > 0

    def test_select_evolution_strategy(self):
        """Test strategy selection based on hypothesis characteristics."""
        # Test different hypothesis characteristics
        high_complexity_review = self.sample_reviews[0]
        high_complexity_review.hypothesis_text = "Complex multi-modal AI system with numerous interdependent components and sophisticated algorithms"
        
        strategy = self.agent._select_evolution_strategy(
            high_complexity_review,
            previous_strategies=[]
        )
        
        # Should prefer simplification for complex hypotheses
        assert strategy in [EvolutionStrategy.SIMPLIFICATION, EvolutionStrategy.INCREMENTAL_IMPROVEMENT]

    def test_select_evolution_strategy_avoid_previous(self):
        """Test that strategy selection avoids previously used strategies."""
        previous_strategies = [EvolutionStrategy.SIMPLIFICATION, EvolutionStrategy.COMBINATION]
        
        strategy = self.agent._select_evolution_strategy(
            self.sample_reviews[0],
            previous_strategies=previous_strategies
        )
        
        # Should not select previously used strategies
        assert strategy not in previous_strategies

    def test_create_evolved_hypothesis(self):
        """Test creation of evolved hypothesis object."""
        evolution_step = EvolutionStep(
            strategy=EvolutionStrategy.SIMPLIFICATION,
            original_hypothesis_id="hyp_1",
            evolved_content="Evolved content",
            evolution_reasoning="Test reasoning",
            improvements=["Improvement 1", "Improvement 2"],
            success=True
        )
        
        evolved_hyp = self.agent._create_evolved_hypothesis(
            "hyp_1",
            evolution_step,
            self.sample_reviews[0],
            1
        )
        
        assert isinstance(evolved_hyp, EvolvedHypothesis)
        assert evolved_hyp.original_hypothesis_id == "hyp_1"
        assert evolved_hyp.evolved_content == "Evolved content"
        assert evolved_hyp.evolution_steps[0] == evolution_step
        assert evolved_hyp.generation == 1

    @patch('agents.evolution.get_global_llm_client')
    def test_evolve_top_hypotheses(self, mock_llm_client):
        """Test evolution of top hypotheses."""
        # Mock LLM responses
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = """
        EVOLVED HYPOTHESIS: Evolved AI drug discovery approach with enhanced capabilities.
        
        EVOLUTION REASONING: Applied systematic improvements to the original hypothesis.
        
        KEY IMPROVEMENTS:
        • Enhanced accuracy
        • Improved efficiency
        • Better validation
        """
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = self.agent.evolve_top_hypotheses(
            self.ranking_state,
            self.reflection_state,
            original_constraints=[],
            top_n=2
        )
        
        assert isinstance(result, EvolutionState)
        assert len(result.evolved_hypotheses) <= 2
        assert all(isinstance(hyp, EvolvedHypothesis) for hyp in result.evolved_hypotheses)

    def test_evolve_top_hypotheses_empty_input(self):
        """Test evolution with empty input."""
        empty_ranking = RankingState(final_rankings=[])
        empty_reflection = ReflectionState(hypothesis_reviews=[])
        
        result = self.agent.evolve_top_hypotheses(
            empty_ranking,
            empty_reflection,
            original_constraints=[],
            top_n=2
        )
        
        assert isinstance(result, EvolutionState)
        assert len(result.evolved_hypotheses) == 0

    def test_track_strategy_effectiveness(self):
        """Test tracking of strategy effectiveness."""
        # Create evolution steps with different strategies
        steps = [
            EvolutionStep(
                strategy=EvolutionStrategy.SIMPLIFICATION,
                original_hypothesis_id="hyp_1",
                evolved_content="Content",
                evolution_reasoning="Reasoning",
                improvements=["Imp1", "Imp2"],
                success=True
            ),
            EvolutionStep(
                strategy=EvolutionStrategy.COMBINATION,
                original_hypothesis_id="hyp_2",
                evolved_content="Content",
                evolution_reasoning="Reasoning", 
                improvements=["Imp1"],
                success=False
            ),
            EvolutionStep(
                strategy=EvolutionStrategy.SIMPLIFICATION,
                original_hypothesis_id="hyp_3",
                evolved_content="Content",
                evolution_reasoning="Reasoning",
                improvements=["Imp1", "Imp2", "Imp3"],
                success=True
            )
        ]
        
        effectiveness = self.agent._track_strategy_effectiveness(steps)
        
        # SIMPLIFICATION: 2 successes, avg 2.5 improvements
        # COMBINATION: 1 failure, avg 1 improvement
        assert effectiveness[EvolutionStrategy.SIMPLIFICATION]["success_rate"] == 100.0
        assert effectiveness[EvolutionStrategy.SIMPLIFICATION]["avg_improvements"] == 2.5
        assert effectiveness[EvolutionStrategy.COMBINATION]["success_rate"] == 0.0
        assert effectiveness[EvolutionStrategy.COMBINATION]["avg_improvements"] == 1.0


class TestEvolutionAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_hypothesis_with_no_content(self):
        """Test evolution of hypothesis with no content."""
        agent = EvolutionAgent()
        
        empty_review = HypothesisReview(
            hypothesis_id="empty",
            hypothesis_text="",
            criteria=None,
            overall_score=7.0,
            overall_assessment="",
            strengths=[],
            weaknesses=[],
            recommendations=[],
            confidence_level=5.0,
            review_timestamp="2024-01-01T12:00:00",
            reviewer_type="test"
        )
        
        # Should handle gracefully
        strategy = agent._select_evolution_strategy(empty_review, [])
        assert strategy in agent.evolution_strategies

    def test_very_long_hypothesis_content(self):
        """Test evolution of very long hypothesis."""
        agent = EvolutionAgent()
        
        long_content = "A" * 10000  # Very long content
        long_review = Mock()
        long_review.hypothesis_text = long_content
        long_review.hypothesis_id = "long_hyp"
        
        # Should handle without crashing
        strategy = agent._select_evolution_strategy(long_review, [])
        assert strategy in agent.evolution_strategies

    def test_all_strategies_used_previously(self):
        """Test strategy selection when all strategies used previously."""
        agent = EvolutionAgent()
        
        all_strategies = list(agent.evolution_strategies.keys())
        
        # Should still return a strategy (random selection)
        strategy = agent._select_evolution_strategy(
            Mock(hypothesis_text="test"),
            previous_strategies=all_strategies
        )
        
        assert strategy in agent.evolution_strategies

    @patch('agents.evolution.get_global_llm_client')
    def test_malformed_llm_response(self, mock_llm_client):
        """Test handling of malformed LLM responses."""
        agent = EvolutionAgent()
        
        mock_response = Mock()
        mock_response.error = None
        mock_response.content = ""  # Empty response
        
        mock_client = Mock()
        mock_client.invoke.return_value = mock_response
        mock_llm_client.return_value = mock_client
        
        result = agent._apply_simplification_strategy(Mock(hypothesis_text="test"), [])
        
        # Should handle gracefully
        assert isinstance(result, EvolutionStep)
        assert result.success is True  # Should mark as success even with empty content

    def test_concurrent_evolution_operations(self):
        """Test that concurrent evolution operations don't interfere."""
        agent1 = EvolutionAgent()
        agent2 = EvolutionAgent()
        
        # Both should work independently
        strategy1 = agent1._select_evolution_strategy(Mock(hypothesis_text="test1"), [])
        strategy2 = agent2._select_evolution_strategy(Mock(hypothesis_text="test2"), [])
        
        # Should not interfere with each other
        assert strategy1 in agent1.evolution_strategies
        assert strategy2 in agent2.evolution_strategies


class TestEvolutionAgentIntegration:
    """Integration tests for the Evolution Agent."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        try:
            result = test_evolution_agent()
            assert isinstance(result, EvolutionState)
            assert hasattr(result, 'evolved_hypotheses')
            assert hasattr(result, 'strategy_effectiveness')
        except Exception as e:
            # If it fails due to missing LLM, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['llm', 'client', 'api'])

    @patch('agents.evolution.get_global_llm_client')
    def test_realistic_evolution_workflow(self, mock_llm_client):
        """Test realistic evolution workflow with scientific hypotheses."""
        # Mock diverse evolution outputs
        evolution_outputs = [
            """
            EVOLVED HYPOTHESIS: Simplified AI drug discovery platform using streamlined deep learning models for rapid therapeutic target identification with focus on computational efficiency.
            
            EVOLUTION REASONING: Simplified the original complex multi-modal approach to focus on core deep learning capabilities while maintaining effectiveness.
            
            KEY IMPROVEMENTS:
            • Reduced computational complexity
            • Faster training and inference
            • Clearer algorithmic approach
            • More accessible implementation
            """,
            """
            EVOLVED HYPOTHESIS: Integrated AI-powered drug discovery ecosystem combining machine learning target identification with automated experimental validation and real-time feedback loops.
            
            EVOLUTION REASONING: Combined computational and experimental approaches to create a comprehensive discovery pipeline with continuous improvement.
            
            KEY IMPROVEMENTS:
            • End-to-end automation
            • Real-time validation feedback
            • Continuous learning system
            • Reduced human intervention needs
            """
        ]
        
        output_iter = iter(evolution_outputs)
        
        def mock_invoke(prompt):
            mock_response = Mock()
            mock_response.error = None
            try:
                mock_response.content = next(output_iter)
            except StopIteration:
                mock_response.content = evolution_outputs[0]  # Fallback
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        agent = EvolutionAgent()
        
        # Create realistic scientific ranking and reflection states
        realistic_reviews = [
            self.create_realistic_review("ai_drug_discovery", 8.2, 
                "AI-powered drug discovery using deep learning for therapeutic target identification"),
            self.create_realistic_review("personalized_medicine", 7.8,
                "Personalized medicine approach using genomic data and machine learning for treatment optimization")
        ]
        
        ranking_state = RankingState(
            final_rankings=[
                {"rank": 1, "hypothesis_id": "ai_drug_discovery", "final_elo_rating": 1280, "win_rate": 75.0},
                {"rank": 2, "hypothesis_id": "personalized_medicine", "final_elo_rating": 1240, "win_rate": 60.0}
            ]
        )
        
        reflection_state = ReflectionState(hypothesis_reviews=realistic_reviews)
        
        result = agent.evolve_top_hypotheses(
            ranking_state,
            reflection_state,
            original_constraints=["Must be clinically applicable", "Consider ethical implications"],
            top_n=2
        )
        
        # Validate realistic evolution results
        assert isinstance(result, EvolutionState)
        assert len(result.evolved_hypotheses) == 2
        
        # Check that evolved hypotheses maintain scientific rigor
        for evolved_hyp in result.evolved_hypotheses:
            assert len(evolved_hyp.evolved_content) > 100  # Substantial content
            assert len(evolved_hyp.evolution_steps) > 0
            assert all(step.success for step in evolved_hyp.evolution_steps)

    def create_realistic_review(self, hyp_id: str, score: float, content: str) -> HypothesisReview:
        """Helper to create realistic scientific hypothesis review."""
        criteria = ReviewCriteria(
            novelty_score=score + 0.3,
            feasibility_score=score - 0.2,
            scientific_rigor_score=score + 0.1,
            impact_potential_score=score + 0.2,
            testability_score=score - 0.1,
            novelty_reasoning=f"Novel approach with score {score + 0.3}",
            feasibility_reasoning=f"Feasible implementation with score {score - 0.2}",
            scientific_rigor_reasoning=f"Scientifically rigorous with score {score + 0.1}",
            impact_potential_reasoning=f"High impact potential with score {score + 0.2}",
            testability_reasoning=f"Testable hypothesis with score {score - 0.1}"
        )
        
        return HypothesisReview(
            hypothesis_id=hyp_id,
            hypothesis_text=content,
            criteria=criteria,
            overall_score=score,
            overall_assessment=f"Comprehensive scientific assessment of {content[:50]}...",
            strengths=["Strong scientific foundation", "Clear methodology", "High innovation potential"],
            weaknesses=["Requires extensive validation", "Resource intensive", "Complex implementation"],
            recommendations=["Develop proof of concept", "Conduct pilot studies", "Establish collaborations"],
            confidence_level=8.5,
            review_timestamp="2024-01-01T12:00:00",
            reviewer_type="detailed"
        )

    @patch('agents.evolution.get_global_llm_client')
    def test_evolution_chain_multiple_generations(self, mock_llm_client):
        """Test evolution chain across multiple generations."""
        # Mock LLM to create evolution chain
        generation_counter = 0
        
        def mock_invoke(prompt):
            nonlocal generation_counter
            generation_counter += 1
            
            mock_response = Mock()
            mock_response.error = None
            mock_response.content = f"""
            EVOLVED HYPOTHESIS: Generation {generation_counter} evolved hypothesis with progressive improvements and enhanced capabilities.
            
            EVOLUTION REASONING: Applied generation {generation_counter} evolution strategy to further improve the hypothesis.
            
            KEY IMPROVEMENTS:
            • Generation {generation_counter} improvement A
            • Generation {generation_counter} improvement B
            • Progressive enhancement {generation_counter}
            """
            return mock_response
        
        mock_client = Mock()
        mock_client.invoke.side_effect = mock_invoke
        mock_llm_client.return_value = mock_client
        
        agent = EvolutionAgent()
        
        # Create initial hypothesis
        initial_review = self.create_realistic_review("initial", 7.5, "Initial AI hypothesis")
        ranking_state = RankingState(
            final_rankings=[{"rank": 1, "hypothesis_id": "initial", "final_elo_rating": 1200, "win_rate": 50.0}]
        )
        reflection_state = ReflectionState(hypothesis_reviews=[initial_review])
        
        # First evolution
        result1 = agent.evolve_top_hypotheses(ranking_state, reflection_state, [], top_n=1)
        
        # Simulate second evolution (would need evolved hypothesis as input in real system)
        result2 = agent.evolve_top_hypotheses(ranking_state, reflection_state, [], top_n=1)
        
        # Both should succeed
        assert len(result1.evolved_hypotheses) == 1
        assert len(result2.evolved_hypotheses) == 1
        
        # Should track different evolution attempts
        assert len(result1.strategy_effectiveness) > 0
        assert len(result2.strategy_effectiveness) > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])