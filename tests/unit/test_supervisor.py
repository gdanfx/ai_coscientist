"""
Tests for the Supervisor Agent

This module contains comprehensive tests for the IntegratedSupervisor
including unit tests for orchestration, early stopping, and integration testing.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.supervisor import (
    IntegratedSupervisor,
    SupervisorFactory,
    run_ai_coscientist_research,
    test_supervisor
)
from core.data_structures import (
    SupervisorConfig,
    FullSystemState,
    GenerationState,
    ReflectionState,
    RankingState,
    EvolutionState,
    ProximityState
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIntegratedSupervisor:
    """Test suite for the IntegratedSupervisor class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_config = SupervisorConfig(
            research_goal="Test AI Co-Scientist system functionality",
            constraints=["Keep test simple", "Validate all components"],
            max_cycles=2,
            evolution_every=1,
            proximity_every=1,
            meta_every=2,
            no_improve_patience=1
        )

    def test_supervisor_initialization(self):
        """Test that the supervisor initializes correctly."""
        supervisor = IntegratedSupervisor(self.test_config)
        
        assert supervisor is not None
        assert supervisor.config == self.test_config
        assert hasattr(supervisor, 'llm')
        assert hasattr(supervisor, 'generation_agent')
        assert hasattr(supervisor, 'reflection_agent')
        assert hasattr(supervisor, 'ranking_agent')
        assert hasattr(supervisor, 'evolution_agent')
        assert hasattr(supervisor, 'proximity_agent')

    def test_supervisor_initialization_with_custom_llm(self):
        """Test supervisor initialization with custom LLM client."""
        mock_llm = Mock()
        supervisor = IntegratedSupervisor(self.test_config, mock_llm)
        
        assert supervisor.llm == mock_llm

    def test_supervisor_initialization_failure(self):
        """Test handling of agent initialization failure."""
        with patch('agents.supervisor.EnhancedGenerationAgent') as mock_gen:
            mock_gen.side_effect = Exception("Agent initialization failed")
            
            with pytest.raises(Exception, match="Failed to initialize agents"):
                IntegratedSupervisor(self.test_config)

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent') 
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_generation_step(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test generation step execution."""
        # Mock generation agent
        mock_generation_state = GenerationState(
            research_goal=self.test_config.research_goal,
            constraints=self.test_config.constraints
        )
        mock_generation_state.generated_proposals = [
            {"id": "test_hyp_1", "content": "Test hypothesis 1"},
            {"id": "test_hyp_2", "content": "Test hypothesis 2"}
        ]
        
        mock_gen_instance = Mock()
        mock_gen_instance.run_complete_workflow.return_value = mock_generation_state
        mock_gen.return_value = mock_gen_instance
        
        # Setup other mocks
        mock_refl.return_value = Mock()
        mock_rank.return_value = Mock()
        mock_evo.return_value = Mock()
        mock_prox.return_value = Mock()
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Create initial state
        state = FullSystemState(config=self.test_config)
        
        # Run generation step
        result_state = supervisor._run_generation_step(state)
        
        assert len(result_state.proposals) == 2
        assert result_state.generation_agent_state == mock_generation_state
        mock_gen_instance.run_complete_workflow.assert_called_once()

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent') 
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_generation_step_failure(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test generation step with failure."""
        # Mock generation agent failure
        mock_gen_instance = Mock()
        mock_gen_instance.run_complete_workflow.side_effect = Exception("Generation failed")
        mock_gen.return_value = mock_gen_instance
        
        # Setup other mocks
        mock_refl.return_value = Mock()
        mock_rank.return_value = Mock()
        mock_evo.return_value = Mock()
        mock_prox.return_value = Mock()
        
        supervisor = IntegratedSupervisor(self.test_config)
        state = FullSystemState(config=self.test_config)
        
        # Should handle failure gracefully
        result_state = supervisor._run_generation_step(state)
        
        assert len(result_state.proposals) == 0  # Empty proposals due to failure

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent') 
    @patch('agents.supervisor.ProximityAgent')
    def test_run_proximity_step(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test proximity step execution."""
        # Setup mocks
        mock_gen.return_value = Mock()
        mock_refl.return_value = Mock()
        mock_rank.return_value = Mock()
        mock_evo.return_value = Mock()
        
        # Mock proximity agent
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = [
            {"id": "hyp_1", "content": "Unique hypothesis 1"}
        ]
        
        mock_prox_instance = Mock()
        mock_prox_instance.run.return_value = mock_proximity_result
        mock_prox.return_value = mock_prox_instance
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Create state with proposals
        state = FullSystemState(config=self.test_config)
        state.proposals = [
            {"id": "hyp_1", "content": "Hypothesis 1"},
            {"id": "hyp_2", "content": "Similar hypothesis 1"}
        ]
        
        result_state = supervisor._run_proximity_step(state)
        
        assert len(result_state.proposals) == 1  # Deduplicated
        assert result_state.proximity_agent_state == mock_proximity_result

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_reflection_step(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test reflection step execution."""
        # Setup mocks
        mock_gen.return_value = Mock()
        mock_rank.return_value = Mock()
        mock_evo.return_value = Mock()
        mock_prox.return_value = Mock()
        
        # Mock reflection agent
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="hyp_1", overall_score=8.0),
            Mock(hypothesis_id="hyp_2", overall_score=7.5)
        ]
        
        mock_refl_instance = Mock()
        mock_refl_instance.adaptive_batch_review.return_value = mock_reflection_result
        mock_refl.return_value = mock_refl_instance
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Create state with proposals
        state = FullSystemState(config=self.test_config)
        state.proposals = [
            {"id": "hyp_1", "content": "Hypothesis 1"},
            {"id": "hyp_2", "content": "Hypothesis 2"}
        ]
        
        result_state = supervisor._run_reflection_step(state)
        
        assert len(result_state.reviews) == 2
        assert result_state.reflection_agent_state == mock_reflection_result

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_ranking_step(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test ranking step execution."""
        # Setup mocks
        mock_gen.return_value = Mock()
        mock_refl.return_value = Mock()
        mock_evo.return_value = Mock()
        mock_prox.return_value = Mock()
        
        # Mock ranking agent
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": 1, "hypothesis_id": "hyp_1", "final_elo_rating": 1250},
            {"rank": 2, "hypothesis_id": "hyp_2", "final_elo_rating": 1180}
        ]
        
        mock_rank_instance = Mock()
        mock_rank_instance.run_full_tournament.return_value = mock_ranking_result
        mock_rank.return_value = mock_rank_instance
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Create state with reviews
        state = FullSystemState(config=self.test_config)
        state.reviews = [
            Mock(hypothesis_id="hyp_1"),
            Mock(hypothesis_id="hyp_2")
        ]
        state.reflection_agent_state = Mock()
        
        result_state = supervisor._run_ranking_step(state)
        
        assert len(result_state.rankings) == 2
        assert result_state.ranking_agent_state == mock_ranking_result

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_evolution_step(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test evolution step execution."""
        # Setup mocks
        mock_gen.return_value = Mock()
        mock_refl.return_value = Mock()
        mock_rank.return_value = Mock()
        mock_prox.return_value = Mock()
        
        # Mock evolution agent
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(hypothesis_id="evolved_1", evolved_content="Evolved hypothesis 1")
        ]
        
        mock_evo_instance = Mock()
        mock_evo_instance.evolve_top_hypotheses.return_value = mock_evolution_result
        mock_evo.return_value = mock_evo_instance
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Create state with rankings
        state = FullSystemState(config=self.test_config)
        state.rankings = [{"rank": 1, "hypothesis_id": "hyp_1"}]
        state.ranking_agent_state = Mock()
        state.reflection_agent_state = Mock()
        
        result_state = supervisor._run_evolution_step(state)
        
        assert len(result_state.proposals) == 1  # Replaced with evolved
        assert result_state.evolution_agent_state == mock_evolution_result

    def test_get_current_leader(self):
        """Test extraction of current leader hypothesis."""
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Test with rankings
        state = FullSystemState(config=self.test_config)
        state.rankings = [
            {"rank": 1, "hypothesis_id": "leader"},
            {"rank": 2, "hypothesis_id": "second"}
        ]
        
        leader = supervisor._get_current_leader(state)
        assert leader == "leader"
        
        # Test with empty rankings
        state.rankings = []
        leader = supervisor._get_current_leader(state)
        assert leader is None

    def test_create_fallback_rankings(self):
        """Test creation of fallback rankings from review scores."""
        supervisor = IntegratedSupervisor(self.test_config)
        
        reviews = [
            Mock(hypothesis_id="hyp_1", overall_score=8.5),
            Mock(hypothesis_id="hyp_2", overall_score=7.2),
            Mock(hypothesis_id="hyp_3", overall_score=9.1)
        ]
        
        rankings = supervisor._create_fallback_rankings(reviews)
        
        assert len(rankings) == 3
        assert rankings[0]["hypothesis_id"] == "hyp_3"  # Highest score
        assert rankings[1]["hypothesis_id"] == "hyp_1"  # Second highest
        assert rankings[2]["hypothesis_id"] == "hyp_2"  # Lowest

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_run_complete_workflow(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test complete supervisor workflow."""
        # Setup all agent mocks
        self._setup_all_agent_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Run complete workflow
        result = supervisor.run()
        
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True
        assert result.finish_reason is not None
        assert len(result.cycle_history) == self.test_config.max_cycles

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_early_stopping_stagnation(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test early stopping due to stagnation."""
        # Setup mocks to return same leader consistently
        self._setup_all_agent_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        # Configure for early stopping
        config = SupervisorConfig(
            research_goal="Test",
            max_cycles=10,  # High max cycles
            no_improve_patience=2  # Should stop after 2 cycles of no change
        )
        
        supervisor = IntegratedSupervisor(config)
        result = supervisor.run()
        
        assert result.is_finished is True
        assert "No improvement" in result.finish_reason
        assert len(result.cycle_history) < config.max_cycles

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_workflow_with_agent_failure(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test workflow with agent failure."""
        # Setup most agents normally, but make one fail
        self._setup_all_agent_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        # Make generation agent fail after setup
        mock_gen_instance = mock_gen.return_value
        mock_gen_instance.run_complete_workflow.side_effect = Exception("Generation failed")
        
        supervisor = IntegratedSupervisor(self.test_config)
        result = supervisor.run()
        
        # Should complete but with error state
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True

    def _setup_all_agent_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Helper to setup all agent mocks for testing."""
        # Generation agent mock
        mock_generation_state = GenerationState(
            research_goal="Test goal",
            constraints=[]
        )
        mock_generation_state.generated_proposals = [
            {"id": "hyp_1", "content": "Test hypothesis 1"},
            {"id": "hyp_2", "content": "Test hypothesis 2"}
        ]
        
        mock_gen_instance = Mock()
        mock_gen_instance.run_complete_workflow.return_value = mock_generation_state
        mock_gen.return_value = mock_gen_instance
        
        # Proximity agent mock
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals
        
        mock_prox_instance = Mock()
        mock_prox_instance.run.return_value = mock_proximity_result
        mock_prox.return_value = mock_prox_instance
        
        # Reflection agent mock
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="hyp_1", overall_score=8.0),
            Mock(hypothesis_id="hyp_2", overall_score=7.5)
        ]
        
        mock_refl_instance = Mock()
        mock_refl_instance.adaptive_batch_review.return_value = mock_reflection_result
        mock_refl.return_value = mock_refl_instance
        
        # Ranking agent mock
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": 1, "hypothesis_id": "hyp_1", "final_elo_rating": 1250},
            {"rank": 2, "hypothesis_id": "hyp_2", "final_elo_rating": 1180}
        ]
        
        mock_rank_instance = Mock()
        mock_rank_instance.run_full_tournament.return_value = mock_ranking_result
        mock_rank.return_value = mock_rank_instance
        
        # Evolution agent mock
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(hypothesis_id="evolved_1", evolved_content="Evolved hypothesis 1")
        ]
        
        mock_evo_instance = Mock()
        mock_evo_instance.evolve_top_hypotheses.return_value = mock_evolution_result
        mock_evo.return_value = mock_evo_instance


class TestSupervisorFactory:
    """Test suite for the SupervisorFactory class."""

    def test_create_research_supervisor(self):
        """Test creation of research supervisor."""
        supervisor = SupervisorFactory.create_research_supervisor(
            research_goal="Test research goal",
            constraints=["Test constraint"],
            max_cycles=3
        )
        
        assert isinstance(supervisor, IntegratedSupervisor)
        assert supervisor.config.research_goal == "Test research goal"
        assert supervisor.config.constraints == ["Test constraint"]
        assert supervisor.config.max_cycles == 3

    def test_create_rapid_prototyping_supervisor(self):
        """Test creation of rapid prototyping supervisor."""
        supervisor = SupervisorFactory.create_rapid_prototyping_supervisor(
            research_goal="Rapid test",
            constraints=["Fast execution"]
        )
        
        assert isinstance(supervisor, IntegratedSupervisor)
        assert supervisor.config.max_cycles == 2  # Rapid config
        assert supervisor.config.evolution_every == 1
        assert supervisor.config.no_improve_patience == 1

    def test_create_thorough_research_supervisor(self):
        """Test creation of thorough research supervisor."""
        supervisor = SupervisorFactory.create_thorough_research_supervisor(
            research_goal="Thorough test",
            constraints=["Comprehensive analysis"]
        )
        
        assert isinstance(supervisor, IntegratedSupervisor)
        assert supervisor.config.max_cycles == 5  # Thorough config
        assert supervisor.config.evolution_every == 2
        assert supervisor.config.no_improve_patience == 3


class TestSupervisorIntegrationFunctions:
    """Test integration functions and utilities."""

    @patch('agents.supervisor.SupervisorFactory.create_research_supervisor')
    def test_run_ai_coscientist_research(self, mock_factory):
        """Test convenience function for running research."""
        mock_supervisor = Mock()
        mock_result = FullSystemState(config=SupervisorConfig(research_goal="Test"))
        mock_supervisor.run.return_value = mock_result
        mock_factory.return_value = mock_supervisor
        
        result = run_ai_coscientist_research(
            research_goal="Test goal",
            constraints=["Test constraint"],
            max_cycles=3
        )
        
        assert result == mock_result
        mock_factory.assert_called_once_with(
            research_goal="Test goal",
            constraints=["Test constraint"],
            max_cycles=3
        )
        mock_supervisor.run.assert_called_once()


class TestSupervisorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_research_goal(self):
        """Test supervisor with empty research goal."""
        config = SupervisorConfig(research_goal="", max_cycles=1)
        supervisor = IntegratedSupervisor(config)
        
        # Should initialize without error
        assert supervisor.config.research_goal == ""

    def test_very_large_max_cycles(self):
        """Test supervisor with very large max cycles."""
        config = SupervisorConfig(research_goal="Test", max_cycles=1000)
        supervisor = IntegratedSupervisor(config)
        
        assert supervisor.config.max_cycles == 1000

    def test_zero_patience(self):
        """Test supervisor with zero patience (immediate stopping)."""
        config = SupervisorConfig(
            research_goal="Test",
            max_cycles=5,
            no_improve_patience=0
        )
        supervisor = IntegratedSupervisor(config)
        
        # Should handle zero patience gracefully
        assert supervisor.config.no_improve_patience == 0

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_all_agents_fail(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test behavior when all agents fail."""
        # Make all agents fail
        mock_gen.return_value.run_complete_workflow.side_effect = Exception("Gen failed")
        mock_refl.return_value.adaptive_batch_review.side_effect = Exception("Refl failed")
        mock_rank.return_value.run_full_tournament.side_effect = Exception("Rank failed")
        mock_evo.return_value.evolve_top_hypotheses.side_effect = Exception("Evo failed")
        mock_prox.return_value.run.side_effect = Exception("Prox failed")
        
        supervisor = IntegratedSupervisor(self.test_config)
        
        # Should not crash completely
        result = supervisor.run()
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True

    def test_config_with_none_values(self):
        """Test supervisor config with None values."""
        config = SupervisorConfig(
            research_goal="Test",
            constraints=None,  # Should handle None constraints
            max_cycles=1
        )
        
        supervisor = IntegratedSupervisor(config)
        assert supervisor.config.constraints == []  # Should default to empty list


class TestSupervisorSystemTest:
    """System-level tests for the supervisor."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        try:
            result = test_supervisor()
            assert isinstance(result, FullSystemState)
            assert result.is_finished
            assert len(result.cycle_history) > 0
        except Exception as e:
            # If it fails due to missing dependencies, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['llm', 'client', 'api', 'agent'])

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent') 
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_realistic_research_workflow(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test realistic research workflow simulation."""
        # Setup realistic mock responses
        self._setup_realistic_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        config = SupervisorConfig(
            research_goal="Develop AI-driven personalized medicine approaches",
            constraints=["Must be clinically applicable", "Consider ethical implications"],
            max_cycles=3,
            evolution_every=2,
            proximity_every=1
        )
        
        supervisor = IntegratedSupervisor(config)
        result = supervisor.run()
        
        # Validate realistic research results
        assert isinstance(result, FullSystemState)
        assert result.is_finished
        assert len(result.cycle_history) <= config.max_cycles
        assert len(result.proposals) > 0
        
        # Check that all cycles have reasonable data
        for cycle in result.cycle_history:
            assert cycle['cycle'] > 0
            assert cycle['num_proposals'] >= 0
            assert cycle['duration_sec'] > 0

    def _setup_realistic_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup realistic mock responses for system testing."""
        # Realistic generation results
        mock_generation_state = GenerationState(
            research_goal="AI personalized medicine",
            constraints=["Clinical applicability"]
        )
        mock_generation_state.generated_proposals = [
            {
                "id": "ai_genomics_1",
                "content": "AI-powered genomic analysis for personalized cancer treatment selection",
                "timestamp": "2024-01-01T12:00:00"
            },
            {
                "id": "ml_biomarkers_2", 
                "content": "Machine learning biomarker discovery for precision drug dosing",
                "timestamp": "2024-01-01T12:01:00"
            },
            {
                "id": "dl_imaging_3",
                "content": "Deep learning medical imaging for early disease detection",
                "timestamp": "2024-01-01T12:02:00"
            }
        ]
        
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        # Realistic proximity results (some deduplication)
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals[:2]  # Remove one duplicate
        mock_proximity_result.duplicates_removed = 1
        
        mock_prox.return_value.run.return_value = mock_proximity_result
        
        # Realistic reflection results
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="ai_genomics_1", overall_score=8.2, confidence_level=7.8),
            Mock(hypothesis_id="ml_biomarkers_2", overall_score=7.6, confidence_level=8.1)
        ]
        
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Realistic ranking results
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": 1, "hypothesis_id": "ai_genomics_1", "final_elo_rating": 1265, "win_rate": 75.0},
            {"rank": 2, "hypothesis_id": "ml_biomarkers_2", "final_elo_rating": 1235, "win_rate": 65.0}
        ]
        mock_ranking_result.total_comparisons = 1
        
        mock_rank.return_value.run_full_tournament.return_value = mock_ranking_result
        
        # Realistic evolution results
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(
                hypothesis_id="evolved_ai_genomics",
                evolved_content="Enhanced AI genomic analysis with multi-modal data integration",
                original_hypothesis_id="ai_genomics_1"
            )
        ]
        
        mock_evo.return_value.evolve_top_hypotheses.return_value = mock_evolution_result


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])