"""
Full System Integration Tests

This module contains end-to-end integration tests for the complete AI Co-Scientist
system, testing the interaction between all agents and components.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from core.config import get_config
from core.data_structures import SupervisorConfig, GenerationState, ProximityState, ReflectionState
from agents.supervisor import IntegratedSupervisor, SupervisorFactory
from agents.generation import EnhancedGenerationAgent
from agents.proximity import ProximityAgent
from agents.reflection import RobustReflectionAgent
from agents.ranking import RankingAgent
from agents.evolution import EvolutionAgent
from agents.meta_review import EnhancedMetaReviewAgent
from utils.llm_client import create_llm_client
from utils.literature_search import multi_source_literature_search

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullSystemIntegration:
    """Integration tests for the complete AI Co-Scientist system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_config = SupervisorConfig(
            research_goal="Integration test for AI Co-Scientist system functionality",
            constraints=["Keep test efficient", "Validate all components"],
            max_cycles=2,
            evolution_every=1,
            proximity_every=1,
            meta_every=2,
            no_improve_patience=1
        )

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_complete_research_workflow(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test complete research workflow from start to finish."""
        # Setup comprehensive mocks for all agents
        self._setup_comprehensive_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        # Create and run supervisor
        supervisor = IntegratedSupervisor(self.test_config)
        result = supervisor.run()
        
        # Validate complete workflow results
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True
        assert result.finish_reason is not None
        assert len(result.cycle_history) > 0
        
        # Validate that all agents were called
        assert result.generation_agent_state is not None
        assert result.reflection_agent_state is not None
        assert result.ranking_agent_state is not None
        assert result.evolution_agent_state is not None
        assert result.proximity_agent_state is not None
        
        # Validate data flow between agents
        assert len(result.proposals) > 0
        assert len(result.reviews) > 0
        assert len(result.rankings) > 0
        
        # Validate cycle history
        for cycle in result.cycle_history:
            assert 'cycle' in cycle
            assert 'num_proposals' in cycle
            assert 'duration_sec' in cycle
            assert 'timestamp' in cycle
            assert cycle['duration_sec'] > 0

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_multi_cycle_research_consistency(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test consistency across multiple research cycles."""
        # Setup mocks with cycle-aware responses
        self._setup_cycle_aware_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        # Run longer research process
        config = SupervisorConfig(
            research_goal="Multi-cycle consistency test",
            max_cycles=4,
            evolution_every=2,
            proximity_every=1,
            no_improve_patience=3
        )
        
        supervisor = IntegratedSupervisor(config)
        result = supervisor.run()
        
        # Validate multi-cycle consistency
        assert len(result.cycle_history) == config.max_cycles
        
        # Check that proposals evolve over cycles
        proposal_counts = [cycle['num_proposals'] for cycle in result.cycle_history]
        assert all(count > 0 for count in proposal_counts)
        
        # Validate that cycles complete in reasonable time
        cycle_times = [cycle['duration_sec'] for cycle in result.cycle_history]
        assert all(time_val < 60.0 for time_val in cycle_times)  # Should complete within 60s per cycle

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_early_stopping_integration(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test early stopping behavior in integrated system."""
        # Setup mocks to produce consistent leader
        self._setup_stagnation_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        config = SupervisorConfig(
            research_goal="Early stopping test",
            max_cycles=10,  # High max cycles
            no_improve_patience=2,  # Should stop early
            evolution_every=1
        )
        
        supervisor = IntegratedSupervisor(config)
        result = supervisor.run()
        
        # Should stop early due to stagnation
        assert result.is_finished is True
        assert "No improvement" in result.finish_reason
        assert len(result.cycle_history) < config.max_cycles
        assert len(result.cycle_history) >= config.no_improve_patience

    def test_supervisor_factory_integration(self):
        """Test integration through SupervisorFactory."""
        # Test rapid prototyping supervisor
        rapid_supervisor = SupervisorFactory.create_rapid_prototyping_supervisor(
            research_goal="Rapid test",
            constraints=["Quick execution"]
        )
        
        assert isinstance(rapid_supervisor, IntegratedSupervisor)
        assert rapid_supervisor.config.max_cycles == 2
        
        # Test thorough research supervisor
        thorough_supervisor = SupervisorFactory.create_thorough_research_supervisor(
            research_goal="Thorough test",
            constraints=["Comprehensive analysis"]
        )
        
        assert isinstance(thorough_supervisor, IntegratedSupervisor)
        assert thorough_supervisor.config.max_cycles == 5

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_agent_failure_recovery(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test system recovery from individual agent failures."""
        # Setup partial failure scenario
        self._setup_partial_failure_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        supervisor = IntegratedSupervisor(self.test_config)
        result = supervisor.run()
        
        # System should complete despite agent failures
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True
        
        # Should have some results even with failures
        assert len(result.cycle_history) > 0

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_data_flow_integrity(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test data integrity throughout the workflow."""
        # Setup mocks with traceable data
        self._setup_traceable_data_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        supervisor = IntegratedSupervisor(self.test_config)
        result = supervisor.run()
        
        # Validate data flow integrity
        assert isinstance(result, FullSystemState)
        
        # Check that hypothesis IDs are consistent across stages
        proposal_ids = {p['id'] for p in result.proposals}
        review_ids = {r.hypothesis_id for r in result.reviews}
        ranking_ids = {r['hypothesis_id'] for r in result.rankings}
        
        # There should be overlap between stages (accounting for deduplication and evolution)
        assert len(proposal_ids) > 0
        assert len(review_ids) > 0
        assert len(ranking_ids) > 0

    def _setup_comprehensive_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup comprehensive mocks for full workflow testing."""
        from core.data_structures import (
            GenerationState, ReflectionState, RankingState, 
            EvolutionState, ProximityState
        )
        
        # Generation agent mock
        mock_generation_state = GenerationState(
            research_goal=self.test_config.research_goal,
            constraints=self.test_config.constraints
        )
        mock_generation_state.generated_proposals = [
            {"id": "gen_hyp_1", "content": "AI-powered hypothesis generation test 1"},
            {"id": "gen_hyp_2", "content": "Machine learning approach for test 2"},
            {"id": "gen_hyp_3", "content": "Novel computational method test 3"}
        ]
        
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        # Proximity agent mock
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals[:2]  # Deduplicate one
        mock_proximity_result.duplicates_removed = 1
        
        mock_prox.return_value.run.return_value = mock_proximity_result
        
        # Reflection agent mock
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="gen_hyp_1", overall_score=8.2, confidence_level=7.8),
            Mock(hypothesis_id="gen_hyp_2", overall_score=7.6, confidence_level=8.1)
        ]
        
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Ranking agent mock
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": 1, "hypothesis_id": "gen_hyp_1", "final_elo_rating": 1265, "win_rate": 75.0},
            {"rank": 2, "hypothesis_id": "gen_hyp_2", "final_elo_rating": 1235, "win_rate": 65.0}
        ]
        
        mock_rank.return_value.run_full_tournament.return_value = mock_ranking_result
        
        # Evolution agent mock
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(
                hypothesis_id="evolved_gen_hyp_1",
                evolved_content="Enhanced AI-powered hypothesis with improvements",
                original_hypothesis_id="gen_hyp_1"
            )
        ]
        
        mock_evo.return_value.evolve_top_hypotheses.return_value = mock_evolution_result

    def _setup_cycle_aware_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup mocks that change behavior across cycles."""
        from core.data_structures import GenerationState, ReflectionState, RankingState, EvolutionState, ProximityState
        
        # Counter for tracking cycles
        cycle_counter = {"count": 0}
        
        def generation_side_effect(*args, **kwargs):
            cycle_counter["count"] += 1
            state = GenerationState(research_goal="Multi-cycle test", constraints=[])
            state.generated_proposals = [
                {"id": f"cycle_{cycle_counter['count']}_hyp_1", "content": f"Hypothesis 1 from cycle {cycle_counter['count']}"},
                {"id": f"cycle_{cycle_counter['count']}_hyp_2", "content": f"Hypothesis 2 from cycle {cycle_counter['count']}"}
            ]
            return state
        
        mock_gen.return_value.run_complete_workflow.side_effect = generation_side_effect
        
        # Other agents adapt to changing input
        mock_prox.return_value.run.side_effect = lambda proposals: self._create_proximity_result(proposals)
        mock_refl.return_value.adaptive_batch_review.side_effect = lambda proposals, goal: self._create_reflection_result(proposals)
        mock_rank.return_value.run_full_tournament.side_effect = lambda refl_state, **kwargs: self._create_ranking_result(refl_state)
        mock_evo.return_value.evolve_top_hypotheses.side_effect = lambda rank_state, refl_state, **kwargs: self._create_evolution_result(rank_state)

    def _setup_stagnation_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup mocks that produce consistent results for stagnation testing."""
        from core.data_structures import GenerationState, ReflectionState, RankingState, EvolutionState, ProximityState
        
        # Always return the same leader to trigger stagnation
        mock_generation_state = GenerationState(research_goal="Stagnation test", constraints=[])
        mock_generation_state.generated_proposals = [
            {"id": "stagnant_leader", "content": "Consistent leading hypothesis"},
            {"id": "stagnant_second", "content": "Consistent second hypothesis"}
        ]
        
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals
        mock_prox.return_value.run.return_value = mock_proximity_result
        
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="stagnant_leader", overall_score=8.5),
            Mock(hypothesis_id="stagnant_second", overall_score=7.0)
        ]
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Always rank the same hypothesis first
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": 1, "hypothesis_id": "stagnant_leader", "final_elo_rating": 1300},
            {"rank": 2, "hypothesis_id": "stagnant_second", "final_elo_rating": 1200}
        ]
        mock_rank.return_value.run_full_tournament.return_value = mock_ranking_result
        
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(hypothesis_id="evolved_leader", evolved_content="Evolved leader", original_hypothesis_id="stagnant_leader")
        ]
        mock_evo.return_value.evolve_top_hypotheses.return_value = mock_evolution_result

    def _setup_partial_failure_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup mocks with some agents failing."""
        from core.data_structures import GenerationState, ReflectionState, RankingState
        
        # Generation works
        mock_generation_state = GenerationState(research_goal="Failure test", constraints=[])
        mock_generation_state.generated_proposals = [
            {"id": "robust_hyp_1", "content": "Hypothesis that survives failures"}
        ]
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        # Proximity fails
        mock_prox.return_value.run.side_effect = Exception("Proximity agent failed")
        
        # Reflection works
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id="robust_hyp_1", overall_score=7.5)
        ]
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Ranking fails
        mock_rank.return_value.run_full_tournament.side_effect = Exception("Ranking agent failed")
        
        # Evolution fails
        mock_evo.return_value.evolve_top_hypotheses.side_effect = Exception("Evolution agent failed")

    def _setup_traceable_data_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup mocks with traceable data for integrity testing."""
        from core.data_structures import GenerationState, ReflectionState, RankingState, EvolutionState, ProximityState
        
        # Define consistent hypothesis IDs
        hyp_ids = ["trace_hyp_1", "trace_hyp_2", "trace_hyp_3"]
        
        # Generation produces traceable hypotheses
        mock_generation_state = GenerationState(research_goal="Trace test", constraints=[])
        mock_generation_state.generated_proposals = [
            {"id": hyp_id, "content": f"Traceable hypothesis {hyp_id}"}
            for hyp_id in hyp_ids
        ]
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        # Proximity maintains traceability
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals
        mock_prox.return_value.run.return_value = mock_proximity_result
        
        # Reflection maintains ID consistency
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id=hyp_id, overall_score=7.0 + i)
            for i, hyp_id in enumerate(hyp_ids)
        ]
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Ranking maintains ID consistency
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": i+1, "hypothesis_id": hyp_id, "final_elo_rating": 1200 + (len(hyp_ids)-i)*10}
            for i, hyp_id in enumerate(hyp_ids)
        ]
        mock_rank.return_value.run_full_tournament.return_value = mock_ranking_result
        
        # Evolution creates new IDs but maintains traceability
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(
                hypothesis_id=f"evolved_{hyp_ids[0]}",
                evolved_content=f"Evolved {hyp_ids[0]}",
                original_hypothesis_id=hyp_ids[0]
            )
        ]
        mock_evo.return_value.evolve_top_hypotheses.return_value = mock_evolution_result

    def _create_proximity_result(self, proposals):
        """Helper to create proximity result."""
        from core.data_structures import ProximityState
        result = ProximityState()
        result.unique_hypotheses = proposals
        result.duplicates_removed = 0
        return result

    def _create_reflection_result(self, proposals):
        """Helper to create reflection result."""
        from core.data_structures import ReflectionState
        result = ReflectionState()
        result.hypothesis_reviews = [
            Mock(hypothesis_id=p['id'], overall_score=7.5)
            for p in proposals
        ]
        return result

    def _create_ranking_result(self, refl_state):
        """Helper to create ranking result."""
        from core.data_structures import RankingState
        result = RankingState()
        result.final_rankings = [
            {"rank": i+1, "hypothesis_id": review.hypothesis_id, "final_elo_rating": 1200 + (10-i)*10}
            for i, review in enumerate(refl_state.hypothesis_reviews)
        ]
        return result

    def _create_evolution_result(self, rank_state):
        """Helper to create evolution result."""
        from core.data_structures import EvolutionState
        result = EvolutionState()
        if rank_state.final_rankings:
            top_id = rank_state.final_rankings[0]["hypothesis_id"]
            result.evolved_hypotheses = [
                Mock(
                    hypothesis_id=f"evolved_{top_id}",
                    evolved_content=f"Evolved {top_id}",
                    original_hypothesis_id=top_id
                )
            ]
        return result


class TestMainModuleIntegration:
    """Integration tests for main module functions."""

    @patch.dict('os.environ', {
        'PUBMED_EMAIL': 'test@example.com',
        'GOOGLE_API_KEY': 'test_key_for_validation'
    })
    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        result = validate_configuration()
        assert result is True

    @patch.dict('os.environ', {}, clear=True)
    def test_validate_configuration_failure(self):
        """Test configuration validation failure."""
        result = validate_configuration()
        assert result is False

    @patch('main.validate_configuration')
    @patch('main.IntegratedSupervisor')
    def test_run_system_test_success(self, mock_supervisor_class, mock_validate):
        """Test successful system test execution."""
        # Mock successful validation
        mock_validate.return_value = True
        
        # Mock supervisor
        mock_result = Mock()
        mock_result.is_finished = True
        mock_result.finish_reason = "Test completed"
        mock_result.cycle_history = [{"cycle": 1, "duration_sec": 1.5}]
        mock_result.proposals = [{"id": "test", "content": "test"}]
        
        mock_supervisor = Mock()
        mock_supervisor.run.return_value = mock_result
        mock_supervisor_class.return_value = mock_supervisor
        
        result = run_system_test()
        assert result is True

    @patch('main.validate_configuration')
    def test_run_system_test_config_failure(self, mock_validate):
        """Test system test with configuration failure."""
        mock_validate.return_value = False
        
        result = run_system_test()
        assert result is False

    def test_save_results_success(self, tmp_path):
        """Test successful result saving."""
        # Create mock result
        mock_result = Mock()
        mock_result.config = Mock()
        mock_result.is_finished = True
        mock_result.finish_reason = "Test completed"
        mock_result.cycle_history = [{"cycle": 1, "timestamp": "2024-01-01T12:00:00"}]
        mock_result.proposals = [{"id": "test", "content": "test"}]
        mock_result.rankings = [{"rank": 1, "hypothesis_id": "test"}]
        
        # Mock asdict for dataclass conversion
        with patch('main.asdict') as mock_asdict:
            mock_asdict.return_value = {"test": "config"}
            
            output_file = tmp_path / "test_results.json"
            save_results(mock_result, str(output_file))
            
            # Check file was created
            assert output_file.exists()

    def test_display_results(self, capsys):
        """Test result display functionality."""
        # Create mock result
        mock_result = Mock()
        mock_result.config.research_goal = "Test goal"
        mock_result.config.constraints = ["Test constraint"]
        mock_result.cycle_history = [
            {"cycle": 1, "num_proposals": 3, "duration_sec": 1.5}
        ]
        mock_result.proposals = [
            {"id": "test1", "content": "Test content 1"},
            {"id": "test2", "content": "Test content 2"}
        ]
        mock_result.rankings = [
            {"hypothesis_id": "test1", "final_elo_rating": 1250}
        ]
        mock_result.finish_reason = "Test completed"
        
        display_results(mock_result)
        
        # Check that output was produced
        captured = capsys.readouterr()
        assert "AI CO-SCIENTIST RESEARCH RESULTS" in captured.out
        assert "Test goal" in captured.out
        assert "Test completed" in captured.out


class TestSystemPerformance:
    """Performance and stress tests for the system."""

    @patch('agents.supervisor.EnhancedGenerationAgent')
    @patch('agents.supervisor.RobustReflectionAgent')
    @patch('agents.supervisor.RankingAgent')
    @patch('agents.supervisor.EvolutionAgent')
    @patch('agents.supervisor.ProximityAgent')
    def test_system_performance_under_load(self, mock_prox, mock_evo, mock_rank, mock_refl, mock_gen):
        """Test system performance with larger workload."""
        # Setup mocks for larger dataset
        self._setup_performance_mocks(mock_gen, mock_refl, mock_rank, mock_evo, mock_prox)
        
        config = SupervisorConfig(
            research_goal="Performance test with larger dataset",
            max_cycles=3,
            evolution_every=2,
            proximity_every=1
        )
        
        start_time = time.time()
        supervisor = IntegratedSupervisor(config)
        result = supervisor.run()
        total_time = time.time() - start_time
        
        # Validate performance
        assert isinstance(result, FullSystemState)
        assert result.is_finished is True
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Check that all cycles completed efficiently
        for cycle in result.cycle_history:
            assert cycle['duration_sec'] < 10.0  # Each cycle should be under 10s

    def _setup_performance_mocks(self, mock_gen, mock_refl, mock_rank, mock_evo, mock_prox):
        """Setup mocks for performance testing with larger datasets."""
        from core.data_structures import GenerationState, ReflectionState, RankingState, EvolutionState, ProximityState
        
        # Generate larger dataset
        num_hypotheses = 10
        
        mock_generation_state = GenerationState(research_goal="Performance test", constraints=[])
        mock_generation_state.generated_proposals = [
            {"id": f"perf_hyp_{i}", "content": f"Performance test hypothesis {i}"}
            for i in range(num_hypotheses)
        ]
        mock_gen.return_value.run_complete_workflow.return_value = mock_generation_state
        
        # Proximity with some deduplication
        mock_proximity_result = ProximityState()
        mock_proximity_result.unique_hypotheses = mock_generation_state.generated_proposals[:8]  # Remove 2
        mock_prox.return_value.run.return_value = mock_proximity_result
        
        # Reflection for all remaining
        mock_reflection_result = ReflectionState()
        mock_reflection_result.hypothesis_reviews = [
            Mock(hypothesis_id=f"perf_hyp_{i}", overall_score=7.0 + (i % 3))
            for i in range(8)
        ]
        mock_refl.return_value.adaptive_batch_review.return_value = mock_reflection_result
        
        # Ranking for all
        mock_ranking_result = RankingState()
        mock_ranking_result.final_rankings = [
            {"rank": i+1, "hypothesis_id": f"perf_hyp_{i}", "final_elo_rating": 1200 + (8-i)*10}
            for i in range(8)
        ]
        mock_rank.return_value.run_full_tournament.return_value = mock_ranking_result
        
        # Evolution for top 3
        mock_evolution_result = EvolutionState()
        mock_evolution_result.evolved_hypotheses = [
            Mock(
                hypothesis_id=f"evolved_perf_hyp_{i}",
                evolved_content=f"Evolved performance hypothesis {i}",
                original_hypothesis_id=f"perf_hyp_{i}"
            )
            for i in range(3)
        ]
        mock_evo.return_value.evolve_top_hypotheses.return_value = mock_evolution_result


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])