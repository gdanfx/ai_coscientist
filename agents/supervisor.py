"""
Supervisor Agent - Central Research Orchestrator

This module implements the IntegratedSupervisor that orchestrates the complete
AI Co-Scientist workflow through multiple research cycles.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.data_structures import (
    SupervisorConfig, SupervisorState, CycleStats, FullSystemState,
    GenerationState, ReflectionState, RankingState, EvolutionState,
    ProximityState, create_timestamp
)
from core.config import get_config

# Import all agents
from agents.generation import EnhancedGenerationAgent
from agents.reflection import RobustReflectionAgent
from agents.ranking import RankingAgent
from agents.evolution import EvolutionAgent
from agents.proximity import ProximityAgent
from agents.meta_review import EnhancedMetaReviewAgent

# Import utilities
from utils.llm_client import get_global_llm_client

logger = logging.getLogger(__name__)


class IntegratedSupervisor:
    """
    Central orchestrator for the AI Co-Scientist research workflow.
    
    Manages the complete research cycle:
    1. Generation: Literature search and hypothesis generation
    2. Proximity: Deduplication and diversity maintenance  
    3. Reflection: Automated peer review
    4. Ranking: Competitive hypothesis ranking
    5. Evolution: Hypothesis improvement and innovation
    6. Meta-analysis: System-level feedback and optimization
    """
    
    def __init__(self, config: SupervisorConfig, llm_client=None):
        """
        Initialize the Supervisor with configuration.
        
        Args:
            config: SupervisorConfig with research parameters
            llm_client: Optional LLM client (uses global if not provided)
        """
        self.config = config
        self.llm = llm_client or get_global_llm_client()
        
        # Initialize all agents
        self._initialize_agents()
        
        logger.info(f"IntegratedSupervisor initialized for: {config.research_goal}")
    
    def _initialize_agents(self):
        """Initialize all agent instances."""
        try:
            self.generation_agent = EnhancedGenerationAgent(self.llm)
            self.reflection_agent = RobustReflectionAgent(self.llm)
            self.ranking_agent = RankingAgent(self.llm)
            self.evolution_agent = EvolutionAgent(self.llm)
            self.proximity_agent = ProximityAgent()
            self.meta_review_agent = EnhancedMetaReviewAgent()
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def run(self) -> FullSystemState:
        """
        Execute the complete multi-cycle research process.
        
        Returns:
            FullSystemState with complete research results
        """
        logger.info(f"Starting AI Co-Scientist research: '{self.config.research_goal}'")
        logger.info("=" * 60)
        
        # Initialize system state
        state = FullSystemState(config=self.config)
        best_leader_id = None
        stagnation_counter = 0
        
        try:
            for cycle in range(1, self.config.max_cycles + 1):
                logger.info(f"\n🔄 CYCLE {cycle} / {self.config.max_cycles}")
                start_time = time.time()
                
                # Run cycle steps
                state = self._run_cycle(state, cycle)
                
                # Update cycle history
                current_leader_id = self._get_current_leader(state)
                cycle_duration = time.time() - start_time
                
                state.cycle_history.append({
                    'cycle': cycle,
                    'num_proposals': len(state.proposals),
                    'best_hypothesis_id': current_leader_id,
                    'duration_sec': cycle_duration,
                    'timestamp': create_timestamp()
                })
                
                logger.info(f"  -> Cycle {cycle} complete. Best hypothesis: {current_leader_id}")
                
                # Check for stagnation (early stopping)
                if current_leader_id and current_leader_id == best_leader_id:
                    stagnation_counter += 1
                else:
                    best_leader_id = current_leader_id
                    stagnation_counter = 0
                
                if stagnation_counter >= self.config.no_improve_patience:
                    state.is_finished = True
                    state.finish_reason = f"No improvement in top hypothesis for {stagnation_counter} cycles"
                    logger.info(f"Early stopping: {state.finish_reason}")
                    break
            
            # Mark as finished if we completed all cycles
            if not state.is_finished:
                state.is_finished = True
                state.finish_reason = "Reached maximum cycles"
            
            logger.info("\n" + "=" * 60)
            logger.info(f"✅ AI Co-Scientist research finished: {state.finish_reason}")
            
            return state
            
        except Exception as e:
            logger.error(f"Research process failed: {e}")
            state.is_finished = True
            state.finish_reason = f"Error: {str(e)}"
            return state
    
    def _run_cycle(self, state: FullSystemState, cycle: int) -> FullSystemState:
        """Run a single research cycle with all agents."""
        
        # 1. Generation (always run)
        state = self._run_generation_step(state)
        
        # 2. Proximity (deduplicate if enabled for this cycle)
        if cycle % self.config.proximity_every == 0:
            state = self._run_proximity_step(state)
        
        # 3. Reflection (always run after generation)
        state = self._run_reflection_step(state)
        
        # 4. Ranking (always run after reflection)
        state = self._run_ranking_step(state)
        
        # 5. Evolution (if enabled for this cycle)
        if cycle % self.config.evolution_every == 0 and len(state.rankings) > 0:
            state = self._run_evolution_step(state)
            
            # After evolution, re-evaluate the evolved hypotheses
            logger.info("  -> Post-Evolution Re-evaluation:")
            state = self._run_reflection_step(state)
            state = self._run_ranking_step(state)
        
        # 6. Meta-review (if enabled for this cycle) 
        if cycle % self.config.meta_every == 0:
            state = self._run_meta_review_step(state)
        
        return state
    
    def _run_generation_step(self, state: FullSystemState) -> FullSystemState:
        """Run the Generation Agent."""
        logger.info("  -> Running Generation Agent...")
        
        try:
            generation_state = self.generation_agent.run_complete_workflow(
                research_goal=self.config.research_goal,
                constraints=self.config.constraints
            )
            
            # Add new proposals to state
            new_proposals = generation_state.generated_proposals
            state.proposals.extend(new_proposals)
            state.generation_agent_state = generation_state
            
            logger.info(f"     + Added {len(new_proposals)} new hypotheses")
            
        except Exception as e:
            logger.error(f"Generation agent failed: {e}")
            # Continue with empty proposals rather than failing completely
            state.proposals = []
        
        return state
    
    def _run_proximity_step(self, state: FullSystemState) -> FullSystemState:
        """Run the Proximity Agent for deduplication."""
        logger.info("  -> Running Proximity Agent...")
        
        initial_count = len(state.proposals)
        
        if initial_count < 2:
            logger.info("     - Skipping, not enough hypotheses to compare")
            return state
        
        try:
            proximity_result = self.proximity_agent.run(state.proposals)
            
            # Update proposals with deduplicated set
            state.proposals = proximity_result.unique_hypotheses
            state.proximity_agent_state = proximity_result
            
            removed_count = initial_count - len(state.proposals)
            logger.info(f"     - Deduplicated proposals. Removed: {removed_count}, Kept: {len(state.proposals)}")
            
        except Exception as e:
            logger.error(f"Proximity agent failed: {e}")
            # Continue with original proposals
        
        return state
    
    def _run_reflection_step(self, state: FullSystemState) -> FullSystemState:
        """Run the Reflection Agent for peer review."""
        logger.info("  -> Running Reflection Agent...")
        
        if not state.proposals:
            logger.warning("     ! No proposals to reflect on")
            return state
        
        try:
            reflection_state = self.reflection_agent.adaptive_batch_review(
                hypotheses=state.proposals,
                research_goal=self.config.research_goal
            )
            
            state.reviews = reflection_state.hypothesis_reviews
            state.reflection_agent_state = reflection_state
            
            logger.info(f"     - Reviewed {len(state.reviews)} hypotheses")
            
        except Exception as e:
            logger.error(f"Reflection agent failed: {e}")
            state.reviews = []
        
        return state
    
    def _run_ranking_step(self, state: FullSystemState) -> FullSystemState:
        """Run the Ranking Agent for competitive ranking."""
        logger.info("  -> Running Ranking Agent...")
        
        if len(state.reviews) < 2:
            logger.info("     - Skipping, not enough hypotheses to rank")
            
            # Create minimal ranking state for single hypothesis
            if state.reviews:
                state.rankings = [{'rank': 1, 'hypothesis_id': state.reviews[0].hypothesis_id}]
                state.ranking_agent_state = RankingState(final_rankings=state.rankings)
            else:
                state.rankings = []
                state.ranking_agent_state = RankingState()
            
            return state
        
        try:
            ranking_state = self.ranking_agent.run_full_tournament(
                reflection_state=state.reflection_agent_state,
                num_rounds=2  # Shorter tournaments for faster cycles
            )
            
            state.rankings = ranking_state.final_rankings
            state.ranking_agent_state = ranking_state
            
            logger.info(f"     - Ranked {len(state.rankings)} hypotheses")
            
        except Exception as e:
            logger.error(f"Ranking agent failed: {e}")
            # Create fallback rankings based on reflection scores
            state.rankings = self._create_fallback_rankings(state.reviews)
        
        return state
    
    def _run_evolution_step(self, state: FullSystemState) -> FullSystemState:
        """Run the Evolution Agent for hypothesis improvement."""
        logger.info("  -> Running Evolution Agent...")
        
        if not state.rankings or not state.ranking_agent_state:
            logger.warning("     ! No ranked hypotheses to evolve")
            return state
        
        try:
            evolution_state = self.evolution_agent.evolve_top_hypotheses(
                ranking_state=state.ranking_agent_state,
                reflection_state=state.reflection_agent_state,
                original_constraints=self.config.constraints,
                top_n=min(3, len(state.rankings))  # Evolve top 3 or fewer
            )
            
            # Replace proposals with evolved hypotheses
            evolved_proposals = [
                {"id": hyp.hypothesis_id, "content": hyp.evolved_content}
                for hyp in evolution_state.evolved_hypotheses
            ]
            
            state.proposals = evolved_proposals
            state.evolution_agent_state = evolution_state
            
            logger.info(f"     + Evolved top hypotheses into {len(evolved_proposals)} new proposals")
            
        except Exception as e:
            logger.error(f"Evolution agent failed: {e}")
            # Continue with existing proposals
        
        return state
    
    def _run_meta_review_step(self, state: FullSystemState) -> FullSystemState:
        """Run meta-review analysis to detect patterns and biases."""
        logger.info("  -> Running Meta-review Agent...")
        
        try:
            if not state.reflection_agent_state or not state.reflection_agent_state.hypothesis_reviews:
                logger.warning("     - No reflection data available for meta-review")
                return state
            
            # Convert HypothesisReview objects to dict format for meta-review
            review_dicts = []
            for review in state.reflection_agent_state.hypothesis_reviews:
                review_dict = {
                    'hypothesis_id': review.hypothesis_id,
                    'hypothesis_content': getattr(review, 'hypothesis_text', ''),
                    'review': {
                        'novelty': review.criteria.novelty_score,
                        'feasibility': review.criteria.feasibility_score,
                        'rigor': review.criteria.scientific_rigor_score,
                        'impact': review.criteria.impact_potential_score,
                        'testability': review.criteria.testability_score,
                        'overall': review.overall_score
                    }
                }
                review_dicts.append(review_dict)
            
            # Create a mock reflection state with dict format
            class MockReflectionState:
                def __init__(self, reviews):
                    self.hypothesis_reviews = reviews
            
            mock_state = MockReflectionState(review_dicts)
            
            # Run meta-review analysis
            meta_review_state = self.meta_review_agent.run(mock_state)
            state.meta_review_agent_state = meta_review_state
            
            # Log insights
            logger.info(f"     - Detected {len(meta_review_state.pattern_insights)} evaluation patterns")
            logger.info(f"     - Found {len(meta_review_state.criterion_correlations)} criterion correlations")
            logger.info(f"     - Analysis quality: {meta_review_state.analysis_quality}")
            
            # Log key recommendations (full text for important insights)
            if meta_review_state.actionable_for_generation:
                logger.info(f"     - Generation recommendations: {meta_review_state.actionable_for_generation}")
            if meta_review_state.actionable_for_reflection:
                logger.info(f"     - Reflection recommendations: {meta_review_state.actionable_for_reflection}")
            
            return state
            
        except Exception as e:
            logger.error(f"Meta-review analysis failed: {e}")
            return state
    
    def _get_current_leader(self, state: FullSystemState) -> Optional[str]:
        """Get the current leading hypothesis ID."""
        if state.rankings:
            return state.rankings[0].get('hypothesis_id')
        return None
    
    def _create_fallback_rankings(self, reviews: List) -> List[Dict[str, Any]]:
        """Create fallback rankings based on reflection scores."""
        if not reviews:
            return []
        
        # Sort by overall score
        sorted_reviews = sorted(reviews, key=lambda r: r.overall_score, reverse=True)
        
        rankings = []
        for rank, review in enumerate(sorted_reviews, 1):
            rankings.append({
                'rank': rank,
                'hypothesis_id': review.hypothesis_id,
                'final_elo_rating': 1200 + (review.overall_score - 5) * 20,  # Convert score to Elo-like rating
                'win_rate': review.overall_score * 10  # Convert to percentage
            })
        
        return rankings


class SupervisorFactory:
    """Factory for creating supervisor instances with different configurations."""
    
    @staticmethod
    def create_research_supervisor(research_goal: str, 
                                 constraints: List[str] = None,
                                 max_cycles: int = 3,
                                 **kwargs) -> IntegratedSupervisor:
        """Create a supervisor for general research tasks."""
        config = SupervisorConfig(
            research_goal=research_goal,
            constraints=constraints or [],
            max_cycles=max_cycles,
            **kwargs
        )
        return IntegratedSupervisor(config)
    
    @staticmethod
    def create_rapid_prototyping_supervisor(research_goal: str,
                                          constraints: List[str] = None) -> IntegratedSupervisor:
        """Create a supervisor optimized for rapid prototyping (2-3 cycles)."""
        config = SupervisorConfig(
            research_goal=research_goal,
            constraints=constraints or [],
            max_cycles=2,
            evolution_every=1,
            proximity_every=1,
            meta_every=2,
            no_improve_patience=1
        )
        return IntegratedSupervisor(config)
    
    @staticmethod
    def create_thorough_research_supervisor(research_goal: str,
                                          constraints: List[str] = None) -> IntegratedSupervisor:
        """Create a supervisor for thorough research (5+ cycles with meta-analysis)."""
        config = SupervisorConfig(
            research_goal=research_goal,
            constraints=constraints or [],
            max_cycles=5,
            evolution_every=2,
            proximity_every=1,
            meta_every=2,
            no_improve_patience=3
        )
        return IntegratedSupervisor(config)


def run_ai_coscientist_research(research_goal: str,
                              constraints: List[str] = None,
                              max_cycles: int = 3,
                              **kwargs) -> FullSystemState:
    """
    Convenience function to run complete AI Co-Scientist research.
    
    Args:
        research_goal: The research question to investigate
        constraints: List of research constraints
        max_cycles: Maximum number of research cycles
        **kwargs: Additional supervisor configuration
        
    Returns:
        FullSystemState with complete research results
    """
    supervisor = SupervisorFactory.create_research_supervisor(
        research_goal=research_goal,
        constraints=constraints,
        max_cycles=max_cycles,
        **kwargs
    )
    
    return supervisor.run()


def test_supervisor():
    """Test the supervisor with a sample research goal."""
    logger.info("Testing Integrated Supervisor...")
    
    # Test configuration
    test_config = SupervisorConfig(
        research_goal="Test AI Co-Scientist system integration",
        constraints=["Keep test simple", "Focus on system validation"],
        max_cycles=2,
        evolution_every=1,
        proximity_every=1,
        meta_every=2,
        no_improve_patience=1
    )
    
    try:
        # Create and run supervisor
        supervisor = IntegratedSupervisor(test_config)
        result = supervisor.run()
        
        # Validate results
        assert result.is_finished
        assert len(result.cycle_history) > 0
        assert result.finish_reason
        
        logger.info("Supervisor test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Supervisor test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_supervisor()
    print(f"Test completed with {len(test_result.cycle_history)} cycles")