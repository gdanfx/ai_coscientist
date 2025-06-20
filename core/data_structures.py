"""
AI Co-Scientist Data Structures

This module contains all shared data structures used for inter-agent communication
in the AI Co-Scientist system. These dataclasses define the state objects that
agents pass between each other.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import datetime

# ==============================================================================
# GENERATION AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class GenerationState:
    """State shared across all nodes in the generation workflow"""
    research_goal: str = ""
    constraints: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    literature_findings: List[Dict[str, Any]] = field(default_factory=list)
    synthesized_knowledge: str = ""
    generated_proposals: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    status: str = "initialized"

    def __post_init__(self):
        """Ensure all list fields are properly initialized"""
        if self.constraints is None:
            self.constraints = []
        if self.search_queries is None:
            self.search_queries = []
        if self.literature_findings is None:
            self.literature_findings = []
        if self.generated_proposals is None:
            self.generated_proposals = []

# ==============================================================================
# REFLECTION AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class ReviewCriteria:
    """Structured criteria for hypothesis evaluation"""
    novelty_score: float
    feasibility_score: float
    scientific_rigor_score: float
    impact_potential_score: float
    testability_score: float
    novelty_reasoning: str
    feasibility_reasoning: str
    scientific_rigor_reasoning: str
    impact_potential_reasoning: str
    testability_reasoning: str

@dataclass
class HypothesisReview:
    """Complete review of a single hypothesis"""
    hypothesis_id: str
    hypothesis_text: str
    criteria: ReviewCriteria
    overall_score: float
    overall_assessment: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence_level: float
    review_timestamp: str
    reviewer_type: str

@dataclass
class ReflectionState:
    """Complete state of the reflection process"""
    hypothesis_reviews: List[HypothesisReview] = field(default_factory=list)
    review_statistics: Optional[Dict[str, float]] = None
    quality_flags: Optional[List[str]] = None
    batch_summary: Optional[str] = None

# ==============================================================================
# RANKING AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class EloRating:
    """Elo rating for a hypothesis"""
    hypothesis_id: str
    current_rating: float
    initial_rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    rating_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.rating_history:
            self.rating_history = [self.initial_rating]

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        total_games = self.wins + self.losses + self.draws
        return (self.wins / total_games * 100) if total_games > 0 else 0.0

    def update_rating(self, new_rating: float, result: str):
        """Update rating and statistics"""
        self.current_rating = new_rating
        self.rating_history.append(new_rating)
        self.games_played += 1

        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1

@dataclass
class PairwiseComparison:
    """Record of a single pairwise comparison"""
    comparison_id: str
    hypothesis_a_id: str
    hypothesis_b_id: str
    winner_id: Optional[str]  # None for draw
    confidence: float  # 1-10 scale
    reasoning: str
    debate_transcript: str
    reflection_influence: float  # How much reflection scores influenced decision
    timestamp: str
    comparison_round: int

@dataclass
class RankingState:
    """Complete state of the ranking tournament"""
    elo_ratings: Dict[str, EloRating] = field(default_factory=dict)
    pairwise_comparisons: List[PairwiseComparison] = field(default_factory=list)
    tournament_rounds: int = 0
    final_rankings: List[Dict[str, Any]] = field(default_factory=list)
    ranking_statistics: Dict[str, float] = field(default_factory=dict)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    tournament_summary: str = ""

# ==============================================================================
# EVOLUTION AGENT DATA STRUCTURES
# ==============================================================================

class EvolutionStrategy(Enum):
    """Types of evolution strategies available"""
    SIMPLIFICATION = "simplification"
    COMBINATION = "combination"
    ANALOGICAL_REASONING = "analogical_reasoning"
    RADICAL_VARIATION = "radical_variation"
    CONSTRAINT_RELAXATION = "constraint_relaxation"
    DOMAIN_TRANSFER = "domain_transfer"

@dataclass
class EvolutionStep:
    """Record of a single evolution step"""
    step_id: str
    parent_hypothesis_ids: List[str]
    strategy_used: EvolutionStrategy
    evolution_prompt: str
    evolved_content: str
    improvement_rationale: str
    expected_benefits: List[str]
    potential_risks: List[str]
    novelty_increase: float  # -5 to +5 scale
    feasibility_change: float  # -5 to +5 scale
    impact_increase: float  # -5 to +5 scale
    confidence: float  # 1-10 scale
    timestamp: str

@dataclass
class EvolvedHypothesis:
    """Complete evolved hypothesis with lineage"""
    hypothesis_id: str
    evolved_content: str
    parent_hypothesis_ids: List[str]
    evolution_lineage: List[EvolutionStep]
    generation_number: int
    predicted_scores: Dict[str, float]  # Predicted reflection scores
    improvement_summary: str
    competitive_advantages: List[str]
    implementation_roadmap: List[str]

@dataclass
class EvolutionState:
    """Complete state of the evolution process"""
    evolved_hypotheses: List[EvolvedHypothesis] = field(default_factory=list)
    evolution_statistics: Dict[str, float] = field(default_factory=dict)
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    evolution_tree: Dict[str, Any] = field(default_factory=dict)
    generation_summary: str = ""
    best_evolved_hypothesis: Optional[str] = None

# ==============================================================================
# META-REVIEW AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class PatternInsight:
    """Represents a discovered pattern in hypothesis reviews"""
    description: str
    frequency: int
    sample_hypotheses: List[str]
    confidence: float = 0.0

@dataclass
class ClusteringMetrics:
    """Metrics about the clustering process"""
    n_samples: int
    n_clusters_requested: int
    n_clusters_actual: int
    clustering_method: str
    silhouette_score: Optional[float] = None

@dataclass
class AdvancedMetrics:
    """Extended metrics for deeper analysis"""
    diversity_score: float
    consensus_score: float
    trend_analysis: Dict[str, float]
    outlier_detection: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class MetaReviewState:
    """Complete state of meta-review analysis"""
    pattern_insights: List[PatternInsight]
    criterion_correlations: Dict[str, float]
    actionable_for_generation: str
    actionable_for_reflection: str
    clustering_metrics: ClusteringMetrics
    created_at: str
    analysis_quality: str = "high"
    advanced_metrics: Optional[AdvancedMetrics] = None

# ==============================================================================
# PROXIMITY AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class HypothesisCluster:
    """Represents a cluster of similar hypotheses"""
    cluster_id: int
    hypothesis_ids: List[str]
    representative_id: str
    representative_text: str
    similarity_score: float  # Average intra-cluster similarity
    cluster_size: int = 1

@dataclass
class ProximityState:
    """Complete state of the proximity analysis"""
    clusters: List[HypothesisCluster] = field(default_factory=list)
    unique_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    removed_count: int = 0
    analysis_timestamp: str = ""
    clustering_method: str = "unknown"
    original_count: int = 0
    diversity_score: float = 0.0

# ==============================================================================
# SUPERVISOR AGENT DATA STRUCTURES
# ==============================================================================

@dataclass
class SupervisorConfig:
    """Configuration for the supervisor's research process"""
    research_goal: str
    constraints: List[str] = field(default_factory=list)
    max_cycles: int = 3
    evolution_every: int = 2
    meta_every: int = 2
    proximity_every: int = 1
    no_improve_patience: int = 2

@dataclass
class CycleStats:
    """Statistics for a single research cycle"""
    cycle_id: int
    n_hypotheses: int
    best_rank_text: str
    timestamp: str

@dataclass
class SupervisorState:
    """Complete state of the supervisor's research process"""
    config: SupervisorConfig
    cycles: List[CycleStats] = field(default_factory=list)
    generation_state: Any = None
    reflection_state: Any = None
    ranking_state: Any = None
    evolution_state: Any = None
    proximity_state: Any = None
    meta_state: Any = None
    finished: bool = False
    finish_reason: str = ""

# ==============================================================================
# UNIFIED SYSTEM STATE
# ==============================================================================

@dataclass
class FullSystemState:
    """Unified state object for the complete AI Co-Scientist system"""
    config: SupervisorConfig
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[HypothesisReview] = field(default_factory=list)
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    generation_agent_state: Optional[GenerationState] = None
    reflection_agent_state: Optional[ReflectionState] = None
    ranking_agent_state: Optional[RankingState] = None
    evolution_agent_state: Optional[EvolutionState] = None
    proximity_agent_state: Optional[ProximityState] = None
    meta_review_agent_state: Optional[MetaReviewState] = None
    cycle_history: List[Dict[str, Any]] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = ""

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_timestamp() -> str:
    """Create a standardized timestamp string"""
    return datetime.datetime.now().isoformat()

def create_hypothesis_id(prefix: str = "hyp", suffix: str = "") -> str:
    """Create a unique hypothesis ID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    if suffix:
        return f"{prefix}_{timestamp}_{suffix}"
    return f"{prefix}_{timestamp}"

def validate_score(score: float, min_val: float = 1.0, max_val: float = 10.0) -> float:
    """Validate and clamp a score to the valid range"""
    return max(min_val, min(max_val, score))

def calculate_weighted_score(criteria: ReviewCriteria, 
                           weights: Dict[str, float] = None) -> float:
    """Calculate weighted overall score from criteria"""
    if weights is None:
        weights = {
            'novelty_score': 0.25,
            'feasibility_score': 0.20,
            'scientific_rigor_score': 0.25,
            'impact_potential_score': 0.20,
            'testability_score': 0.10
        }
    
    return (
        criteria.novelty_score * weights.get('novelty_score', 0.25) +
        criteria.feasibility_score * weights.get('feasibility_score', 0.20) +
        criteria.scientific_rigor_score * weights.get('scientific_rigor_score', 0.25) +
        criteria.impact_potential_score * weights.get('impact_potential_score', 0.20) +
        criteria.testability_score * weights.get('testability_score', 0.10)
    )

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Generation Agent
    'GenerationState',
    
    # Reflection Agent
    'ReviewCriteria',
    'HypothesisReview', 
    'ReflectionState',
    
    # Ranking Agent
    'EloRating',
    'PairwiseComparison',
    'RankingState',
    
    # Evolution Agent
    'EvolutionStrategy',
    'EvolutionStep',
    'EvolvedHypothesis',
    'EvolutionState',
    
    # Meta-Review Agent
    'PatternInsight',
    'ClusteringMetrics',
    'AdvancedMetrics',
    'MetaReviewState',
    
    # Proximity Agent
    'HypothesisCluster',
    'ProximityState',
    
    # Supervisor Agent
    'SupervisorConfig',
    'CycleStats',
    'SupervisorState',
    'FullSystemState',
    
    # Utility Functions
    'create_timestamp',
    'create_hypothesis_id',
    'validate_score',
    'calculate_weighted_score'
]