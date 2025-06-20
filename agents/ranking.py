"""
Ranking Agent - Elo Tournament System

This module implements the RankingAgent that uses an Elo rating system
to conduct competitive hypothesis ranking through pairwise debates.
"""

import re
import math
import random
import statistics
import logging
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations

from core.data_structures import (
    EloRating, PairwiseComparison, RankingState, HypothesisReview,
    create_timestamp, create_hypothesis_id
)
from utils.llm_client import get_global_llm_client

logger = logging.getLogger(__name__)


class EloSystem:
    """Elo rating system for hypothesis tournaments."""

    def __init__(self, k_factor: float = 32, initial_rating: float = 1200):
        self.k_factor = k_factor  # Rating volatility
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))

    def update_ratings(self, rating_a: float, rating_b: float,
                      actual_score_a: float) -> Tuple[float, float]:
        """
        Update Elo ratings based on match result.
        actual_score_a: 1 for A wins, 0 for A loses, 0.5 for draw
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        actual_score_b = 1 - actual_score_a

        new_rating_a = rating_a + self.k_factor * (actual_score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_score_b - expected_b)

        return new_rating_a, new_rating_b

    def rating_difference_to_win_probability(self, rating_diff: float) -> float:
        """Convert rating difference to win probability."""
        return 1 / (1 + 10**(-rating_diff / 400))


class RankingAgent:
    """
    Ranking Agent that implements Elo-based tournament system with self-debate
    for pairwise hypothesis comparison and ranking.
    """

    def __init__(self, llm_client=None, k_factor: float = 32, initial_rating: float = 1200):
        self.llm = llm_client or get_global_llm_client()
        self.elo_system = EloSystem(k_factor, initial_rating)
        self.initial_rating = initial_rating

    def initialize_ratings(self, hypothesis_reviews: List[HypothesisReview]) -> Dict[str, EloRating]:
        """Initialize Elo ratings for all hypotheses."""
        ratings = {}

        for review in hypothesis_reviews:
            # Use reflection score to adjust initial rating slightly
            reflection_bonus = (review.overall_score - 6.0) * 20  # ±40 points max
            initial_rating = self.initial_rating + reflection_bonus

            ratings[review.hypothesis_id] = EloRating(
                hypothesis_id=review.hypothesis_id,
                current_rating=initial_rating,
                initial_rating=initial_rating,
                games_played=0,
                wins=0,
                losses=0,
                draws=0,
                rating_history=[initial_rating]
            )

        logger.info(f"Initialized {len(ratings)} hypothesis ratings")
        return ratings

    def pairwise_debate(self, hypothesis_a: HypothesisReview, hypothesis_b: HypothesisReview,
                       comparison_round: int) -> PairwiseComparison:
        """Conduct self-debate to compare two hypotheses."""
        logger.info(f"Starting pairwise debate: {hypothesis_a.hypothesis_id} vs {hypothesis_b.hypothesis_id}")

        debate_prompt = f"""You are a senior scientific review panel conducting a rigorous comparison between two research hypotheses. You must engage in self-debate to determine which hypothesis is superior.

HYPOTHESIS A ({hypothesis_a.hypothesis_id}):
Text: {hypothesis_a.hypothesis_text}
Reflection Score: {hypothesis_a.overall_score:.2f}/10
Key Strengths: {'; '.join(hypothesis_a.strengths[:3])}
Key Weaknesses: {'; '.join(hypothesis_a.weaknesses[:3])}

HYPOTHESIS B ({hypothesis_b.hypothesis_id}):
Text: {hypothesis_b.hypothesis_text}
Reflection Score: {hypothesis_b.overall_score:.2f}/10
Key Strengths: {'; '.join(hypothesis_b.strengths[:3])}
Key Weaknesses: {'; '.join(hypothesis_b.weaknesses[:3])}

CONDUCT A STRUCTURED DEBATE:

**ROUND 1 - Advocate for Hypothesis A:**
Present the strongest case for why Hypothesis A is superior. Consider novelty, feasibility, impact potential, and scientific rigor.

**ROUND 2 - Advocate for Hypothesis B:**
Present the strongest case for why Hypothesis B is superior. Consider the same criteria and directly address A's advantages.

**ROUND 3 - Critical Analysis:**
Critically examine both hypotheses. What are the decisive factors? Which hypothesis has the strongest foundation and highest potential for breakthrough science?

**FINAL JUDGMENT:**
Winner: [A or B or DRAW]
Confidence: [1-10 scale]
Reflection Influence: [How much did the reflection scores influence your decision? 1-10 scale]

**Reasoning:** [2-3 sentences explaining the decisive factors in your decision]

Conduct this debate rigorously and choose the hypothesis that would most likely lead to significant scientific advancement."""

        try:
            response = self.llm.invoke(debate_prompt)
            if response.error:
                logger.error(f"LLM error during debate: {response.error}")
                return self._create_fallback_comparison(hypothesis_a, hypothesis_b, comparison_round)
            
            return self._parse_debate_result(response.content, hypothesis_a, hypothesis_b, comparison_round)

        except Exception as e:
            logger.error(f"Debate failed for {hypothesis_a.hypothesis_id} vs {hypothesis_b.hypothesis_id}: {e}")
            return self._create_fallback_comparison(hypothesis_a, hypothesis_b, comparison_round)

    def _parse_debate_result(self, debate_text: str, hypothesis_a: HypothesisReview,
                           hypothesis_b: HypothesisReview, comparison_round: int) -> PairwiseComparison:
        """Parse the debate result to determine winner."""
        # Extract winner using multiple patterns
        winner_patterns = [
            r'winner.*?[:\s]*([ABab]|draw|tie)',
            r'final.*?judgment.*?[:\s]*([ABab]|draw|tie)',
            r'decision.*?[:\s]*([ABab]|draw|tie)',
            r'superior.*?hypothesis.*?([ABab])'
        ]

        winner_id = None
        for pattern in winner_patterns:
            match = re.search(pattern, debate_text, re.IGNORECASE)
            if match:
                result = match.group(1).lower()
                if result in ['a', 'hypothesis a']:
                    winner_id = hypothesis_a.hypothesis_id
                elif result in ['b', 'hypothesis b']:
                    winner_id = hypothesis_b.hypothesis_id
                elif result in ['draw', 'tie']:
                    winner_id = None
                break

        # If no clear winner found, use reflection scores as fallback
        if winner_id is None:
            if hypothesis_a.overall_score > hypothesis_b.overall_score + 0.5:
                winner_id = hypothesis_a.hypothesis_id
            elif hypothesis_b.overall_score > hypothesis_a.overall_score + 0.5:
                winner_id = hypothesis_b.hypothesis_id
            # Otherwise remains None (draw)

        # Extract confidence
        confidence_match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', debate_text, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 7.0

        # Extract reflection influence
        influence_match = re.search(r'reflection.*?influence.*?(\d+(?:\.\d+)?)', debate_text, re.IGNORECASE)
        reflection_influence = float(influence_match.group(1)) if influence_match else 5.0

        # Extract reasoning
        reasoning_patterns = [
            r'reasoning.*?:(.*?)(?:\n\n|$)',
            r'decisive.*?factors.*?:(.*?)(?:\n\n|$)',
            r'explanation.*?:(.*?)(?:\n\n|$)'
        ]

        reasoning = "Reasoning could not be extracted from debate."
        for pattern in reasoning_patterns:
            match = re.search(pattern, debate_text, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()[:300]
                break

        comparison_id = f"{hypothesis_a.hypothesis_id}_vs_{hypothesis_b.hypothesis_id}_r{comparison_round}"

        return PairwiseComparison(
            comparison_id=comparison_id,
            hypothesis_a_id=hypothesis_a.hypothesis_id,
            hypothesis_b_id=hypothesis_b.hypothesis_id,
            winner_id=winner_id,
            confidence=confidence,
            reasoning=reasoning,
            debate_transcript=debate_text[:1000] + "..." if len(debate_text) > 1000 else debate_text,
            reflection_influence=reflection_influence,
            timestamp=create_timestamp(),
            comparison_round=comparison_round
        )

    def _create_fallback_comparison(self, hypothesis_a: HypothesisReview,
                                  hypothesis_b: HypothesisReview, comparison_round: int) -> PairwiseComparison:
        """Create fallback comparison based on reflection scores."""
        score_diff = hypothesis_a.overall_score - hypothesis_b.overall_score

        if abs(score_diff) < 0.3:
            winner_id = None  # Draw for very close scores
        elif score_diff > 0:
            winner_id = hypothesis_a.hypothesis_id
        else:
            winner_id = hypothesis_b.hypothesis_id

        comparison_id = f"{hypothesis_a.hypothesis_id}_vs_{hypothesis_b.hypothesis_id}_r{comparison_round}_fallback"

        return PairwiseComparison(
            comparison_id=comparison_id,
            hypothesis_a_id=hypothesis_a.hypothesis_id,
            hypothesis_b_id=hypothesis_b.hypothesis_id,
            winner_id=winner_id,
            confidence=5.0,  # Lower confidence for fallback
            reasoning=f"Fallback comparison based on reflection scores: {hypothesis_a.overall_score:.2f} vs {hypothesis_b.overall_score:.2f}",
            debate_transcript="Fallback comparison - no debate transcript available",
            reflection_influence=10.0,  # Fully based on reflection
            timestamp=create_timestamp(),
            comparison_round=comparison_round
        )

    def run_tournament_round(self, hypothesis_reviews: List[HypothesisReview],
                           elo_ratings: Dict[str, EloRating], round_number: int,
                           max_comparisons: int = None) -> List[PairwiseComparison]:
        """Run a single round of the tournament with pairwise comparisons."""
        logger.info(f"Starting tournament round {round_number}")

        # Create review lookup for easy access
        review_lookup = {review.hypothesis_id: review for review in hypothesis_reviews}

        # Generate all possible pairs
        hypothesis_ids = list(elo_ratings.keys())
        all_pairs = list(combinations(hypothesis_ids, 2))

        # Limit comparisons if specified
        if max_comparisons and len(all_pairs) > max_comparisons:
            # Prioritize pairs with similar ratings for more informative comparisons
            def rating_similarity(pair):
                rating_a = elo_ratings[pair[0]].current_rating
                rating_b = elo_ratings[pair[1]].current_rating
                return abs(rating_a - rating_b)

            all_pairs.sort(key=rating_similarity)
            selected_pairs = all_pairs[:max_comparisons]
        else:
            selected_pairs = all_pairs

        round_comparisons = []

        for hypothesis_a_id, hypothesis_b_id in selected_pairs:
            # Get reviews
            review_a = review_lookup.get(hypothesis_a_id)
            review_b = review_lookup.get(hypothesis_b_id)

            if not review_a or not review_b:
                logger.warning(f"Could not find reviews for pair: {hypothesis_a_id}, {hypothesis_b_id}. Skipping.")
                continue

            # Conduct debate
            comparison = self.pairwise_debate(review_a, review_b, round_number)
            round_comparisons.append(comparison)

            # Update Elo ratings
            rating_a = elo_ratings[hypothesis_a_id].current_rating
            rating_b = elo_ratings[hypothesis_b_id].current_rating

            # Convert result to scores
            if comparison.winner_id == hypothesis_a_id:
                score_a, score_b = 1.0, 0.0
                result_a, result_b = "win", "loss"
            elif comparison.winner_id == hypothesis_b_id:
                score_a, score_b = 0.0, 1.0
                result_a, result_b = "loss", "win"
            else:
                score_a, score_b = 0.5, 0.5
                result_a, result_b = "draw", "draw"

            # Update ratings
            new_rating_a, new_rating_b = self.elo_system.update_ratings(rating_a, rating_b, score_a)

            elo_ratings[hypothesis_a_id].update_rating(new_rating_a, result_a)
            elo_ratings[hypothesis_b_id].update_rating(new_rating_b, result_b)

        logger.info(f"Completed round {round_number}: {len(round_comparisons)} comparisons")
        return round_comparisons

    def run_full_tournament(self, reflection_state, num_rounds: int = 3,
                          max_comparisons_per_round: int = None) -> RankingState:
        """Run complete tournament with multiple rounds."""
        logger.info(f"Starting full tournament with {num_rounds} rounds")

        # Initialize ranking state
        ranking_state = RankingState()

        # Initialize Elo ratings
        ranking_state.elo_ratings = self.initialize_ratings(reflection_state.hypothesis_reviews)

        # Run tournament rounds
        for round_num in range(1, num_rounds + 1):
            round_comparisons = self.run_tournament_round(
                reflection_state.hypothesis_reviews,
                ranking_state.elo_ratings,
                round_num,
                max_comparisons_per_round
            )
            ranking_state.pairwise_comparisons.extend(round_comparisons)

        ranking_state.tournament_rounds = num_rounds

        # Generate final rankings and statistics
        ranking_state.final_rankings = self._generate_final_rankings(ranking_state)
        ranking_state.ranking_statistics = self._compute_ranking_statistics(ranking_state)
        ranking_state.convergence_metrics = self._compute_convergence_metrics(ranking_state)
        ranking_state.tournament_summary = self._generate_tournament_summary(ranking_state)

        logger.info(f"Tournament completed: {len(ranking_state.pairwise_comparisons)} total comparisons")
        return ranking_state

    def _generate_final_rankings(self, ranking_state: RankingState) -> List[Dict[str, Any]]:
        """Generate final hypothesis rankings."""
        rankings = []

        # Sort hypotheses by final Elo rating
        sorted_hypotheses = sorted(
            ranking_state.elo_ratings.items(),
            key=lambda x: x[1].current_rating,
            reverse=True
        )

        for rank, (hypothesis_id, elo_rating) in enumerate(sorted_hypotheses, 1):
            ranking_entry = {
                'rank': rank,
                'hypothesis_id': hypothesis_id,
                'final_elo_rating': elo_rating.current_rating,
                'initial_elo_rating': elo_rating.initial_rating,
                'rating_change': elo_rating.current_rating - elo_rating.initial_rating,
                'games_played': elo_rating.games_played,
                'wins': elo_rating.wins,
                'losses': elo_rating.losses,
                'draws': elo_rating.draws,
                'win_rate': elo_rating.win_rate,
                'rating_volatility': statistics.stdev(elo_rating.rating_history) if len(elo_rating.rating_history) > 1 else 0
            }
            rankings.append(ranking_entry)

        return rankings

    def _compute_ranking_statistics(self, ranking_state: RankingState) -> Dict[str, float]:
        """Compute comprehensive ranking statistics."""
        if not ranking_state.elo_ratings:
            return {}

        if not ranking_state.pairwise_comparisons:
            return {'total_hypotheses': len(ranking_state.elo_ratings), 'total_comparisons': 0}

        current_ratings = [rating.current_rating for rating in ranking_state.elo_ratings.values()]
        initial_ratings = [rating.initial_rating for rating in ranking_state.elo_ratings.values()]
        rating_changes = [current - initial for current, initial in zip(current_ratings, initial_ratings)]

        total_comparisons = len(ranking_state.pairwise_comparisons)
        decisive_wins = len([c for c in ranking_state.pairwise_comparisons if c.winner_id is not None])
        draws = total_comparisons - decisive_wins

        avg_confidence = statistics.mean([c.confidence for c in ranking_state.pairwise_comparisons])
        avg_reflection_influence = statistics.mean([c.reflection_influence for c in ranking_state.pairwise_comparisons])

        return {
            'total_hypotheses': len(ranking_state.elo_ratings),
            'total_comparisons': total_comparisons,
            'decisive_outcomes': decisive_wins,
            'draws': draws,
            'draw_rate': (draws / total_comparisons * 100) if total_comparisons > 0 else 0,
            'mean_final_rating': statistics.mean(current_ratings),
            'rating_spread': max(current_ratings) - min(current_ratings),
            'mean_rating_change': statistics.mean(rating_changes),
            'max_rating_change': max(rating_changes),
            'min_rating_change': min(rating_changes),
            'rating_volatility': statistics.stdev(current_ratings) if len(current_ratings) > 1 else 0,
            'average_confidence': avg_confidence,
            'reflection_influence': avg_reflection_influence,
            'tournament_rounds': ranking_state.tournament_rounds
        }

    def _compute_convergence_metrics(self, ranking_state: RankingState) -> Dict[str, float]:
        """Compute metrics indicating ranking convergence."""
        convergence = {}

        # Rating stability (how much ratings changed in later rounds)
        if ranking_state.tournament_rounds >= 2:
            early_ratings = {}
            late_ratings = {}

            for hypothesis_id, elo_rating in ranking_state.elo_ratings.items():
                history = elo_rating.rating_history
                if len(history) >= 3:
                    early_ratings[hypothesis_id] = history[len(history)//2]
                    late_ratings[hypothesis_id] = history[-1]

            if early_ratings and late_ratings:
                rating_stability = statistics.mean([
                    abs(late_ratings[h_id] - early_ratings[h_id])
                    for h_id in early_ratings.keys()
                ])
                convergence['rating_stability'] = rating_stability
                convergence['converged'] = rating_stability < 50  # Threshold for convergence

        # Ranking consistency (how often the same hypothesis wins)
        if ranking_state.pairwise_comparisons:
            winner_counts = {}
            for comparison in ranking_state.pairwise_comparisons:
                if comparison.winner_id:
                    winner_counts[comparison.winner_id] = winner_counts.get(comparison.winner_id, 0) + 1

            if winner_counts:
                total_wins = sum(winner_counts.values())
                max_wins = max(winner_counts.values())
                convergence['winner_concentration'] = max_wins / total_wins if total_wins > 0 else 0

        return convergence

    def _generate_tournament_summary(self, ranking_state: RankingState) -> str:
        """Generate comprehensive tournament summary."""
        stats = ranking_state.ranking_statistics
        convergence = ranking_state.convergence_metrics
        rankings = ranking_state.final_rankings

        if not stats or not rankings:
            return "Summary not available."

        # Get top 3 hypotheses
        top_3 = rankings[:3] if len(rankings) >= 3 else rankings

        summary = f"""
RANKING AGENT TOURNAMENT SUMMARY

🏆 TOURNAMENT RESULTS:
• Total hypotheses: {stats.get('total_hypotheses', 0)}
• Tournament rounds: {stats.get('tournament_rounds', 0)}
• Total comparisons: {stats.get('total_comparisons', 0)}
• Decisive outcomes: {stats.get('decisive_outcomes', 0)} ({100-stats.get('draw_rate', 0):.1f}%)
• Draws: {stats.get('draws', 0)} ({stats.get('draw_rate', 0):.1f}%)

📊 RATING ANALYSIS:
• Final rating range: {stats.get('mean_final_rating', 0) - stats.get('rating_volatility', 0)/2:.0f} - {stats.get('mean_final_rating', 0) + stats.get('rating_volatility', 0)/2:.0f}
• Rating spread: {stats.get('rating_spread', 0):.0f} points
• Average rating change: {stats.get('mean_rating_change', 0):+.1f} points
• Biggest winner: {stats.get('max_rating_change', 0):+.1f} points
• Biggest loser: {stats.get('min_rating_change', 0):+.1f} points

🎯 DECISION QUALITY:
• Average confidence: {stats.get('average_confidence', 0):.1f}/10
• Reflection influence: {stats.get('reflection_influence', 0):.1f}/10
• Tournament convergence: {'Yes' if convergence.get('converged', False) else 'Partial'}

🥇 TOP 3 RANKINGS:
{self._format_top_rankings(top_3)}

📈 CONVERGENCE ANALYSIS:
• Rating stability: {convergence.get('rating_stability', 0):.1f} points
• Winner concentration: {convergence.get('winner_concentration', 0)*100:.1f}%
• Tournament quality: {'Excellent' if stats.get('average_confidence', 0) > 7.5 else 'Good' if stats.get('average_confidence', 0) > 6.0 else 'Fair'}

💡 RECOMMENDATIONS:
{self._generate_ranking_recommendations(stats, convergence, rankings)}
"""

        return summary.strip()

    def _format_top_rankings(self, top_rankings: List[Dict[str, Any]]) -> str:
        """Format top rankings for display."""
        formatted = ""
        for ranking in top_rankings:
            formatted += f"""
• #{ranking['rank']}: {ranking['hypothesis_id']}
  Elo: {ranking['final_elo_rating']:.0f} ({ranking['rating_change']:+.0f})
  Record: {ranking['wins']}-{ranking['losses']}-{ranking['draws']} ({ranking['win_rate']:.1f}%)"""

        return formatted

    def _generate_ranking_recommendations(self, stats: Dict[str, float],
                                        convergence: Dict[str, float],
                                        rankings: List[Dict[str, Any]]) -> str:
        """Generate strategic recommendations based on tournament results."""
        recommendations = []

        if not stats or not rankings:
            return "No recommendations available."

        # Top hypothesis recommendations
        if rankings:
            top_hypothesis = rankings[0]
            if top_hypothesis['final_elo_rating'] > stats.get('mean_final_rating', 1200) + 100:
                recommendations.append(f"• PRIORITY: {top_hypothesis['hypothesis_id']} is clearly superior - proceed to Evolution Agent")
            elif top_hypothesis['win_rate'] > 70:
                recommendations.append(f"• STRONG CANDIDATE: {top_hypothesis['hypothesis_id']} shows consistent performance")

        # Convergence recommendations
        if convergence.get('converged', False):
            recommendations.append("• STABLE: Rankings have converged - results are reliable")
        else:
            recommendations.append("• CONTINUE: Consider additional tournament rounds for stability")

        # Quality recommendations
        avg_confidence = stats.get('average_confidence', 0)
        if avg_confidence < 6.0:
            recommendations.append("• REVIEW: Low confidence suggests difficult comparisons - validate manually")
        elif avg_confidence > 8.0:
            recommendations.append("• CONFIDENT: High confidence in ranking decisions")

        # Draw rate recommendations
        draw_rate = stats.get('draw_rate', 0)
        if draw_rate > 30:
            recommendations.append("• SIMILAR QUALITY: High draw rate indicates comparable hypotheses")
        elif draw_rate < 10:
            recommendations.append("• CLEAR DIFFERENCES: Low draw rate shows distinct quality levels")

        # Next steps
        if len(rankings) >= 3:
            top_3_ratings = [r['final_elo_rating'] for r in rankings[:3]]
            if max(top_3_ratings) - min(top_3_ratings) < 50:
                recommendations.append("• NEXT: Top hypotheses are close - consider parallel development")
            else:
                recommendations.append("• NEXT: Clear winner identified - focus evolution efforts")

        return '\n'.join(recommendations) if recommendations else "• Proceed with evolved hypotheses as planned"


# Factory function for easy instantiation
def create_ranking_agent(**kwargs) -> RankingAgent:
    """Factory function to create a ranking agent with optional parameters."""
    return RankingAgent(**kwargs)


# Integration function
def run_ranking_agent(reflection_state, tournament_rounds: int = 3,
                     max_comparisons_per_round: int = None):
    """
    Integration function for the Ranking Agent.
    
    Args:
        reflection_state: State object with hypothesis_reviews
        tournament_rounds: Number of tournament rounds to run
        max_comparisons_per_round: Maximum comparisons per round
        
    Returns:
        RankingState with tournament results
    """
    logger.info("🏆 Engaging Ranking Agent...")
    
    if not hasattr(reflection_state, 'hypothesis_reviews') or not reflection_state.hypothesis_reviews:
        logger.warning("No hypothesis reviews found in reflection_state for ranking")
        return RankingState()

    # Initialize and run the agent
    ranking_agent = RankingAgent()
    ranking_results = ranking_agent.run_full_tournament(
        reflection_state, tournament_rounds, max_comparisons_per_round
    )

    logger.info(f"✅ Ranking Agent finished: {len(ranking_results.final_rankings)} hypotheses ranked")
    
    return ranking_results


# Testing function
def test_ranking_agent():
    """Test the Ranking Agent with sample hypotheses."""
    logger.info("Testing Ranking Agent...")
    
    # Test data structure imports
    from core.data_structures import ReviewCriteria, HypothesisReview, ReflectionState
    
    # Create mock reviews for testing
    mock_reviews = []
    hypotheses_content = [
        "Novel epigenetic reprogramming approach targeting DNMT3A for liver fibrosis treatment",
        "AI-driven personalized medicine platform for optimized combination therapies",
        "Single-cell epigenomic mapping to identify therapeutic vulnerabilities",
        "Multi-target combination therapy using BRD4 and HDAC inhibitors"
    ]
    
    for i, content in enumerate(hypotheses_content):
        criteria = ReviewCriteria(
            novelty_score=7.0 + i * 0.5,
            feasibility_score=6.0 + i * 0.3,
            scientific_rigor_score=7.5 + i * 0.2,
            impact_potential_score=8.0 + i * 0.4,
            testability_score=6.5 + i * 0.3,
            novelty_reasoning="Novel approach",
            feasibility_reasoning="Technically feasible",
            scientific_rigor_reasoning="Well-grounded",
            impact_potential_reasoning="High impact",
            testability_reasoning="Clear validation"
        )
        
        review = HypothesisReview(
            hypothesis_id=f"test_hyp_{i+1}",
            hypothesis_text=content,
            criteria=criteria,
            overall_score=7.0 + i * 0.4,
            overall_assessment=f"Strong hypothesis with good potential",
            strengths=["Novel approach", "Clear rationale", "Strong foundation"],
            weaknesses=["Needs validation", "Resource requirements"],
            recommendations=["Further testing", "Optimize approach"],
            confidence_level=8.0,
            review_timestamp=create_timestamp(),
            reviewer_type="detailed"
        )
        mock_reviews.append(review)
    
    # Create mock reflection state
    mock_reflection_state = ReflectionState()
    mock_reflection_state.hypothesis_reviews = mock_reviews
    
    try:
        # Test agent
        agent = RankingAgent(llm_client=None)  # Use None to trigger heuristic fallback
        result = agent.run_full_tournament(mock_reflection_state, num_rounds=2, max_comparisons_per_round=4)
        
        # Validate results
        assert len(result.final_rankings) == len(mock_reviews)
        assert result.ranking_statistics is not None
        assert isinstance(result.convergence_metrics, dict)
        assert result.tournament_summary is not None
        
        logger.info(f"Ranking Agent test completed successfully: "
                   f"{len(result.final_rankings)} hypotheses ranked, "
                   f"tournament quality: {result.ranking_statistics.get('average_confidence', 0):.1f}/10")
        return result
        
    except Exception as e:
        logger.error(f"Ranking Agent test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_ranking_agent()
    print(f"Test completed: {len(test_result.final_rankings)} hypotheses ranked, "
          f"mean confidence: {test_result.ranking_statistics.get('average_confidence', 0):.1f}/10")