"""
Meta-Review Agent - System-Level Analysis and Bias Detection

This module implements the Enhanced Meta-Review Agent that performs system-level
analysis of hypothesis evaluation patterns, detects biases, and provides actionable
insights for improving the research process.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from core.data_structures import (
    MetaReviewState, PatternInsight, ClusteringMetrics, AdvancedMetrics,
    create_timestamp
)
from utils.text_processing import TextAnalyzer, create_analyzer

logger = logging.getLogger(__name__)


class EnhancedMetaReviewAgent:
    """
    Performs system-level analysis of hypothesis evaluation patterns and detects biases.
    
    Analyzes correlation patterns across review criteria, identifies systematic
    biases in evaluation, and provides actionable recommendations for improving
    the research process.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the Meta-Review Agent.
        
        Args:
            confidence_threshold: Minimum confidence for pattern insights
        """
        self.confidence_threshold = confidence_threshold
        self.text_analyzer = create_analyzer()
        
    def run(self, reflection_state: Any) -> MetaReviewState:
        """
        Perform comprehensive meta-review analysis.
        
        Args:
            reflection_state: State object containing hypothesis reviews
            
        Returns:
            MetaReviewState with pattern insights and recommendations
        """
        logger.info("Starting meta-review analysis")
        
        if not hasattr(reflection_state, 'hypothesis_reviews') or not reflection_state.hypothesis_reviews:
            logger.warning("No hypothesis reviews found for meta-analysis")
            return self._create_empty_state()
        
        try:
            # Extract review data
            reviews = reflection_state.hypothesis_reviews
            
            # Core analysis components
            pattern_insights = self._detect_patterns(reviews)
            correlations = self._analyze_correlations(reviews)
            clustering_metrics = self._analyze_clustering_patterns(reviews)
            advanced_metrics = self._compute_advanced_metrics(reviews)
            
            # Generate actionable recommendations
            generation_recommendations = self._generate_recommendations_for_generation(
                pattern_insights, correlations
            )
            reflection_recommendations = self._generate_recommendations_for_reflection(
                pattern_insights, correlations
            )
            
            # Determine overall analysis quality
            analysis_quality = self._assess_analysis_quality(reviews, correlations)
            
            state = MetaReviewState(
                pattern_insights=pattern_insights,
                criterion_correlations=correlations,
                actionable_for_generation=generation_recommendations,
                actionable_for_reflection=reflection_recommendations,
                clustering_metrics=clustering_metrics,
                created_at=create_timestamp(),
                analysis_quality=analysis_quality,
                advanced_metrics=advanced_metrics
            )
            
            logger.info(f"Meta-review analysis completed: {len(pattern_insights)} patterns detected, "
                       f"quality: {analysis_quality}")
            
            return state
            
        except Exception as e:
            logger.error(f"Meta-review analysis failed: {e}")
            return self._create_empty_state()
    
    def _detect_patterns(self, reviews: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Detect systematic patterns in hypothesis evaluations."""
        patterns = []
        
        try:
            # Pattern 1: Score distribution patterns
            score_patterns = self._analyze_score_distributions(reviews)
            patterns.extend(score_patterns)
            
            # Pattern 2: Criteria bias patterns
            bias_patterns = self._detect_criteria_bias(reviews)
            patterns.extend(bias_patterns)
            
            # Pattern 3: Content-based patterns
            content_patterns = self._analyze_content_patterns(reviews)
            patterns.extend(content_patterns)
            
            # Pattern 4: Consistency patterns
            consistency_patterns = self._analyze_consistency_patterns(reviews)
            patterns.extend(consistency_patterns)
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        
        return patterns
    
    def _analyze_score_distributions(self, reviews: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Analyze score distribution patterns across criteria."""
        patterns = []
        
        # Extract scores by criteria
        criteria_scores = defaultdict(list)
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                for criterion, score in review_data.items():
                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        criteria_scores[criterion].append(score)
        
        # Analyze distributions
        for criterion, scores in criteria_scores.items():
            if len(scores) < 3:  # Need minimum samples
                continue
                
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Detect extreme distributions
            if std_score < 1.0:  # Very low variance
                patterns.append(PatternInsight(
                    description=f"Low variance in {criterion} scores (std={std_score:.2f})",
                    frequency=len(scores),
                    sample_hypotheses=[str(i) for i in range(min(3, len(scores)))],
                    confidence=min(0.9, 1.0 - std_score)
                ))
            
            if mean_score > 8.0:  # High average scores
                patterns.append(PatternInsight(
                    description=f"Consistently high {criterion} scores (mean={mean_score:.2f})",
                    frequency=len(scores),
                    sample_hypotheses=[str(i) for i in range(min(3, len(scores)))],
                    confidence=min(0.9, (mean_score - 5.0) / 5.0)
                ))
            
            if mean_score < 3.0:  # Low average scores
                patterns.append(PatternInsight(
                    description=f"Consistently low {criterion} scores (mean={mean_score:.2f})",
                    frequency=len(scores),
                    sample_hypotheses=[str(i) for i in range(min(3, len(scores)))],
                    confidence=min(0.9, (5.0 - mean_score) / 5.0)
                ))
        
        return patterns
    
    def _detect_criteria_bias(self, reviews: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Detect systematic biases in evaluation criteria."""
        patterns = []
        
        # Count criteria usage frequency
        criteria_counts = Counter()
        total_reviews = len(reviews)
        
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                for criterion in review_data.keys():
                    criteria_counts[criterion] += 1
        
        # Detect under/over-used criteria
        for criterion, count in criteria_counts.items():
            usage_rate = count / total_reviews if total_reviews > 0 else 0
            
            if usage_rate < 0.5:  # Under-used criterion
                patterns.append(PatternInsight(
                    description=f"Under-utilized evaluation criterion: {criterion}",
                    frequency=count,
                    sample_hypotheses=[f"Used in {count}/{total_reviews} reviews"],
                    confidence=max(0.5, 1.0 - usage_rate * 2)
                ))
            
            if usage_rate > 0.95 and total_reviews > 5:  # Over-used criterion
                patterns.append(PatternInsight(
                    description=f"Over-relied evaluation criterion: {criterion}",
                    frequency=count,
                    sample_hypotheses=[f"Used in {count}/{total_reviews} reviews"],
                    confidence=min(0.9, (usage_rate - 0.8) * 5)
                ))
        
        return patterns
    
    def _analyze_content_patterns(self, reviews: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Analyze patterns in hypothesis content and reviews."""
        patterns = []
        
        if not self.text_analyzer:
            return patterns
        
        try:
            # Extract hypothesis contents
            hypothesis_texts = []
            for review in reviews:
                content = review.get('hypothesis_content', '')
                if isinstance(content, str) and len(content) > 10:
                    hypothesis_texts.append(content)
            
            if len(hypothesis_texts) < 3:
                return patterns
            
            # Analyze keyword patterns
            keywords = self.text_analyzer.extract_keywords(hypothesis_texts, top_k=10)
            
            if keywords:
                # Detect dominant themes
                total_words = sum(count for _, count in keywords)
                dominant_keywords = [(word, count) for word, count in keywords 
                                   if count / total_words > 0.1]
                
                if dominant_keywords:
                    dominant_word, dominant_count = dominant_keywords[0]
                    patterns.append(PatternInsight(
                        description=f"Dominant research theme: '{dominant_word}' appears in {dominant_count} hypotheses",
                        frequency=dominant_count,
                        sample_hypotheses=[dominant_word],
                        confidence=min(0.9, dominant_count / len(hypothesis_texts))
                    ))
            
            # Analyze hypothesis diversity
            if len(hypothesis_texts) >= 5:
                avg_similarity = self._calculate_average_similarity(hypothesis_texts)
                if avg_similarity > 0.7:
                    patterns.append(PatternInsight(
                        description=f"Low hypothesis diversity detected (avg similarity: {avg_similarity:.2f})",
                        frequency=len(hypothesis_texts),
                        sample_hypotheses=hypothesis_texts[:3],
                        confidence=min(0.9, avg_similarity)
                    ))
                    
        except Exception as e:
            logger.warning(f"Content pattern analysis failed: {e}")
        
        return patterns
    
    def _analyze_consistency_patterns(self, reviews: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Analyze consistency patterns in evaluations."""
        patterns = []
        
        # Group reviews by hypothesis if possible
        hypothesis_groups = defaultdict(list)
        for review in reviews:
            hypothesis_id = review.get('hypothesis_id', 'unknown')
            hypothesis_groups[hypothesis_id].append(review)
        
        # Analyze multi-review consistency
        inconsistent_count = 0
        for hypothesis_id, hypothesis_reviews in hypothesis_groups.items():
            if len(hypothesis_reviews) > 1:
                # Calculate score variance across reviews for same hypothesis
                score_variance = self._calculate_review_variance(hypothesis_reviews)
                if score_variance > 4.0:  # High variance threshold
                    inconsistent_count += 1
        
        if inconsistent_count > 0:
            patterns.append(PatternInsight(
                description=f"Inconsistent scoring detected across {inconsistent_count} hypotheses",
                frequency=inconsistent_count,
                sample_hypotheses=[f"Variance > 4.0 in {inconsistent_count} cases"],
                confidence=min(0.8, inconsistent_count / len(hypothesis_groups))
            ))
        
        return patterns
    
    def _analyze_correlations(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlations between evaluation criteria."""
        correlations = {}
        
        try:
            # Extract criteria scores
            criteria_data = defaultdict(list)
            for review in reviews:
                review_data = review.get('review', {})
                if isinstance(review_data, dict):
                    for criterion, score in review_data.items():
                        if isinstance(score, (int, float)):
                            criteria_data[criterion].append(score)
            
            # Calculate pairwise correlations
            criteria_list = list(criteria_data.keys())
            for i, criterion1 in enumerate(criteria_list):
                for j, criterion2 in enumerate(criteria_list[i+1:], i+1):
                    scores1 = criteria_data[criterion1]
                    scores2 = criteria_data[criterion2]
                    
                    # Align scores (same length)
                    min_len = min(len(scores1), len(scores2))
                    if min_len >= 3:  # Need minimum samples
                        aligned_scores1 = scores1[:min_len]
                        aligned_scores2 = scores2[:min_len]
                        
                        # Calculate correlation
                        correlation = np.corrcoef(aligned_scores1, aligned_scores2)[0, 1]
                        if not np.isnan(correlation):
                            pair_key = f"{criterion1}_vs_{criterion2}"
                            correlations[pair_key] = float(correlation)
                            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
        
        return correlations
    
    def _analyze_clustering_patterns(self, reviews: List[Dict[str, Any]]) -> ClusteringMetrics:
        """Analyze clustering patterns in the reviews."""
        try:
            # Basic clustering metrics
            n_samples = len(reviews)
            unique_criteria = set()
            
            for review in reviews:
                review_data = review.get('review', {})
                if isinstance(review_data, dict):
                    unique_criteria.update(review_data.keys())
            
            return ClusteringMetrics(
                n_samples=n_samples,
                n_clusters_requested=len(unique_criteria),
                n_clusters_actual=len(unique_criteria),
                clustering_method="criteria_analysis",
                silhouette_score=None  # Could implement if needed
            )
            
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return ClusteringMetrics(
                n_samples=len(reviews),
                n_clusters_requested=0,
                n_clusters_actual=0,
                clustering_method="failed"
            )
    
    def _compute_advanced_metrics(self, reviews: List[Dict[str, Any]]) -> AdvancedMetrics:
        """Compute advanced statistical metrics."""
        try:
            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(reviews)
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(reviews)
            
            # Basic trend analysis
            trend_analysis = self._perform_trend_analysis(reviews)
            
            # Outlier detection
            outliers = self._detect_outliers(reviews)
            
            # Placeholder confidence intervals
            confidence_intervals = {
                "overall_score": (0.0, 10.0),
                "diversity": (0.0, 1.0)
            }
            
            return AdvancedMetrics(
                diversity_score=diversity_score,
                consensus_score=consensus_score,
                trend_analysis=trend_analysis,
                outlier_detection=outliers,
                confidence_intervals=confidence_intervals
            )
            
        except Exception as e:
            logger.warning(f"Advanced metrics computation failed: {e}")
            return AdvancedMetrics(
                diversity_score=0.0,
                consensus_score=0.0,
                trend_analysis={},
                outlier_detection=[],
                confidence_intervals={}
            )
    
    def _generate_recommendations_for_generation(self, patterns: List[PatternInsight], 
                                               correlations: Dict[str, float]) -> str:
        """Generate actionable recommendations for the Generation Agent."""
        recommendations = []
        
        # Analyze patterns for generation insights
        for pattern in patterns:
            if "diversity" in pattern.description.lower():
                recommendations.append(
                    "Increase topic diversity in literature search queries to generate more varied hypotheses."
                )
            elif "dominant theme" in pattern.description.lower():
                recommendations.append(
                    "Explore alternative research domains to reduce thematic concentration."
                )
        
        # Analyze correlations for generation insights
        high_correlations = [(k, v) for k, v in correlations.items() if abs(v) > 0.8]
        if high_correlations:
            recommendations.append(
                "Consider generating hypotheses that decouple highly correlated criteria for more independent evaluation."
            )
        
        if not recommendations:
            recommendations.append(
                "Continue current generation strategy - no significant issues detected."
            )
        
        return " ".join(recommendations)
    
    def _generate_recommendations_for_reflection(self, patterns: List[PatternInsight],
                                               correlations: Dict[str, float]) -> str:
        """Generate actionable recommendations for the Reflection Agent."""
        recommendations = []
        
        # Analyze patterns for reflection insights
        for pattern in patterns:
            if "low variance" in pattern.description.lower():
                recommendations.append(
                    "Encourage more discriminating evaluation to increase score variance and improve differentiation."
                )
            elif "under-utilized" in pattern.description.lower():
                recommendations.append(
                    "Emphasize underused evaluation criteria to ensure comprehensive assessment."
                )
            elif "consistently high" in pattern.description.lower():
                recommendations.append(
                    "Review scoring rubric to ensure appropriate use of full scale range."
                )
        
        # Check for evaluation consistency
        if len(correlations) > 0:
            avg_correlation = np.mean(list(correlations.values()))
            if avg_correlation > 0.9:
                recommendations.append(
                    "Criteria appear highly correlated - consider refining evaluation dimensions."
                )
        
        if not recommendations:
            recommendations.append(
                "Evaluation process appears well-calibrated - maintain current approach."
            )
        
        return " ".join(recommendations)
    
    def _assess_analysis_quality(self, reviews: List[Dict[str, Any]], 
                               correlations: Dict[str, float]) -> str:
        """Assess the overall quality of the meta-analysis."""
        n_reviews = len(reviews)
        n_correlations = len(correlations)
        
        if n_reviews < 5:
            return "low"
        elif n_reviews < 15 or n_correlations < 3:
            return "medium"
        else:
            return "high"
    
    def _calculate_average_similarity(self, texts: List[str]) -> float:
        """Calculate average pairwise similarity between texts."""
        if not self.text_analyzer or len(texts) < 2:
            return 0.0
        
        try:
            similarities = []
            for i, text1 in enumerate(texts):
                for text2 in texts[i+1:]:
                    sim = self.text_analyzer.compute_text_similarity(text1, text2)
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_review_variance(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate variance in scores across reviews for same hypothesis."""
        all_scores = []
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                scores = [score for score in review_data.values() 
                         if isinstance(score, (int, float))]
                all_scores.extend(scores)
        
        return np.var(all_scores) if all_scores else 0.0
    
    def _calculate_diversity_score(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate diversity score across reviews."""
        if len(reviews) < 2:
            return 0.0
        
        # Simple diversity based on score spread
        all_scores = []
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                scores = [score for score in review_data.values()
                         if isinstance(score, (int, float))]
                all_scores.extend(scores)
        
        if not all_scores:
            return 0.0
        
        # Normalize diversity to 0-1 range
        score_range = max(all_scores) - min(all_scores)
        max_possible_range = 10.0  # Assuming 0-10 scale
        
        return min(1.0, score_range / max_possible_range)
    
    def _calculate_consensus_score(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate consensus score (inverse of diversity)."""
        diversity = self._calculate_diversity_score(reviews)
        return 1.0 - diversity
    
    def _perform_trend_analysis(self, reviews: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform basic trend analysis on scores."""
        trends = {}
        
        # Group by criteria and analyze trends
        criteria_scores = defaultdict(list)
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                for criterion, score in review_data.items():
                    if isinstance(score, (int, float)):
                        criteria_scores[criterion].append(score)
        
        for criterion, scores in criteria_scores.items():
            if len(scores) >= 3:
                # Simple linear trend (positive/negative slope)
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0]
                trends[f"{criterion}_trend"] = float(slope)
        
        return trends
    
    def _detect_outliers(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """Detect outlier reviews based on score patterns."""
        outliers = []
        
        # Collect all scores to determine global statistics
        all_scores = []
        for review in reviews:
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                scores = [score for score in review_data.values()
                         if isinstance(score, (int, float))]
                all_scores.extend(scores)
        
        if len(all_scores) < 5:
            return outliers
        
        # Calculate statistics
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        # Identify outlier reviews
        for i, review in enumerate(reviews):
            review_data = review.get('review', {})
            if isinstance(review_data, dict):
                scores = [score for score in review_data.values()
                         if isinstance(score, (int, float))]
                
                if scores:
                    review_mean = np.mean(scores)
                    # Check if review mean is more than 2 std deviations away
                    if abs(review_mean - mean_score) > 2 * std_score:
                        outliers.append(f"review_{i}")
        
        return outliers
    
    def _create_empty_state(self) -> MetaReviewState:
        """Create an empty meta-review state for error cases."""
        return MetaReviewState(
            pattern_insights=[],
            criterion_correlations={},
            actionable_for_generation="Insufficient data for meta-analysis",
            actionable_for_reflection="Insufficient data for meta-analysis",
            clustering_metrics=ClusteringMetrics(
                n_samples=0,
                n_clusters_requested=0,
                n_clusters_actual=0,
                clustering_method="none"
            ),
            created_at=create_timestamp(),
            analysis_quality="low"
        )


# Factory function
def create_meta_review_agent(**kwargs) -> EnhancedMetaReviewAgent:
    """Create a meta-review agent with optional parameters."""
    return EnhancedMetaReviewAgent(**kwargs)


# Testing function
def test_meta_review_agent():
    """Test the Meta-Review Agent with sample data."""
    logger.info("Testing Meta-Review Agent...")
    
    # Create sample reflection state with reviews
    sample_reviews = [
        {
            "hypothesis_id": "hyp_1",
            "hypothesis_content": "AI-powered drug discovery using machine learning",
            "review": {
                "novelty": 8.5,
                "feasibility": 7.0,
                "rigor": 6.5,
                "impact": 9.0,
                "testability": 7.5
            }
        },
        {
            "hypothesis_id": "hyp_2", 
            "hypothesis_content": "Quantum computing for molecular simulation",
            "review": {
                "novelty": 9.0,
                "feasibility": 5.0,
                "rigor": 8.0,
                "impact": 8.5,
                "testability": 6.0
            }
        },
        {
            "hypothesis_id": "hyp_3",
            "hypothesis_content": "CRISPR gene editing for cancer treatment",
            "review": {
                "novelty": 7.0,
                "feasibility": 8.0,
                "rigor": 8.5,
                "impact": 9.5,
                "testability": 8.0
            }
        }
    ]
    
    # Create mock reflection state
    class MockReflectionState:
        def __init__(self, reviews):
            self.hypothesis_reviews = reviews
    
    reflection_state = MockReflectionState(sample_reviews)
    
    # Test the agent
    agent = EnhancedMetaReviewAgent()
    result = agent.run(reflection_state)
    
    logger.info(f"✅ Meta-Review Agent test completed:")
    logger.info(f"   - Pattern insights: {len(result.pattern_insights)}")
    logger.info(f"   - Correlations found: {len(result.criterion_correlations)}")
    logger.info(f"   - Analysis quality: {result.analysis_quality}")
    logger.info(f"   - Generation recommendations: {result.actionable_for_generation[:100]}...")
    logger.info(f"   - Reflection recommendations: {result.actionable_for_reflection[:100]}...")
    
    return result


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_result = test_meta_review_agent()
    print("Meta-Review Agent test completed successfully!")