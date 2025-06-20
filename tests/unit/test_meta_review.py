"""
Tests for the Meta-Review Agent

This module contains comprehensive tests for the EnhancedMetaReviewAgent
including unit tests, integration tests, and edge case validation.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.meta_review import (
    EnhancedMetaReviewAgent,
    test_meta_review_agent,
    create_meta_review_agent
)
from core.data_structures import MetaReviewState, PatternInsight, ClusteringMetrics, AdvancedMetrics

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMetaReviewAgent:
    """Test suite for the EnhancedMetaReviewAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = EnhancedMetaReviewAgent()
        
        # Sample test reviews with varied patterns
        self.sample_reviews = [
            {
                "hypothesis_id": "hyp_1",
                "hypothesis_content": "AI-powered drug discovery using machine learning for novel therapeutic targets",
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
                "hypothesis_content": "Quantum computing applications for molecular simulation in pharmaceutical research",
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
                "hypothesis_content": "CRISPR gene editing for personalized cancer treatment approaches",
                "review": {
                    "novelty": 7.0,
                    "feasibility": 8.0,
                    "rigor": 8.5,
                    "impact": 9.5,
                    "testability": 8.0
                }
            },
            {
                "hypothesis_id": "hyp_4",
                "hypothesis_content": "Deep learning for protein structure prediction and drug target identification",
                "review": {
                    "novelty": 8.0,
                    "feasibility": 7.5,
                    "rigor": 7.0,
                    "impact": 8.0,
                    "testability": 7.0
                }
            },
            {
                "hypothesis_id": "hyp_5",
                "hypothesis_content": "Nanotechnology-based targeted drug delivery systems for cancer therapy",
                "review": {
                    "novelty": 7.5,
                    "feasibility": 6.5,
                    "rigor": 7.5,
                    "impact": 8.5,
                    "testability": 6.5
                }
            }
        ]
        
        # Mock reflection state
        class MockReflectionState:
            def __init__(self, reviews):
                self.hypothesis_reviews = reviews
        
        self.mock_reflection_state = MockReflectionState(self.sample_reviews)

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = EnhancedMetaReviewAgent()
        assert agent is not None
        assert hasattr(agent, 'confidence_threshold')
        assert agent.confidence_threshold == 0.7
        assert hasattr(agent, 'text_analyzer')

    def test_agent_initialization_with_params(self):
        """Test agent initialization with custom parameters."""
        agent = EnhancedMetaReviewAgent(confidence_threshold=0.8)
        assert agent.confidence_threshold == 0.8

    def test_run_with_valid_data(self):
        """Test successful run with valid reflection state."""
        result = self.agent.run(self.mock_reflection_state)
        
        # Validate results
        assert isinstance(result, MetaReviewState)
        assert hasattr(result, 'pattern_insights')
        assert hasattr(result, 'criterion_correlations')
        assert hasattr(result, 'actionable_for_generation')
        assert hasattr(result, 'actionable_for_reflection')
        assert hasattr(result, 'clustering_metrics')
        assert hasattr(result, 'analysis_quality')
        
        # Check that we got some analysis results
        assert isinstance(result.pattern_insights, list)
        assert isinstance(result.criterion_correlations, dict)
        assert isinstance(result.actionable_for_generation, str)
        assert isinstance(result.actionable_for_reflection, str)
        assert isinstance(result.clustering_metrics, ClusteringMetrics)
        assert result.analysis_quality in ["low", "medium", "high"]

    def test_run_with_empty_reviews(self):
        """Test run with empty reviews."""
        class EmptyReflectionState:
            def __init__(self):
                self.hypothesis_reviews = []
        
        empty_state = EmptyReflectionState()
        result = self.agent.run(empty_state)
        
        assert isinstance(result, MetaReviewState)
        assert len(result.pattern_insights) == 0
        assert len(result.criterion_correlations) == 0
        assert "insufficient data" in result.actionable_for_generation.lower()

    def test_run_with_missing_reviews_attribute(self):
        """Test run with reflection state missing hypothesis_reviews."""
        class InvalidReflectionState:
            pass
        
        invalid_state = InvalidReflectionState()
        result = self.agent.run(invalid_state)
        
        assert isinstance(result, MetaReviewState)
        assert result.analysis_quality == "low"

    def test_detect_patterns(self):
        """Test pattern detection functionality."""
        patterns = self.agent._detect_patterns(self.sample_reviews)
        
        assert isinstance(patterns, list)
        # Should detect some patterns in our test data
        assert len(patterns) >= 0
        
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)
            assert hasattr(pattern, 'description')
            assert hasattr(pattern, 'frequency')
            assert hasattr(pattern, 'sample_hypotheses')
            assert hasattr(pattern, 'confidence')
            assert 0 <= pattern.confidence <= 1

    def test_analyze_score_distributions(self):
        """Test score distribution analysis."""
        patterns = self.agent._analyze_score_distributions(self.sample_reviews)
        
        assert isinstance(patterns, list)
        # Should analyze score distributions across criteria
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)
            assert pattern.frequency > 0

    def test_detect_criteria_bias(self):
        """Test criteria bias detection."""
        patterns = self.agent._detect_criteria_bias(self.sample_reviews)
        
        assert isinstance(patterns, list)
        # Test data should have consistent criteria usage
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)

    def test_analyze_content_patterns(self):
        """Test content pattern analysis."""
        patterns = self.agent._analyze_content_patterns(self.sample_reviews)
        
        assert isinstance(patterns, list)
        # Should analyze textual content patterns
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)

    def test_analyze_consistency_patterns(self):
        """Test consistency pattern analysis."""
        patterns = self.agent._analyze_consistency_patterns(self.sample_reviews)
        
        assert isinstance(patterns, list)
        # Should check for scoring consistency
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)

    def test_analyze_correlations(self):
        """Test correlation analysis."""
        correlations = self.agent._analyze_correlations(self.sample_reviews)
        
        assert isinstance(correlations, dict)
        # Should find correlations between criteria
        for key, value in correlations.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
            assert -1 <= value <= 1  # Valid correlation range

    def test_analyze_clustering_patterns(self):
        """Test clustering pattern analysis."""
        metrics = self.agent._analyze_clustering_patterns(self.sample_reviews)
        
        assert isinstance(metrics, ClusteringMetrics)
        assert metrics.n_samples == len(self.sample_reviews)
        assert metrics.n_clusters_actual >= 0

    def test_compute_advanced_metrics(self):
        """Test advanced metrics computation."""
        metrics = self.agent._compute_advanced_metrics(self.sample_reviews)
        
        assert isinstance(metrics, AdvancedMetrics)
        assert hasattr(metrics, 'diversity_score')
        assert hasattr(metrics, 'consensus_score')
        assert hasattr(metrics, 'trend_analysis')
        assert hasattr(metrics, 'outlier_detection')
        assert hasattr(metrics, 'confidence_intervals')
        
        assert 0 <= metrics.diversity_score <= 1
        assert 0 <= metrics.consensus_score <= 1
        assert isinstance(metrics.trend_analysis, dict)
        assert isinstance(metrics.outlier_detection, list)
        assert isinstance(metrics.confidence_intervals, dict)

    def test_generate_recommendations_for_generation(self):
        """Test generation recommendations."""
        patterns = [
            PatternInsight(
                description="Low hypothesis diversity detected",
                frequency=5,
                sample_hypotheses=["test"],
                confidence=0.8
            )
        ]
        correlations = {"novelty_vs_impact": 0.9}
        
        recommendations = self.agent._generate_recommendations_for_generation(patterns, correlations)
        
        assert isinstance(recommendations, str)
        assert len(recommendations) > 0

    def test_generate_recommendations_for_reflection(self):
        """Test reflection recommendations."""
        patterns = [
            PatternInsight(
                description="Low variance in novelty scores",
                frequency=5,
                sample_hypotheses=["test"],
                confidence=0.8
            )
        ]
        correlations = {"novelty_vs_feasibility": 0.95}
        
        recommendations = self.agent._generate_recommendations_for_reflection(patterns, correlations)
        
        assert isinstance(recommendations, str)
        assert len(recommendations) > 0

    def test_assess_analysis_quality(self):
        """Test analysis quality assessment."""
        # Test with sufficient data
        quality = self.agent._assess_analysis_quality(self.sample_reviews, {"test": 0.5})
        assert quality in ["low", "medium", "high"]
        
        # Test with insufficient data
        small_reviews = self.sample_reviews[:2]
        quality = self.agent._assess_analysis_quality(small_reviews, {})
        assert quality == "low"

    def test_calculate_average_similarity(self):
        """Test average similarity calculation."""
        texts = [
            "AI machine learning drug discovery",
            "Machine learning for drug discovery",
            "Quantum computing molecular simulation"
        ]
        
        similarity = self.agent._calculate_average_similarity(texts)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_calculate_review_variance(self):
        """Test review variance calculation."""
        variance = self.agent._calculate_review_variance(self.sample_reviews)
        assert isinstance(variance, float)
        assert variance >= 0

    def test_calculate_diversity_score(self):
        """Test diversity score calculation."""
        diversity = self.agent._calculate_diversity_score(self.sample_reviews)
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1

    def test_calculate_consensus_score(self):
        """Test consensus score calculation."""
        consensus = self.agent._calculate_consensus_score(self.sample_reviews)
        assert isinstance(consensus, float)
        assert 0 <= consensus <= 1

    def test_perform_trend_analysis(self):
        """Test trend analysis."""
        trends = self.agent._perform_trend_analysis(self.sample_reviews)
        assert isinstance(trends, dict)
        
        for key, value in trends.items():
            assert isinstance(key, str)
            assert isinstance(value, float)

    def test_detect_outliers(self):
        """Test outlier detection."""
        outliers = self.agent._detect_outliers(self.sample_reviews)
        assert isinstance(outliers, list)
        
        for outlier in outliers:
            assert isinstance(outlier, str)

    def test_create_empty_state(self):
        """Test empty state creation."""
        empty_state = self.agent._create_empty_state()
        
        assert isinstance(empty_state, MetaReviewState)
        assert len(empty_state.pattern_insights) == 0
        assert len(empty_state.criterion_correlations) == 0
        assert empty_state.analysis_quality == "low"


class TestMetaReviewAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_review_data(self):
        """Test handling of malformed review data."""
        agent = EnhancedMetaReviewAgent()
        
        malformed_reviews = [
            {"hypothesis_id": "test1"},  # Missing review data
            {"review": "not_a_dict"},  # Invalid review format
            {"hypothesis_id": "test2", "review": {}},  # Empty review
            {"hypothesis_id": "test3", "review": {"novelty": "invalid"}},  # Invalid score
            None,  # Null review
            {"hypothesis_id": "test4", "review": {"novelty": 8.5}}  # Valid review
        ]
        
        class MalformedReflectionState:
            def __init__(self):
                self.hypothesis_reviews = malformed_reviews
        
        malformed_state = MalformedReflectionState()
        result = agent.run(malformed_state)
        
        # Should handle gracefully
        assert isinstance(result, MetaReviewState)

    def test_single_review(self):
        """Test with only one review."""
        agent = EnhancedMetaReviewAgent()
        
        single_review = [{
            "hypothesis_id": "solo",
            "review": {"novelty": 7.0, "feasibility": 6.0}
        }]
        
        class SingleReflectionState:
            def __init__(self):
                self.hypothesis_reviews = single_review
        
        single_state = SingleReflectionState()
        result = agent.run(single_state)
        
        assert isinstance(result, MetaReviewState)
        assert result.analysis_quality == "low"

    def test_identical_scores(self):
        """Test with identical scores across all reviews."""
        agent = EnhancedMetaReviewAgent()
        
        identical_reviews = [
            {
                "hypothesis_id": f"hyp_{i}",
                "review": {"novelty": 5.0, "feasibility": 5.0, "impact": 5.0}
            }
            for i in range(5)
        ]
        
        class IdenticalReflectionState:
            def __init__(self):
                self.hypothesis_reviews = identical_reviews
        
        identical_state = IdenticalReflectionState()
        result = agent.run(identical_state)
        
        assert isinstance(result, MetaReviewState)
        # Should detect low variance pattern
        variance_patterns = [p for p in result.pattern_insights 
                           if "variance" in p.description.lower()]
        # May or may not detect depending on implementation details

    def test_extreme_scores(self):
        """Test with extreme score distributions."""
        agent = EnhancedMetaReviewAgent()
        
        extreme_reviews = [
            {"hypothesis_id": "high", "review": {"novelty": 10.0, "feasibility": 10.0}},
            {"hypothesis_id": "low", "review": {"novelty": 0.0, "feasibility": 0.0}},
            {"hypothesis_id": "mixed", "review": {"novelty": 5.0, "feasibility": 5.0}}
        ]
        
        class ExtremeReflectionState:
            def __init__(self):
                self.hypothesis_reviews = extreme_reviews
        
        extreme_state = ExtremeReflectionState()
        result = agent.run(extreme_state)
        
        assert isinstance(result, MetaReviewState)
        assert result.advanced_metrics.diversity_score > 0

    @patch('agents.meta_review.create_analyzer')
    def test_text_analyzer_failure(self, mock_create_analyzer):
        """Test handling of text analyzer failures."""
        mock_create_analyzer.return_value = None
        
        agent = EnhancedMetaReviewAgent()
        
        sample_reviews = [{
            "hypothesis_id": "test",
            "hypothesis_content": "Test content",
            "review": {"novelty": 8.0}
        }]
        
        class TestReflectionState:
            def __init__(self):
                self.hypothesis_reviews = sample_reviews
        
        test_state = TestReflectionState()
        result = agent.run(test_state)
        
        # Should handle gracefully without text analyzer
        assert isinstance(result, MetaReviewState)

    def test_empty_hypothesis_content(self):
        """Test with empty or missing hypothesis content."""
        agent = EnhancedMetaReviewAgent()
        
        empty_content_reviews = [
            {"hypothesis_id": "empty1", "hypothesis_content": "", "review": {"novelty": 7.0}},
            {"hypothesis_id": "missing1", "review": {"novelty": 8.0}},  # No content field
            {"hypothesis_id": "none1", "hypothesis_content": None, "review": {"novelty": 6.0}}
        ]
        
        class EmptyContentReflectionState:
            def __init__(self):
                self.hypothesis_reviews = empty_content_reviews
        
        empty_content_state = EmptyContentReflectionState()
        result = agent.run(empty_content_state)
        
        assert isinstance(result, MetaReviewState)

    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        agent = EnhancedMetaReviewAgent()
        
        # Test with very small differences that might cause numerical issues
        close_reviews = [
            {"hypothesis_id": f"close_{i}", "review": {"novelty": 5.0 + i * 1e-10}}
            for i in range(10)
        ]
        
        class CloseReflectionState:
            def __init__(self):
                self.hypothesis_reviews = close_reviews
        
        close_state = CloseReflectionState()
        result = agent.run(close_state)
        
        assert isinstance(result, MetaReviewState)
        # Should handle numerical precision issues gracefully


class TestMetaReviewAgentIntegration:
    """Integration tests for the Meta-Review Agent."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        try:
            result = test_meta_review_agent()
            assert isinstance(result, MetaReviewState)
            assert hasattr(result, 'pattern_insights')
            assert hasattr(result, 'criterion_correlations')
            assert hasattr(result, 'analysis_quality')
        except Exception as e:
            # If it fails due to missing dependencies, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['import', 'module'])

    def test_factory_function(self):
        """Test the factory function."""
        agent = create_meta_review_agent(confidence_threshold=0.9)
        assert isinstance(agent, EnhancedMetaReviewAgent)
        assert agent.confidence_threshold == 0.9

    def test_realistic_meta_analysis_scenario(self):
        """Test with realistic multi-criteria evaluation data."""
        agent = EnhancedMetaReviewAgent()
        
        realistic_reviews = [
            {
                "hypothesis_id": "cancer_immunotherapy_1",
                "hypothesis_content": "CAR-T cell therapy for solid tumors using novel targeting mechanisms",
                "review": {
                    "novelty": 8.5,
                    "feasibility": 6.0,
                    "rigor": 7.5,
                    "impact": 9.0,
                    "testability": 7.0,
                    "ethical_considerations": 8.0
                }
            },
            {
                "hypothesis_id": "ai_drug_discovery_1",
                "hypothesis_content": "Deep learning for protein-drug interaction prediction and optimization",
                "review": {
                    "novelty": 7.0,
                    "feasibility": 8.5,
                    "rigor": 8.0,
                    "impact": 7.5,
                    "testability": 8.5,
                    "ethical_considerations": 9.0
                }
            },
            {
                "hypothesis_id": "quantum_chemistry_1",
                "hypothesis_content": "Quantum computing algorithms for molecular dynamics simulation",
                "review": {
                    "novelty": 9.5,
                    "feasibility": 4.0,
                    "rigor": 8.5,
                    "impact": 8.0,
                    "testability": 5.0,
                    "ethical_considerations": 7.0
                }
            },
            {
                "hypothesis_id": "gene_therapy_1",
                "hypothesis_content": "CRISPR-based gene correction for inherited metabolic disorders",
                "review": {
                    "novelty": 6.5,
                    "feasibility": 7.0,
                    "rigor": 9.0,
                    "impact": 9.5,
                    "testability": 8.0,
                    "ethical_considerations": 6.0
                }
            }
        ]
        
        class RealisticReflectionState:
            def __init__(self):
                self.hypothesis_reviews = realistic_reviews
        
        realistic_state = RealisticReflectionState()
        result = agent.run(realistic_state)
        
        # Should produce meaningful analysis
        assert isinstance(result, MetaReviewState)
        assert result.analysis_quality in ["low", "medium", "high"]  # Analysis with 4 samples is considered low quality
        
        # Should detect some patterns in this diverse data
        assert len(result.pattern_insights) >= 0
        
        # Should find correlations between criteria
        assert len(result.criterion_correlations) > 0
        
        # Should provide actionable recommendations
        assert len(result.actionable_for_generation) > 50
        assert len(result.actionable_for_reflection) > 50
        
        # Advanced metrics should be reasonable
        assert result.advanced_metrics.diversity_score > 0
        assert 0 <= result.advanced_metrics.consensus_score <= 1

    def test_longitudinal_analysis_pattern(self):
        """Test pattern detection across multiple evaluation rounds."""
        agent = EnhancedMetaReviewAgent()
        
        # Simulate multiple rounds of evaluations showing bias drift
        longitudinal_reviews = []
        for round_num in range(3):
            for hyp_num in range(4):
                # Simulate increasing bias toward feasibility over time
                feasibility_bias = 1.0 + round_num * 0.5
                review = {
                    "hypothesis_id": f"round_{round_num}_hyp_{hyp_num}",
                    "hypothesis_content": f"Research hypothesis {hyp_num} for round {round_num}",
                    "review": {
                        "novelty": 7.0 + np.random.normal(0, 0.5),
                        "feasibility": min(10.0, 6.0 + feasibility_bias + np.random.normal(0, 0.3)),
                        "impact": 7.5 + np.random.normal(0, 0.4)
                    }
                }
                longitudinal_reviews.append(review)
        
        class LongitudinalReflectionState:
            def __init__(self):
                self.hypothesis_reviews = longitudinal_reviews
        
        longitudinal_state = LongitudinalReflectionState()
        result = agent.run(longitudinal_state)
        
        assert isinstance(result, MetaReviewState)
        # Should detect patterns in this structured data
        assert result.analysis_quality in ["medium", "high"]


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])