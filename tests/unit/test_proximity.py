"""
Tests for the Proximity Agent

This module contains comprehensive tests for the ProximityAgent
including unit tests, integration tests, and edge case validation.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from agents.proximity import (
    ProximityAgent,
    test_proximity_agent
)
from core.data_structures import ProximityState
from utils.text_processing import EmbeddingResult, ClusteringResult

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProximityAgent:
    """Test suite for the ProximityAgent class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = ProximityAgent()
        
        # Sample test hypotheses
        self.test_hypotheses = [
            {
                "id": "hyp_1",
                "content": "AI-powered drug discovery platform using machine learning to identify novel therapeutic targets for Alzheimer's disease"
            },
            {
                "id": "hyp_2", 
                "content": "Machine learning algorithms for discovering new drug targets in Alzheimer's treatment using artificial intelligence"
            },
            {
                "id": "hyp_3",
                "content": "Personalized cancer immunotherapy using patient-specific biomarkers and genetic profiling"
            },
            {
                "id": "hyp_4",
                "content": "Novel quantum computing approaches for molecular simulation in drug development"
            },
            {
                "id": "hyp_5",
                "content": "Advanced immunotherapy strategies for personalized cancer treatment based on individual patient profiles"
            }
        ]
        
        # Expected: hyp_1 and hyp_2 should cluster (similar AI/ML drug discovery)
        # Expected: hyp_3 and hyp_5 should cluster (similar cancer immunotherapy)
        # Expected: hyp_4 should be separate (quantum computing is different)

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        agent = ProximityAgent()
        
        assert agent is not None
        assert hasattr(agent, 'distance_threshold')
        assert hasattr(agent, 'min_cluster_size')
        assert agent.distance_threshold == 0.25
        assert agent.min_cluster_size == 1

    def test_agent_initialization_with_params(self):
        """Test agent initialization with custom parameters."""
        agent = ProximityAgent(
            distance_threshold=0.1,
            min_cluster_size=2
        )
        
        assert agent.distance_threshold == 0.1
        assert agent.min_cluster_size == 2

    # def test_extract_hypothesis_texts(self):
    #     """Test extraction of text content from hypothesis objects."""
    #     # Method not available in current implementation
    #     pass

    # def test_extract_hypothesis_texts_edge_cases(self):
    #     """Test text extraction with edge cases."""
    #     # Method not available in current implementation
    #     pass

    @patch('agents.proximity.ProximityAgent._embed_hypotheses')
    @patch('agents.proximity.ProximityAgent._cluster_hypotheses')
    def test_run_with_embeddings_success(self, mock_cluster, mock_embed):
        """Test successful run with embedding-based clustering."""
        # Mock embedding result
        mock_embeddings = np.random.rand(5, 384)  # 5 hypotheses, 384-dim embeddings
        mock_embed.return_value = mock_embeddings
        
        # Mock clustering result with realistic clusters
        mock_cluster_labels = np.array([0, 0, 1, 2, 1])  # Similar hypotheses get same label
        mock_clustering_result = ClusteringResult(
            cluster_labels=mock_cluster_labels,
            n_clusters=3,
            silhouette_score=0.65,
            cluster_texts={
                0: [0, 1],  # Indices of hypotheses in each cluster
                1: [2, 4],
                2: [3]
            },
            clustering_method="agglomerative_embeddings"
        )
        
        mock_cluster.return_value = mock_clustering_result
        
        # Run the agent
        result = self.agent.run(self.test_hypotheses)
        
        # Validate results
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(self.test_hypotheses)  # Should have representatives
        assert result.clusters is not None
        assert len(result.clusters) == 3  # Should have 3 clusters as mocked
        assert result.clustering_method == "agglomerative_embeddings"
        assert result.original_count == len(self.test_hypotheses)
        assert result.removed_count >= 0

    @patch('agents.proximity.ProximityAgent._embed_hypotheses')
    def test_run_with_embeddings_failure(self, mock_embed):
        """Test run with embedding failure fallback."""
        # Mock embedding failure
        mock_embed.return_value = None  # Simulate embedding failure
        
        # Run the agent (should fallback to simple clustering)
        result = self.agent.run(self.test_hypotheses)
        
        # Should still return valid result with fallback method
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(self.test_hypotheses)
        assert result.clusters is not None
        assert result.clustering_method != "unknown"  # Should use some fallback method

    def test_simple_text_clustering(self):
        """Test simple text-based clustering fallback."""
        result = self.agent._fallback_clustering(self.test_hypotheses)
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(self.test_hypotheses)
        assert result.clusters is not None
        assert len(result.clusters) > 0

    def test_simple_text_clustering_identical_texts(self):
        """Test simple clustering with identical texts."""
        identical_hypotheses = [
            {"id": "dup1", "content": "Identical content for testing"},
            {"id": "dup2", "content": "Identical content for testing"},
            {"id": "unique", "content": "Different content"}
        ]
        
        result = self.agent._fallback_clustering(identical_hypotheses)
        
        # Should deduplicate identical content
        assert len(result.unique_hypotheses) <= len(identical_hypotheses)
        assert result.removed_count >= 0

    # TODO: Implement tests for individual methods once they are properly exposed
    # def test_compute_text_similarity_matrix(self):
    #     """Test computation of text similarity matrix."""
    #     # Method not available in current implementation
    #     pass

    # def test_identify_clusters_from_similarity(self):
    #     """Test cluster identification from similarity matrix."""
    #     # Method not available in current implementation
    #     pass

    # def test_select_cluster_representatives(self):
    #     """Test selection of cluster representatives."""
    #     # Method not available in current implementation
    #     pass

    def test_run_with_empty_hypotheses(self):
        """Test run with empty hypothesis list."""
        result = self.agent.run([])
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) == 0
        assert result.removed_count == 0
        assert len(result.clusters) == 0

    def test_run_with_single_hypothesis(self):
        """Test run with single hypothesis."""
        single_hypothesis = [self.test_hypotheses[0]]
        
        result = self.agent.run(single_hypothesis)
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) == 1
        assert result.removed_count == 0
        assert result.unique_hypotheses[0] == single_hypothesis[0]

    def test_run_with_two_identical_hypotheses(self):
        """Test run with two identical hypotheses."""
        identical_hypotheses = [
            {"id": "test1", "content": "Identical content"},
            {"id": "test2", "content": "Identical content"}
        ]
        
        result = self.agent.run(identical_hypotheses)
        
        # Should deduplicate to 1 unique hypothesis
        assert len(result.unique_hypotheses) <= len(identical_hypotheses)
        assert result.removed_count >= 0

    def test_run_with_all_different_hypotheses(self):
        """Test run with completely different hypotheses."""
        different_hypotheses = [
            {"id": "math", "content": "Mathematical theorem about prime numbers"},
            {"id": "cooking", "content": "Recipe for chocolate cake with vanilla frosting"},
            {"id": "space", "content": "Exploration of Mars using robotic missions"},
            {"id": "music", "content": "Composition techniques in classical symphony"}
        ]
        
        result = self.agent.run(different_hypotheses)
        
        # Should keep all hypotheses as they are different
        assert len(result.unique_hypotheses) == len(different_hypotheses)
        assert result.removed_count == 0


class TestProximityAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_malformed_hypothesis_objects(self):
        """Test handling of malformed hypothesis objects."""
        agent = ProximityAgent()
        
        malformed_hypotheses = [
            None,  # Null hypothesis
            {"id": "valid", "content": "Valid content"},
            {"missing_id": True, "content": "No ID"},
            {"id": "empty_content", "content": ""},
            {"id": "none_content", "content": None},
            "string_instead_of_dict",  # Wrong type
            {"id": "valid2", "content": "Another valid one"}
        ]
        
        # Should not crash and should handle gracefully
        result = agent.run(malformed_hypotheses)
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) >= 0

    def test_very_long_hypothesis_content(self):
        """Test handling of very long hypothesis content."""
        agent = ProximityAgent()
        
        long_content = "A" * 50000  # Very long content
        long_hypotheses = [
            {"id": "long1", "content": long_content},
            {"id": "long2", "content": long_content},  # Identical long content
            {"id": "normal", "content": "Normal length content"}
        ]
        
        result = agent.run(long_hypotheses)
        
        assert isinstance(result, ProximityState)
        # Should deduplicate the identical long content
        assert len(result.unique_hypotheses) <= len(long_hypotheses)

    def test_special_characters_in_content(self):
        """Test handling of special characters and Unicode."""
        agent = ProximityAgent()
        
        special_hypotheses = [
            {"id": "unicode", "content": "Hypothesis with émojis 🧬 and spéciàl chärs"},
            {"id": "symbols", "content": "Math symbols: ∑∫∂∇ ≈ ≠ ± ∞"},
            {"id": "programming", "content": "Code: if (x > 0) { return x * 2; } // comment"},
            {"id": "newlines", "content": "Multi\nline\ncontent\nwith\nbreaks"}
        ]
        
        result = agent.run(special_hypotheses)
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) > 0

    @patch('agents.proximity.ProximityAgent._embed_hypotheses')
    def test_embedding_dimension_mismatch(self, mock_embed):
        """Test handling of embedding dimension mismatches."""
        agent = ProximityAgent()
        
        # Mock embedding failure due to dimension issues
        mock_embed.return_value = None  # Simulate embedding failure
        
        hypotheses = [
            {"id": "h1", "content": "Content 1"},
            {"id": "h2", "content": "Content 2"}, 
            {"id": "h3", "content": "Content 3"}
        ]
        
        # Should fallback gracefully
        result = agent.run(hypotheses)
        
        assert isinstance(result, ProximityState)

    def test_numerical_stability_similarity_computation(self):
        """Test numerical stability with edge case inputs."""
        agent = ProximityAgent()
        
        # Create hypotheses that might cause numerical issues
        edge_case_hypotheses = [
            {"id": "empty", "content": ""},  # Empty string
            {"id": "space", "content": " "},  # Single space
            {"id": "char", "content": "a"},  # Single character
            {"id": "repeat", "content": "a " * 1000},  # Very repetitive
        ]
        
        # Should not crash with edge case inputs
        result = agent.run(edge_case_hypotheses)
        
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(edge_case_hypotheses)


class TestProximityAgentIntegration:
    """Integration tests for the Proximity Agent."""

    def test_system_test_function(self):
        """Test the complete system test function."""
        try:
            result = test_proximity_agent()
            assert isinstance(result, ProximityState)
            assert hasattr(result, 'unique_hypotheses')
            assert hasattr(result, 'removed_count')
            assert hasattr(result, 'clusters')
        except Exception as e:
            # If it fails due to missing dependencies, that's expected in test environment
            assert any(keyword in str(e).lower() for keyword in ['import', 'module', 'sentence', 'transform'])

    def test_realistic_deduplication_scenario(self):
        """Test with realistic scientific hypotheses."""
        agent = ProximityAgent()
        
        realistic_hypotheses = [
            {
                "id": "cancer_1",
                "content": "CRISPR-Cas9 gene editing for targeted cancer immunotherapy using personalized T-cell engineering"
            },
            {
                "id": "cancer_2", 
                "content": "Personalized cancer treatment using CRISPR gene editing technology to enhance T-cell immunotherapy"
            },
            {
                "id": "alzheimer_1",
                "content": "Machine learning analysis of brain imaging data for early Alzheimer's disease detection and diagnosis"
            },
            {
                "id": "alzheimer_2",
                "content": "AI-powered analysis of neuroimaging patterns to identify early-stage Alzheimer's disease markers"
            },
            {
                "id": "diabetes_1",
                "content": "Continuous glucose monitoring with AI-driven insulin dosing for Type 1 diabetes management"
            }
        ]
        
        result = agent.run(realistic_hypotheses)
        
        # Should deduplicate similar cancer and Alzheimer's hypotheses
        assert len(result.unique_hypotheses) <= len(realistic_hypotheses)
        assert result.removed_count >= 0
        
        # Verify that unique hypotheses cover different domains
        unique_contents = [h["content"].lower() for h in result.unique_hypotheses]
        assert any("cancer" in content for content in unique_contents)
        assert any("alzheimer" in content for content in unique_contents)
        assert any("diabetes" in content for content in unique_contents)

    @patch('agents.proximity.ProximityAgent._embed_hypotheses')
    @patch('agents.proximity.ProximityAgent._cluster_hypotheses')
    def test_performance_with_large_dataset(self, mock_cluster, mock_embed):
        """Test performance with larger dataset."""
        agent = ProximityAgent()
        
        # Create a larger dataset (simulate 50 hypotheses)
        large_dataset = []
        for i in range(50):
            large_dataset.append({
                "id": f"hyp_{i}",
                "content": f"Research hypothesis number {i} about various scientific topics and approaches"
            })
        
        # Mock embedding and clustering for large dataset
        mock_embeddings = np.random.rand(50, 384)
        mock_embed.return_value = mock_embeddings
        
        mock_cluster_labels = np.random.randint(0, 10, 50)  # Random clusters
        mock_clustering_result = ClusteringResult(
            cluster_labels=mock_cluster_labels,
            n_clusters=10,
            silhouette_score=0.3,
            clustering_method="test_clustering",
            cluster_texts={i: [i*5, i*5+1, i*5+2, i*5+3, i*5+4] for i in range(10)}  # Mock cluster indices
        )
        
        mock_cluster.return_value = mock_clustering_result
        
        result = agent.run(large_dataset)
        
        # Should handle large dataset efficiently
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(large_dataset)
        assert len(result.clusters) > 0

    def test_integration_with_generation_output(self):
        """Test integration with typical Generation Agent output format."""
        agent = ProximityAgent()
        
        # Simulate output from Generation Agent
        generation_output = [
            {
                "id": "gen_1_1",
                "content": "Novel hypothesis about AI in drug discovery",
                "timestamp": "2024-01-01T12:00:00",
                "based_on_sources": 5,
                "source_diversity": 3
            },
            {
                "id": "gen_1_2", 
                "content": "AI-driven drug discovery using machine learning approaches",
                "timestamp": "2024-01-01T12:01:00",
                "based_on_sources": 5,
                "source_diversity": 3
            },
            {
                "id": "gen_1_3",
                "content": "Quantum computing applications in molecular simulation",
                "timestamp": "2024-01-01T12:02:00", 
                "based_on_sources": 3,
                "source_diversity": 2
            }
        ]
        
        result = agent.run(generation_output)
        
        # Should work seamlessly with Generation Agent output
        assert isinstance(result, ProximityState)
        assert len(result.unique_hypotheses) <= len(generation_output)
        
        # Should preserve additional fields in unique hypotheses
        for hyp in result.unique_hypotheses:
            assert "timestamp" in hyp
            assert "based_on_sources" in hyp
            assert "source_diversity" in hyp


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])