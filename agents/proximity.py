"""
Proximity Agent - Hypothesis Clustering & Deduplication

This module implements the ProximityAgent that performs semantic deduplication
of hypotheses using sentence embeddings and clustering to maintain intellectual diversity.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.data_structures import create_timestamp, ProximityState, HypothesisCluster
from utils.text_processing import create_embedder, TextClustering

logger = logging.getLogger(__name__)


class ProximityAgent:
    """
    Clusters similar hypotheses and removes duplicates to maintain idea diversity.
    
    Uses sentence embeddings and agglomerative clustering to identify semantically
    similar hypotheses and selects representatives from each cluster.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 distance_threshold: float = 0.25, 
                 min_cluster_size: int = 1):
        """
        Initialize the Proximity Agent.

        Args:
            model_name: The sentence-transformer model to use for embeddings
            distance_threshold: The clustering distance threshold (lower = more clusters)
            min_cluster_size: Minimum size for a cluster to be considered valid
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        
        # Initialize embedder (will be created lazily)
        self.embedder = None
        self.clusterer = TextClustering()
        
        logger.info(f"ProximityAgent initialized with model '{model_name}' "
                   f"and distance threshold {distance_threshold}")

    def _get_embedder(self):
        """Get or create the embedder (lazy initialization)."""
        if self.embedder is None:
            self.embedder = create_embedder(self.model_name)
            if self.embedder is None:
                logger.warning("Could not create sentence transformer embedder, using fallback")
        return self.embedder

    def run(self, proposals: List[Dict[str, Any]]) -> ProximityState:
        """
        Run the complete clustering and deduplication process.

        Args:
            proposals: List of hypothesis dictionaries with 'id' and 'content' keys

        Returns:
            ProximityState object containing clustering results and unique hypotheses
        """
        if not proposals or len(proposals) < 2:
            logger.warning("Not enough hypotheses to run proximity analysis")
            state = ProximityState(
                unique_hypotheses=proposals,
                original_count=len(proposals),
                removed_count=0,
                analysis_timestamp=create_timestamp(),
                clustering_method="insufficient_data"
            )
            return state

        logger.info(f"Running proximity analysis on {len(proposals)} hypotheses")

        # Step 1: Extract text content for embedding
        texts = [p.get('content', '') if p and isinstance(p, dict) else '' for p in proposals]
        
        # Step 2: Generate embeddings
        embeddings = self._embed_hypotheses(texts)
        
        if embeddings is None:
            # Fallback to simple text-based clustering if embeddings fail
            return self._fallback_clustering(proposals)

        # Step 3: Cluster the embeddings  
        cluster_result = self._cluster_hypotheses(texts, embeddings)
        
        # Step 4: Select representatives and build final state
        proximity_state = self._select_representatives(
            proposals, cluster_result, embeddings
        )

        proximity_state.analysis_timestamp = create_timestamp()
        proximity_state.original_count = len(proposals)
        proximity_state.removed_count = len(proposals) - len(proximity_state.unique_hypotheses)
        
        # Calculate diversity score
        proximity_state.diversity_score = self._calculate_diversity_score(proximity_state)

        logger.info(f"Proximity analysis complete: {len(proximity_state.clusters)} clusters, "
                   f"{proximity_state.removed_count} hypotheses removed, "
                   f"{len(proximity_state.unique_hypotheses)} unique hypotheses retained")

        return proximity_state

    def _embed_hypotheses(self, texts: List[str]) -> Optional[np.ndarray]:
        """Convert hypothesis text into numerical embeddings."""
        embedder = self._get_embedder()
        
        if embedder is None:
            logger.warning("No embedder available, will use fallback clustering")
            return None

        try:
            embedding_result = embedder.embed_texts(texts, show_progress=False)
            logger.debug(f"Generated embeddings with shape: {embedding_result.embeddings.shape}")
            return embedding_result.embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def _cluster_hypotheses(self, texts: List[str], 
                           embeddings: np.ndarray) -> Any:
        """Group similar hypothesis embeddings using clustering."""
        try:
            # Use text processing utility for clustering
            cluster_result = self.clusterer.cluster_texts(
                texts=texts,
                method='agglomerative',
                distance_threshold=self.distance_threshold,
                embeddings=embeddings
            )
            
            logger.debug(f"Clustering produced {cluster_result.n_clusters} clusters")
            return cluster_result
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback to simple clustering
            return self.clusterer.cluster_texts(
                texts=texts,
                method='tfidf',
                n_clusters=min(len(texts) // 2, 5)
            )

    def _select_representatives(self, proposals: List[Dict[str, Any]], 
                              cluster_result: Any,
                              embeddings: Optional[np.ndarray]) -> ProximityState:
        """
        Select the best representative from each cluster.
        
        Strategy: Choose the hypothesis closest to the cluster centroid
        """
        state = ProximityState(clustering_method=cluster_result.clustering_method)
        
        if not hasattr(cluster_result, 'cluster_texts') or not cluster_result.cluster_texts:
            # Fallback: treat each hypothesis as its own cluster
            logger.warning("No valid clusters found, treating each hypothesis as unique")
            # Filter out invalid proposals
            valid_proposals = [p for p in proposals if p and isinstance(p, dict)]
            state.unique_hypotheses = valid_proposals
            state.clusters = [
                HypothesisCluster(
                    cluster_id=i,
                    hypothesis_ids=[prop.get('id', f'unknown_{i}')],
                    representative_id=prop.get('id', f'unknown_{i}'),
                    representative_text=prop.get('content', ''),
                    similarity_score=1.0,
                    cluster_size=1
                )
                for i, prop in enumerate(valid_proposals)
            ]
            return state

        # Process each cluster
        for cluster_id, cluster_indices in cluster_result.cluster_texts.items():
            if not cluster_indices:
                continue
                
            # Get proposals for this cluster
            cluster_proposals = []
            cluster_embeddings = []
            
            for text_idx in cluster_indices:
                if text_idx < len(proposals) and proposals[text_idx] and isinstance(proposals[text_idx], dict):
                    cluster_proposals.append(proposals[text_idx])
                    if embeddings is not None and text_idx < len(embeddings):
                        cluster_embeddings.append(embeddings[text_idx])

            if not cluster_proposals:
                continue

            # Select representative
            if len(cluster_proposals) == 1:
                representative_idx = 0
                avg_similarity = 1.0
            else:
                representative_idx, avg_similarity = self._find_best_representative(
                    cluster_embeddings if cluster_embeddings else None,
                    cluster_proposals
                )

            representative = cluster_proposals[representative_idx]
            
            # Add to unique hypotheses
            state.unique_hypotheses.append(representative)
            
            # Create cluster info
            cluster_info = HypothesisCluster(
                cluster_id=cluster_id,
                hypothesis_ids=[p.get('id', f'unknown_{i}') for i, p in enumerate(cluster_proposals)],
                representative_id=representative.get('id', f'rep_{cluster_id}'),
                representative_text=representative.get('content', ''),
                similarity_score=float(avg_similarity),
                cluster_size=len(cluster_proposals)
            )
            state.clusters.append(cluster_info)

        return state

    def _find_best_representative(self, cluster_embeddings: Optional[List], 
                                cluster_proposals: List[Dict[str, Any]]) -> tuple:
        """
        Find the best representative from a cluster.
        
        Returns: (representative_index, average_similarity)
        """
        if not cluster_embeddings or len(cluster_embeddings) != len(cluster_proposals):
            # Fallback: select by content length (middle-sized content often most representative)
            lengths = [len(p.get('content', '')) for p in cluster_proposals]
            median_length = np.median(lengths)
            representative_idx = min(range(len(lengths)), 
                                   key=lambda i: abs(lengths[i] - median_length))
            return representative_idx, 0.8

        try:
            # Convert to numpy array
            embeddings_array = np.array(cluster_embeddings)
            
            # Calculate centroid
            centroid = np.mean(embeddings_array, axis=0)
            
            # Find closest to centroid (highest cosine similarity)
            similarities = []
            for emb in embeddings_array:
                # Cosine similarity
                sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
                similarities.append(sim)
            
            representative_idx = np.argmax(similarities)
            avg_similarity = np.mean(similarities)
            
            return int(representative_idx), float(avg_similarity)
            
        except Exception as e:
            logger.warning(f"Representative selection failed: {e}, using fallback")
            # Fallback to first item
            return 0, 0.7

    def _fallback_clustering(self, proposals: List[Dict[str, Any]]) -> ProximityState:
        """Fallback clustering when embeddings are not available."""
        logger.info("Using fallback text-based clustering")
        
        texts = [p.get('content', '') if p and isinstance(p, dict) else '' for p in proposals]
        
        # Use simple text clustering
        cluster_result = self.clusterer.cluster_texts(
            texts=texts,
            method='tfidf',
            n_clusters=max(1, len(texts) // 3)  # Conservative clustering
        )
        
        return self._select_representatives(proposals, cluster_result, None)

    def _calculate_diversity_score(self, state: ProximityState) -> float:
        """
        Calculate a diversity score for the clustering result.
        
        Higher score = more diverse (many small clusters)
        Lower score = less diverse (few large clusters)
        """
        if not state.clusters:
            return 0.0
            
        try:
            # Use inverse of largest cluster proportion as diversity metric
            cluster_sizes = [cluster.cluster_size for cluster in state.clusters]
            total_items = sum(cluster_sizes)
            
            if total_items == 0:
                return 0.0
                
            # Normalized entropy-like measure
            proportions = [size / total_items for size in cluster_sizes]
            diversity = 1.0 - max(proportions)  # 1 - largest cluster proportion
            
            # Bonus for having more clusters
            cluster_bonus = min(0.2, len(state.clusters) / 10)
            
            return min(1.0, diversity + cluster_bonus)
            
        except Exception as e:
            logger.warning(f"Diversity calculation failed: {e}")
            return 0.5  # Default moderate diversity


def run_proximity_agent(generation_state) -> Any:
    """
    Integration function for the Proximity Agent.
    
    Args:
        generation_state: State object with generated_proposals attribute
        
    Returns:
        Modified generation_state with deduplicated proposals
    """
    logger.info("🌐 Engaging Proximity Agent...")
    
    # Get proposals from generation state
    proposals = getattr(generation_state, 'generated_proposals', [])
    if not proposals:
        logger.warning("No proposals found in generation_state for proximity analysis")
        return generation_state

    # Initialize and run the agent
    proximity_agent = ProximityAgent(distance_threshold=0.25)
    proximity_results = proximity_agent.run(proposals)

    # Update the generation state with unique hypotheses
    generation_state.generated_proposals = proximity_results.unique_hypotheses

    # Store proximity results for logging/meta-review
    if not hasattr(generation_state, 'proximity_results'):
        setattr(generation_state, 'proximity_results', [])
    generation_state.proximity_results.append(proximity_results)

    logger.info(f"✅ Proximity Agent finished: {proximity_results.removed_count} "
               f"redundant hypotheses removed, diversity score: {proximity_results.diversity_score:.3f}")
    
    return generation_state


# Factory function for easy instantiation
def create_proximity_agent(**kwargs) -> ProximityAgent:
    """Factory function to create a proximity agent with optional parameters."""
    return ProximityAgent(**kwargs)


# Testing function
def test_proximity_agent():
    """Test the Proximity Agent with sample hypotheses."""
    logger.info("Testing Proximity Agent...")
    
    # Create test hypotheses with clear clusters and duplicates
    test_hypotheses = [
        {"id": "hyp_a1", "content": "Targeting protein kinase C alpha will reduce neuroinflammation in Alzheimer's disease."},
        {"id": "hyp_a2", "content": "Inhibiting the PKC-alpha enzyme can alleviate brain inflammation associated with Alzheimer's."},
        {"id": "hyp_b1", "content": "Using gut microbiome modulation is a viable strategy to slow Parkinson's disease progression."},
        {"id": "hyp_c1", "content": "A novel nanoparticle delivery system can cross the blood-brain barrier to deliver drugs."},
        {"id": "hyp_a3", "content": "PKC-alpha inhibition is a key mechanism for controlling neuroinflammatory responses in Alzheimer's."},
        {"id": "hyp_b2", "content": "Altering gut bacteria composition could be a therapeutic method for slowing down Parkinson's."}
    ]

    try:
        # Test agent
        agent = ProximityAgent(distance_threshold=0.25)
        result = agent.run(test_hypotheses)

        # Validate results
        assert len(result.unique_hypotheses) <= len(test_hypotheses)
        assert result.removed_count >= 0
        assert len(result.clusters) > 0
        assert result.original_count == len(test_hypotheses)
        
        logger.info(f"Proximity Agent test completed successfully: "
                   f"{len(result.clusters)} clusters, {result.removed_count} removed, "
                   f"diversity score: {result.diversity_score:.3f}")
        return result

    except Exception as e:
        logger.error(f"Proximity Agent test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_proximity_agent()
    print(f"Test completed: {len(test_result.unique_hypotheses)} unique hypotheses from "
          f"{test_result.original_count} original hypotheses")