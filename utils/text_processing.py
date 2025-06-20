"""
Text Processing Utilities

Provides text processing capabilities including embeddings, clustering,
and text analysis for the AI Co-Scientist system.
"""

import re
import logging
import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
from dataclasses import dataclass

# Import ML libraries with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    AgglomerativeClustering = None
    cosine_similarity = None
    silhouette_score = None
    SKLEARN_AVAILABLE = False

try:
    from core.config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result from text embedding operation"""
    embeddings: np.ndarray
    texts: List[str]
    model_name: str
    processing_time: float

@dataclass
class ClusteringResult:
    """Result from clustering operation"""
    cluster_labels: np.ndarray
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    cluster_texts: Optional[Dict[int, List[str]]] = None
    clustering_method: str = "unknown"

class TextEmbedding:
    """Handles text embedding using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize text embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for text embedding")
        
        if CONFIG_AVAILABLE:
            config = get_config()
            self.model_name = model_name or config.model.sentence_transformer_model
        else:
            self.model_name = model_name or "all-MiniLM-L6-v2"  # Default model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_texts(self, texts: List[str], show_progress: bool = False) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        import time
        
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        start_time = time.time()
        
        try:
            # Clean and preprocess texts
            cleaned_texts = [self.preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts, 
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated embeddings for {len(texts)} texts in {processing_time:.2f}s")
            
            return EmbeddingResult(
                embeddings=embeddings,
                texts=cleaned_texts,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray = None) -> np.ndarray:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (if None, compute self-similarity)
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings1, embeddings2.T) / (
            np.linalg.norm(embeddings1, axis=1, keepdims=True) * 
            np.linalg.norm(embeddings2, axis=1, keepdims=True).T
        )
        
        return similarity_matrix
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text for better embedding quality"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation that aids understanding
        text = re.sub(r'[^\w\s.,!?;:\-()]', ' ', text)
        
        return text

class TextClustering:
    """Handles text clustering using various algorithms"""
    
    def __init__(self):
        """Initialize text clustering"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, clustering capabilities limited")
    
    def cluster_texts(self, texts: List[str], n_clusters: int = None, 
                     method: str = 'agglomerative', distance_threshold: float = 0.5,
                     embeddings: np.ndarray = None) -> ClusteringResult:
        """
        Cluster texts using specified method
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters (if None, determined automatically)
            method: Clustering method ('agglomerative', 'tfidf')
            distance_threshold: Distance threshold for agglomerative clustering
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            ClusteringResult with cluster information
        """
        if not texts:
            raise ValueError("No texts provided for clustering")
        
        if method == 'agglomerative' and embeddings is not None:
            return self._cluster_with_embeddings(texts, embeddings, n_clusters, distance_threshold)
        elif method == 'tfidf':
            return self._cluster_with_tfidf(texts, n_clusters)
        else:
            # Fallback to simple clustering
            return self._simple_clustering(texts, n_clusters or 3)
    
    def _cluster_with_embeddings(self, texts: List[str], embeddings: np.ndarray, 
                               n_clusters: int = None, distance_threshold: float = 0.5) -> ClusteringResult:
        """Cluster using pre-computed embeddings"""
        if not SKLEARN_AVAILABLE:
            return self._simple_clustering(texts, n_clusters or 3)
        
        try:
            # Use agglomerative clustering with cosine distance
            if n_clusters is None:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='cosine',
                    linkage='average'
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
            
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Compute silhouette score if possible
            silhouette = None
            if len(set(cluster_labels)) > 1:
                try:
                    silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')
                except Exception:
                    pass
            
            # Organize texts by cluster (store indices, not actual texts)
            cluster_texts = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(i)
            
            return ClusteringResult(
                cluster_labels=cluster_labels,
                n_clusters=len(set(cluster_labels)),
                silhouette_score=silhouette,
                cluster_texts=cluster_texts,
                clustering_method="agglomerative_embeddings"
            )
            
        except Exception as e:
            logger.warning(f"Embedding-based clustering failed: {e}, using fallback")
            return self._simple_clustering(texts, n_clusters or 3)
    
    def _cluster_with_tfidf(self, texts: List[str], n_clusters: int = None) -> ClusteringResult:
        """Cluster using TF-IDF features"""
        if not SKLEARN_AVAILABLE:
            return self._simple_clustering(texts, n_clusters or 3)
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_for_tfidf(text) for text in texts]
            
            # Create TF-IDF features
            vectorizer = TfidfVectorizer(
                max_features=min(100, len(texts) * 5),
                stop_words="english",
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 2),
                lowercase=True
            )
            
            X = vectorizer.fit_transform(processed_texts)
            
            if X.shape[1] == 0:
                logger.warning("No features extracted from TF-IDF, using simple clustering")
                return self._simple_clustering(texts, n_clusters or 3)
            
            # Determine number of clusters
            actual_n_clusters = n_clusters or min(len(texts) // 2, 5)
            actual_n_clusters = max(1, min(actual_n_clusters, len(texts) - 1))
            
            if actual_n_clusters >= len(texts):
                return self._simple_clustering(texts, n_clusters or 3)
            
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=actual_n_clusters,
                metric='cosine',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(X.toarray())
            
            # Organize texts by cluster (store indices, not actual texts)
            cluster_texts = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(i)
            
            return ClusteringResult(
                cluster_labels=cluster_labels,
                n_clusters=len(set(cluster_labels)),
                cluster_texts=cluster_texts,
                clustering_method="agglomerative_tfidf"
            )
            
        except Exception as e:
            logger.warning(f"TF-IDF clustering failed: {e}, using fallback")
            return self._simple_clustering(texts, n_clusters or 3)
    
    def _simple_clustering(self, texts: List[str], n_clusters: int) -> ClusteringResult:
        """Simple fallback clustering based on text length and word overlap"""
        try:
            if not texts:
                return ClusteringResult(cluster_labels=np.array([]), n_clusters=0, clustering_method="empty_input")
            
            if len(texts) == 1:
                return ClusteringResult(
                    cluster_labels=np.array([0]),
                    n_clusters=1,
                    cluster_texts={0: [0]},
                    clustering_method="single_text"
                )
            
            # Simple clustering based on word overlap
            cluster_labels = []
            clusters = []
            
            for text in texts:
                words = set(text.lower().split())
                
                # Find best matching cluster
                best_cluster = -1
                best_similarity = 0.0
                
                for i, cluster_words in enumerate(clusters):
                    if not words:
                        similarity = 0.0
                    else:
                        intersection = len(words.intersection(cluster_words))
                        union = len(words.union(cluster_words))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    if similarity > best_similarity and similarity > 0.2:  # Threshold
                        best_cluster = i
                        best_similarity = similarity
                
                # Assign to existing cluster or create new one
                if best_cluster >= 0 and len(clusters) < n_clusters:
                    cluster_labels.append(best_cluster)
                    clusters[best_cluster].update(words)
                elif len(clusters) < n_clusters:
                    cluster_labels.append(len(clusters))
                    clusters.append(words)
                else:
                    # Assign to first cluster if we've reached max clusters
                    cluster_labels.append(0)
                    if clusters:
                        clusters[0].update(words)
            
            # Organize texts by cluster (store indices, not actual texts)
            cluster_texts = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(i)
            
            return ClusteringResult(
                cluster_labels=np.array(cluster_labels),
                n_clusters=len(set(cluster_labels)),
                cluster_texts=cluster_texts,
                clustering_method="simple_text"
            )
            
        except Exception as e:
            logger.error(f"Simple clustering failed: {e}")
            # Ultimate fallback - single cluster
            return ClusteringResult(
                cluster_labels=np.zeros(len(texts)),
                n_clusters=1,
                cluster_texts={0: list(range(len(texts)))},
                clustering_method="fallback_single_cluster"
            )
    
    @staticmethod
    def _preprocess_for_tfidf(text: str) -> str:
        """Preprocess text for TF-IDF vectorization"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words
        words = [word for word in text.split() if len(word) >= 3]
        return ' '.join(words)

class TextAnalyzer:
    """Advanced text analysis utilities"""
    
    @staticmethod
    def extract_keywords(texts: List[str], top_k: int = 10) -> List[Tuple[str, int]]:
        """Extract top keywords from a collection of texts"""
        if not texts:
            return []
        
        # Combine all texts and extract words
        all_words = []
        for text in texts:
            if text:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'are', 'that', 'this', 'with', 'from', 'they', 'were', 
            'been', 'have', 'will', 'was', 'for', 'can', 'may', 'could', 'would',
            'should', 'not', 'but', 'our', 'all', 'any', 'one', 'two', 'new',
            'use', 'used', 'more', 'also', 'than', 'such', 'other', 'only'
        }
        
        filtered_counts = [(word, count) for word, count in word_counts.items() 
                          if word not in stop_words and len(word) >= 4]
        
        # Return top keywords
        return sorted(filtered_counts, key=lambda x: x[1], reverse=True)[:top_k]
    
    @staticmethod
    def compute_text_similarity(text1: str, text2: str) -> float:
        """Compute similarity between two texts using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def extract_sentences(text: str, max_length: int = 150) -> List[str]:
        """Extract meaningful sentences from text"""
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and clean sentences
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 < len(sentence) <= max_length:
                meaningful_sentences.append(sentence)
        
        return meaningful_sentences

# Factory functions for easy usage
def create_embedder(model_name: str = None) -> Optional[TextEmbedding]:
    """Create text embedder with error handling"""
    try:
        return TextEmbedding(model_name)
    except ImportError:
        logger.warning("sentence-transformers not available, embedder creation failed")
        return None
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        return None

def create_clusterer() -> TextClustering:
    """Create text clusterer"""
    return TextClustering()

def create_analyzer() -> TextAnalyzer:
    """Create text analyzer"""
    return TextAnalyzer()

# Convenience functions
def embed_and_cluster(texts: List[str], n_clusters: int = None, 
                     distance_threshold: float = 0.5) -> Tuple[Optional[EmbeddingResult], ClusteringResult]:
    """Convenience function to embed texts and cluster them"""
    embedder = create_embedder()
    clusterer = create_clusterer()
    
    embedding_result = None
    if embedder:
        try:
            embedding_result = embedder.embed_texts(texts)
            cluster_result = clusterer.cluster_texts(
                texts, n_clusters, 'agglomerative', distance_threshold, 
                embedding_result.embeddings
            )
        except Exception as e:
            logger.warning(f"Embedding-based clustering failed: {e}, using TF-IDF")
            cluster_result = clusterer.cluster_texts(texts, n_clusters, 'tfidf')
    else:
        cluster_result = clusterer.cluster_texts(texts, n_clusters, 'tfidf')
    
    return embedding_result, cluster_result

if __name__ == "__main__":
    # Test text processing utilities
    test_texts = [
        "Machine learning approaches for drug discovery in cancer research",
        "Deep learning models for predicting protein-drug interactions",
        "Artificial intelligence in personalized medicine and therapy",
        "Natural language processing for biomedical text mining",
        "Computer vision applications in medical imaging diagnosis"
    ]
    
    print("🔤 Testing Text Processing Utilities")
    print("=" * 50)
    
    # Test keyword extraction
    analyzer = create_analyzer()
    keywords = analyzer.extract_keywords(test_texts, top_k=5)
    print(f"📊 Top keywords: {keywords}")
    
    # Test clustering
    print("\n🔍 Testing text clustering...")
    embedding_result, cluster_result = embed_and_cluster(test_texts, n_clusters=3)
    
    if embedding_result:
        print(f"✅ Embeddings: {embedding_result.embeddings.shape}")
    
    print(f"✅ Clusters: {cluster_result.n_clusters}")
    print(f"📊 Cluster distribution: {np.bincount(cluster_result.cluster_labels)}")
    
    if cluster_result.cluster_texts:
        for cluster_id, cluster_texts in cluster_result.cluster_texts.items():
            print(f"\n📁 Cluster {cluster_id}:")
            for text in cluster_texts:
                print(f"   - {text[:60]}...")
    
    print("\n✅ Text processing utilities test completed!")