"""
Simple test for cross-encoder refinement using KMeans clustering.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import cross-encoder refinement component
from Clustering.CrossEncoder.refinement import ClusterRefiner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockReranker:
    """Mock cross-encoder reranker for testing."""
    
    def __init__(self, vectors):
        self.vectors = vectors
    
    def compute_similarity(self, queries, passages):
        """Compute similarity between queries and passages using vector similarity + noise.
        
        Args:
            queries: List of query texts
            passages: List of passage texts
            
        Returns:
            List of similarity scores between 0 and 1
        """
        results = []
        
        for query, passage in zip(queries, passages):
            try:
                # Extract indices from text identifiers
                idx1 = int(query.split('_')[-1].split(':')[0])
                idx2 = int(passage.split('_')[-1].split(':')[0])
                
                # Compute cosine similarity with some noise
                vec1 = self.vectors[idx1]
                vec2 = self.vectors[idx2]
                
                # Add some noise to differentiate from pure embedding similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                noise = np.random.normal(0, 0.1)
                similarity = min(1.0, max(0.0, similarity + noise))
                
                results.append(float(similarity))
            except:
                # Fallback for parsing errors
                results.append(float(np.random.uniform(0.3, 0.7)))
        
        return results

def load_test_data(n_samples=200, n_clusters=5):
    """Load test data from 20 newsgroups."""
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space', 'talk.politics.guns']
    categories = categories[:n_clusters]
    
    # Load data
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Limit to requested number of samples
    n_samples = min(n_samples, len(newsgroups.data))
    texts = newsgroups.data[:n_samples]
    true_labels = newsgroups.target[:n_samples]
    
    # Create text IDs for cross-encoder lookup
    texts = [f"text_{i}: {text[:50].replace(chr(10), ' ')}..." for i, text in enumerate(texts)]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100)
    vectors = vectorizer.fit_transform(newsgroups.data[:n_samples]).toarray()
    
    # Normalize vectors
    vectors = normalize(vectors)
    
    logger.info(f"Loaded {len(texts)} texts with {len(categories)} categories")
    
    return texts, vectors, true_labels

def perform_kmeans_clustering(vectors, n_clusters=5):
    """Perform KMeans clustering."""
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(vectors)
    
    # Get cluster assignments
    labels = kmeans.labels_
    
    # Prepare cluster info in the same format expected by the refiner
    clusters = []
    for i in range(n_clusters):
        members = np.where(labels == i)[0]
        cluster = {
            'id': i,
            'size': len(members),
            'members': members.tolist(),
            'centroid': kmeans.cluster_centers_[i]
        }
        clusters.append(cluster)
    
    return labels, clusters

def visualize_clusters(vectors, labels, title):
    """Create a simple 2D visualization of clusters."""
    # Reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    # Load test data
    logger.info("Loading test data...")
    texts, vectors, true_labels = load_test_data(n_samples=200, n_clusters=5)
    
    # Perform KMeans clustering
    logger.info("Performing KMeans clustering...")
    labels, clusters = perform_kmeans_clustering(vectors, n_clusters=5)
    
    # Initialize mock reranker
    logger.info("Initializing mock reranker...")
    reranker = MockReranker(vectors)
    
    # Initialize cluster refiner
    logger.info("Initializing cluster refiner...")
    refiner = ClusterRefiner(
        reranker=reranker,
        embedding_weight=0.5,
        cross_encoder_weight=0.5,
        batch_size=32,
        confidence_threshold=0.6
    )
    
    # Visualize initial clustering
    logger.info("Visualizing initial clustering...")
    fig1 = visualize_clusters(vectors, labels, "Initial KMeans Clustering")
    
    # Refine clusters
    logger.info("Refining clusters with cross-encoder...")
    refined_labels, refined_clusters, refinement_metrics = refiner.refine_clusters(
        clusters=clusters,
        labels=labels,
        vectors=vectors,
        texts=texts,
        refine_method="borderline"
    )
    
    # Visualize refined clustering
    logger.info("Visualizing refined clustering...")
    fig2 = visualize_clusters(vectors, refined_labels, "Cross-Encoder Refined Clustering")
    
    # Create output directory
    output_dir = "cross_encoder_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plots
    fig1.savefig(os.path.join(output_dir, "initial_clustering.png"), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "refined_clustering.png"), dpi=300, bbox_inches='tight')
    
    # Count points that changed clusters
    changes = sum(1 for i, j in zip(labels, refined_labels) if i != j)
    change_percent = changes / len(labels) * 100
    logger.info(f"Points that changed clusters: {changes} ({change_percent:.2f}%)")
    
    # Show results
    logger.info("Refinement complete!")
    logger.info(f"Results saved to {output_dir}")
    
    # Show plots (comment out if running headless)
    plt.show()

if __name__ == "__main__":
    main()
