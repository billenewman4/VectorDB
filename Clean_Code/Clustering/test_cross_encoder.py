"""
Test script for cross-encoder refinement of clustering results.
This demonstrates how the cross-encoder can improve clustering quality
by refining the initial embedding-based clusters.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
import argparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import clustering components
from Clustering.Embedding.hdbscan_clusterer import HdbscanClusterer
from Clustering.CrossEncoder.refinement import ClusterRefiner
from Clustering.Analytics.visualization import ClusterVisualizer
from Clustering.Analytics.evaluation import ClusterEvaluator
from Clustering.Processing.embedding_preprocessing import EmbeddingPreprocessor
from Clustering.pipeline import ClusteringPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockReranker:
    """
    Mock cross-encoder reranker for testing.
    In a real application, you would use your actual cross-encoder from the Cross_Encoder module.
    """
    
    def __init__(self, vectors: np.ndarray):
        """
        Initialize mock reranker.
        
        Args:
            vectors: Embedding vectors to use for similarity calculation
        """
        self.vectors = vectors
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get indices of the texts (this is a hack for testing - in real implementation 
        # you would use your actual cross-encoder)
        try:
            idx1 = int(text1.split('_')[-1])
            idx2 = int(text2.split('_')[-1])
            
            # Calculate cosine similarity between vectors
            vec1 = self.vectors[idx1]
            vec2 = self.vectors[idx2]
            
            # Add some noise to differentiate from pure embedding similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Add some random noise to simulate different cross-encoder behavior
            noise = np.random.normal(0, 0.1)
            similarity = min(1.0, max(0.0, similarity + noise))
            
            return float(similarity)
        except:
            # Fallback to random similarity
            return float(np.random.uniform(0.3, 0.7))

def load_synthetic_data(n_samples: int = 300, n_clusters: int = 5) -> tuple:
    """
    Load synthetic text data based on 20 newsgroups dataset.
    
    Args:
        n_samples: Number of samples to load
        n_clusters: Number of clusters to use
        
    Returns:
        Tuple of (texts, vectors, true_labels)
    """
    # Get subset of 20 newsgroups data
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space', 'talk.politics.guns']
    categories = categories[:n_clusters]  # Limit to requested number of clusters
    
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
    
    # Create numeric ID for each text
    texts = [f"text_{i}: {text[:100].replace(chr(10), ' ')}..." for i, text in enumerate(texts)]
    
    # Create TF-IDF vectors as embedding approximation
    vectorizer = TfidfVectorizer(max_features=100)
    vectors = vectorizer.fit_transform(newsgroups.data[:n_samples]).toarray()
    
    logger.info(f"Loaded {len(texts)} texts with {len(categories)} categories")
    logger.info(f"Vector shape: {vectors.shape}")
    
    return texts, vectors, true_labels

def run_experiment(args) -> None:
    """
    Run cross-encoder refinement experiment.
    
    Args:
        args: Command-line arguments
    """
    # Load test data
    logger.info("Loading test data...")
    texts, vectors, true_labels = load_synthetic_data(
        n_samples=args.n_samples,
        n_clusters=args.n_clusters
    )
    
    # Preprocess vectors
    logger.info("Preprocessing vectors...")
    preprocessor = EmbeddingPreprocessor()
    processed_vectors = preprocessor.normalize(vectors)
    
    # Initialize clusterer
    logger.info("Initializing clusterer...")
    clusterer = HdbscanClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )
    
    # Perform initial clustering
    logger.info("Performing initial clustering...")
    embedding_results = clusterer.fit_predict(processed_vectors, texts)
    embedding_labels = embedding_results["labels"]
    embedding_clusters = embedding_results["clusters"]
    
    # Calculate metrics for initial clustering
    logger.info("Evaluating initial clustering...")
    initial_metrics = ClusterEvaluator.internal_metrics(processed_vectors, embedding_labels)
    
    if args.with_ground_truth:
        external_metrics = ClusterEvaluator.external_metrics(embedding_labels, true_labels)
        initial_metrics.update(external_metrics)
    
    # Initialize cross-encoder reranker
    logger.info("Initializing mock cross-encoder reranker...")
    reranker = MockReranker(processed_vectors)
    
    # Initialize cluster refiner
    logger.info("Initializing cluster refiner...")
    refiner = ClusterRefiner(
        reranker=reranker,
        embedding_weight=args.embedding_weight,
        cross_encoder_weight=args.cross_encoder_weight,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold
    )
    
    # Refine clusters
    logger.info(f"Refining clusters using {args.refinement_method} method...")
    refined_labels, refined_clusters, refinement_metrics = refiner.refine_clusters(
        clusters=embedding_clusters,
        labels=embedding_labels,
        vectors=processed_vectors,
        texts=texts,
        refine_method=args.refinement_method
    )
    
    # Calculate metrics for refined clustering
    logger.info("Evaluating refined clustering...")
    refined_metrics = ClusterEvaluator.internal_metrics(processed_vectors, refined_labels)
    
    if args.with_ground_truth:
        external_metrics = ClusterEvaluator.external_metrics(refined_labels, true_labels)
        refined_metrics.update(external_metrics)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    # First plot: Initial clustering
    fig1 = ClusterVisualizer.visualize_clusters_2d(
        vectors=processed_vectors,
        labels=embedding_labels,
        method='tsne',
        title="Initial Embedding-Based Clustering",
        figsize=(10, 6)
    )
    
    # Second plot: Refined clustering
    fig2 = ClusterVisualizer.visualize_clusters_2d(
        vectors=processed_vectors,
        labels=refined_labels,
        method='tsne',
        title="Cross-Encoder Refined Clustering",
        figsize=(10, 6)
    )
    
    # Create comparison visualization
    fig3 = ClusterVisualizer.compare_clustering_results(
        vectors=processed_vectors,
        labels_list=[embedding_labels, refined_labels],
        names=["Embedding-Based", "Cross-Encoder Refined"],
        method='tsne',
        title="Clustering Comparison",
        figsize=(14, 6)
    )
    
    # Show comparison of metrics
    logger.info("\n--- Clustering Evaluation ---")
    logger.info("\nInitial Embedding-Based Clustering:")
    for metric, value in initial_metrics.items():
        if value is not None:
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nCross-Encoder Refined Clustering:")
    for metric, value in refined_metrics.items():
        if value is not None:
            logger.info(f"{metric}: {value:.4f}")
    
    # Calculate differences
    logger.info("\nMetric Improvements:")
    for metric in initial_metrics:
        if initial_metrics[metric] is not None and refined_metrics[metric] is not None:
            diff = refined_metrics[metric] - initial_metrics[metric]
            if metric in ['davies_bouldin_score']:
                # For DB index, lower is better
                diff = -diff
            logger.info(f"{metric}: {diff:.4f} ({'improved' if diff > 0 else 'declined'})")
    
    # Count points that changed clusters
    changes = sum(1 for i, j in zip(embedding_labels, refined_labels) if i != j)
    change_percent = changes / len(embedding_labels) * 100
    logger.info(f"\nPoints that changed clusters: {changes} ({change_percent:.2f}%)")
    
    # Show plots if requested
    if args.show_plots:
        plt.figure(fig1.number)
        plt.pause(0.1)  # To allow window to appear
        plt.figure(fig2.number)
        plt.pause(0.1)
        plt.figure(fig3.number)
        plt.show()
    
    # Save plots if requested
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        fig1.savefig(os.path.join(args.output_dir, "initial_clustering.png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(args.output_dir, "refined_clustering.png"), dpi=300, bbox_inches='tight')
        fig3.savefig(os.path.join(args.output_dir, "clustering_comparison.png"), dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {args.output_dir}")
    
    # Optional: Run the full pipeline
    if args.run_pipeline:
        logger.info("\n--- Running Full Pipeline ---")
        config = {
            "embedding_clusterer": {
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples
            },
            "cross_encoder": {
                "use_refinement": True,
                "refinement_method": args.refinement_method,
                "embedding_weight": args.embedding_weight,
                "cross_encoder_weight": args.cross_encoder_weight,
                "confidence_threshold": args.confidence_threshold,
                "batch_size": args.batch_size
            },
            "output": {
                "save_results": args.save_plots,
                "output_dir": args.output_dir
            }
        }
        
        pipeline = ClusteringPipeline(config)
        pipeline_results = pipeline.run(vectors, texts, reranker=reranker)
        
        logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test cross-encoder refinement for clustering")
    
    # Data parameters
    parser.add_argument("--n_samples", type=int, default=300, help="Number of samples to use")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of true clusters")
    
    # Clustering parameters
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Min cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=2, help="Min samples for HDBSCAN")
    
    # Refinement parameters
    parser.add_argument("--refinement_method", type=str, default="borderline", 
                        choices=["borderline", "coherence", "full"],
                        help="Refinement method to use")
    parser.add_argument("--embedding_weight", type=float, default=0.5, 
                        help="Weight for embedding similarity")
    parser.add_argument("--cross_encoder_weight", type=float, default=0.5,
                        help="Weight for cross-encoder similarity")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="Confidence threshold for refinement")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for cross-encoder processing")
    
    # Evaluation parameters
    parser.add_argument("--with_ground_truth", action="store_true",
                        help="Whether to evaluate using ground truth labels")
    
    # Output parameters
    parser.add_argument("--show_plots", action="store_true",
                        help="Whether to show plots")
    parser.add_argument("--save_plots", action="store_true",
                        help="Whether to save plots")
    parser.add_argument("--output_dir", type=str, default="cross_encoder_test_results",
                        help="Directory to save results")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Whether to run the full pipeline")
    
    args = parser.parse_args()
    
    run_experiment(args)
