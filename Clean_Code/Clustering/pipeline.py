"""
Clustering pipeline that integrates all components.
Provides end-to-end workflows for embedding-based and cross-encoder enhanced clustering.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import clustering components
from Clustering.base_clusterer import BaseClusterer, check_is_clusterer
from Clustering.Embedding.hdbscan_clusterer import HdbscanClusterer
from Clustering.Processing.embedding_preprocessing import EmbeddingPreprocessor
from Clustering.Analytics.visualization import ClusterVisualizer
from Clustering.Analytics.evaluation import ClusterEvaluator
from Clustering.CrossEncoder.refinement import ClusterRefiner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusteringPipeline:
    """
    End-to-end pipeline for clustering text data using embeddings and cross-encoders.
    
    This pipeline integrates:
    1. Data preprocessing
    2. Embedding-based clustering
    3. Cross-encoder refinement (optional)
    4. Evaluation and visualization
    5. Results storage and analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the clustering pipeline with configuration.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - embedding_clusterer: Configuration for the embedding clusterer
                - preprocessing: Configuration for preprocessing
                - cross_encoder: Configuration for cross-encoder refinement
                - evaluation: Configuration for evaluation
                - visualization: Configuration for visualization
                - output: Configuration for output storage
        """
        # Default configuration
        self.default_config = {
            "embedding_clusterer": {
                "algorithm": "hdbscan",
                "min_cluster_size": 3,
                "min_samples": 2,
                "metric": "cosine",
                "cluster_selection_method": "eom",
                "prediction_data": True
            },
            "preprocessing": {
                "normalize": True,
                "normalize_method": "l2",
                "remove_outliers": False,
                "outlier_method": "zscore",
                "outlier_threshold": 3.0,
                "reduce_dimensions": False,
                "dimension_reduction_method": "pca",
                "n_components": 50
            },
            "cross_encoder": {
                "use_refinement": False,
                "refinement_method": "borderline",
                "embedding_weight": 0.7,
                "cross_encoder_weight": 0.3,
                "confidence_threshold": 0.6,
                "batch_size": 32
            },
            "evaluation": {
                "compute_metrics": True,
                "analyze_clusters": True,
                "analyze_stability": False,
                "n_stability_runs": 5,
                "subsample_ratio": 0.8
            },
            "visualization": {
                "create_visualizations": True,
                "method": "tsne",
                "dims": 2,
                "figsize": (12, 8),
                "save_plots": True
            },
            "output": {
                "save_results": True,
                "output_dir": "clustering_results",
                "save_model": False
            }
        }
        
        # Merge provided config with defaults
        self.config = self.default_config.copy()
        if config:
            self._merge_config(self.config, config)
        
        # Initialize components
        self.clusterer = None
        self.cross_encoder_refiner = None
        self.results = None
    
    def _merge_config(self, default: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            default: Default configuration (modified in-place)
            override: Override configuration
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def _initialize_clusterer(self) -> BaseClusterer:
        """
        Initialize the embedding clusterer based on configuration.
        
        Returns:
            Initialized clusterer
        """
        algorithm = self.config["embedding_clusterer"]["algorithm"].lower()
        
        if algorithm == "hdbscan":
            # Create HDBSCAN clusterer with config parameters
            return HdbscanClusterer(
                min_cluster_size=self.config["embedding_clusterer"]["min_cluster_size"],
                min_samples=self.config["embedding_clusterer"]["min_samples"],
                metric=self.config["embedding_clusterer"]["metric"],
                cluster_selection_method=self.config["embedding_clusterer"]["cluster_selection_method"],
                prediction_data=self.config["embedding_clusterer"]["prediction_data"]
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    
    def _initialize_cross_encoder_refiner(self, reranker: Any) -> Optional[ClusterRefiner]:
        """
        Initialize cross-encoder refiner if enabled.
        
        Args:
            reranker: Cross-encoder reranker instance
            
        Returns:
            Initialized refiner or None if disabled
        """
        if not self.config["cross_encoder"]["use_refinement"]:
            return None
        
        return ClusterRefiner(
            reranker=reranker,
            embedding_weight=self.config["cross_encoder"]["embedding_weight"],
            cross_encoder_weight=self.config["cross_encoder"]["cross_encoder_weight"],
            batch_size=self.config["cross_encoder"]["batch_size"],
            confidence_threshold=self.config["cross_encoder"]["confidence_threshold"]
        )
    
    def _prepare_output_directory(self) -> str:
        """
        Create output directory for results.
        
        Returns:
            Path to output directory
        """
        output_dir = self.config["output"]["output_dir"]
        
        # Add timestamp to output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"clustering_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    def run(self,
           vectors: np.ndarray,
           texts: List[str],
           reranker: Optional[Any] = None,
           metadata: Optional[List[Dict[str, Any]]] = None,
           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            texts: List of text strings corresponding to vectors
            reranker: Optional cross-encoder reranker for refinement
            metadata: Optional metadata for each text item
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary with clustering results
            
        Raises:
            ValueError: If inputs are invalid
        """
        start_time = time.time()
        
        # Validate inputs
        if len(vectors) != len(texts):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but texts has {len(texts)} items")
        
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(f"Length mismatch: texts has {len(texts)} items, but metadata has {len(metadata)} items")
        
        # Set up output directory
        if output_dir:
            self.config["output"]["output_dir"] = output_dir
        
        if self.config["output"]["save_results"]:
            output_dir = self._prepare_output_directory()
            logger.info(f"Results will be saved to: {output_dir}")
        
        # Initialize clusterer
        self.clusterer = self._initialize_clusterer()
        
        # Initialize cross-encoder refiner if enabled and reranker provided
        self.cross_encoder_refiner = None
        if self.config["cross_encoder"]["use_refinement"]:
            if reranker is None:
                logger.warning("Cross-encoder refinement enabled but no reranker provided. Refinement will be skipped.")
            else:
                self.cross_encoder_refiner = self._initialize_cross_encoder_refiner(reranker)
        
        # Step 1: Preprocess vectors
        logger.info("Preprocessing vectors...")
        processed_vectors, processed_texts, preprocessing_metadata = self._preprocess_vectors(vectors, texts)
        
        # Step 2: Perform embedding-based clustering
        logger.info("Performing embedding-based clustering...")
        embedding_results = self._perform_embedding_clustering(processed_vectors, processed_texts)
        
        # Step 3: Refine with cross-encoder (if enabled)
        if self.cross_encoder_refiner and reranker:
            logger.info("Refining clusters with cross-encoder...")
            refined_results = self._refine_with_cross_encoder(
                embedding_results, processed_vectors, processed_texts
            )
        else:
            refined_results = None
        
        # Use refined results if available, otherwise use embedding results
        clustering_results = refined_results if refined_results else embedding_results
        
        # Step 4: Evaluate results
        if self.config["evaluation"]["compute_metrics"]:
            logger.info("Evaluating clustering results...")
            evaluation_results = self._evaluate_clustering(
                processed_vectors, clustering_results["labels"], clustering_results["clusters"]
            )
        else:
            evaluation_results = {}
        
        # Step 5: Visualize results
        if self.config["visualization"]["create_visualizations"]:
            logger.info("Creating visualizations...")
            visualization_results = self._visualize_clustering(
                processed_vectors, clustering_results["labels"], 
                output_dir if self.config["output"]["save_results"] else None
            )
        else:
            visualization_results = {}
        
        # Step 6: Prepare final results
        final_results = {
            "labels": clustering_results["labels"],
            "clusters": clustering_results["clusters"],
            "preprocessing": preprocessing_metadata,
            "embedding_clustering": embedding_results["metrics"],
            "cross_encoder_refinement": refined_results["refinement_metrics"] if refined_results else None,
            "evaluation": evaluation_results,
            "visualization": visualization_results,
            "config": self.config,
            "runtime_seconds": time.time() - start_time
        }
        
        # Add original texts and metadata to results
        if metadata:
            # Map cluster labels to metadata
            for i, (label, meta) in enumerate(zip(final_results["labels"], metadata)):
                if "cluster_id" not in meta:
                    meta["cluster_id"] = int(label)
        
        # Step 7: Save results
        if self.config["output"]["save_results"]:
            self._save_results(final_results, output_dir, texts, metadata)
        
        # Store results
        self.results = final_results
        
        logger.info(f"Clustering completed in {final_results['runtime_seconds']:.2f} seconds")
        logger.info(f"Found {len(final_results['clusters'])} clusters")
        
        return final_results
    
    def _preprocess_vectors(self, 
                          vectors: np.ndarray, 
                          texts: List[str]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Preprocess vectors according to configuration.
        
        Args:
            vectors: Array of embedding vectors
            texts: List of text strings
            
        Returns:
            Tuple with processed vectors, texts, and preprocessing metadata
        """
        config = self.config["preprocessing"]
        
        # Prepare preprocessing parameters
        params = {
            "normalize_method": config["normalize_method"] if config["normalize"] else None,
            "remove_outliers_method": config["outlier_method"] if config["remove_outliers"] else None,
            "outlier_threshold": config["outlier_threshold"],
            "reduce_dimensions_method": config["dimension_reduction_method"] if config["reduce_dimensions"] else None,
            "n_components": config["n_components"]
        }
        
        # Process vectors
        processed_vectors, processed_texts, metadata = EmbeddingPreprocessor.prepare_vectors_for_clustering(
            vectors=vectors,
            data=texts,
            **params
        )
        
        return processed_vectors, processed_texts, metadata
    
    def _perform_embedding_clustering(self, 
                                    vectors: np.ndarray, 
                                    texts: List[str]) -> Dict[str, Any]:
        """
        Perform embedding-based clustering.
        
        Args:
            vectors: Preprocessed vectors
            texts: Preprocessed texts
            
        Returns:
            Dictionary with clustering results
        """
        # Fit clusterer
        results = self.clusterer.fit_predict(vectors, texts)
        
        return results
    
    def _refine_with_cross_encoder(self,
                                 embedding_results: Dict[str, Any],
                                 vectors: np.ndarray,
                                 texts: List[str]) -> Dict[str, Any]:
        """
        Refine clustering results using cross-encoder.
        
        Args:
            embedding_results: Results from embedding clustering
            vectors: Preprocessed vectors
            texts: Preprocessed texts
            
        Returns:
            Dictionary with refined clustering results
        """
        # Extract data from embedding results
        labels = embedding_results["labels"]
        clusters = embedding_results["clusters"]
        
        # Refine clusters
        refined_labels, refined_clusters, refinement_metrics = self.cross_encoder_refiner.refine_clusters(
            clusters=clusters,
            labels=labels,
            vectors=vectors,
            texts=texts,
            refine_method=self.config["cross_encoder"]["refinement_method"]
        )
        
        # Create refined results
        refined_results = {
            "labels": refined_labels,
            "clusters": refined_clusters,
            "refinement_metrics": refinement_metrics
        }
        
        return refined_results
    
    def _evaluate_clustering(self,
                           vectors: np.ndarray,
                           labels: np.ndarray,
                           clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate clustering results.
        
        Args:
            vectors: Preprocessed vectors
            labels: Cluster labels
            clusters: Cluster information
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_results = {}
        
        # Calculate metrics
        metrics = ClusterEvaluator.internal_metrics(vectors, labels)
        evaluation_results["metrics"] = metrics
        
        # Analyze clusters
        if self.config["evaluation"]["analyze_clusters"]:
            cluster_analysis = ClusterEvaluator.analyze_clusters(
                vectors=vectors,
                labels=labels,
                n_features_to_analyze=10
            )
            evaluation_results["cluster_analysis"] = cluster_analysis
        
        # Analyze stability
        if self.config["evaluation"]["analyze_stability"]:
            # Create a new instance of the same clusterer type
            temp_clusterer = self._initialize_clusterer()
            
            stability_results = ClusterEvaluator.analyze_cluster_stability(
                vectors=vectors,
                clusterer=temp_clusterer,
                n_runs=self.config["evaluation"]["n_stability_runs"],
                subsample_ratio=self.config["evaluation"]["subsample_ratio"]
            )
            evaluation_results["stability_analysis"] = stability_results
        
        return evaluation_results
    
    def _visualize_clustering(self,
                            vectors: np.ndarray,
                            labels: np.ndarray,
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualizations for clustering results.
        
        Args:
            vectors: Preprocessed vectors
            labels: Cluster labels
            output_dir: Output directory for saving visualizations
            
        Returns:
            Dictionary with visualization results
        """
        visualization_results = {}
        
        # Set up visualization parameters
        method = self.config["visualization"]["method"]
        dims = self.config["visualization"]["dims"]
        figsize = tuple(self.config["visualization"]["figsize"])
        
        # Create visualization based on dimensions
        if dims == 2:
            fig = ClusterVisualizer.visualize_clusters_2d(
                vectors=vectors,
                labels=labels,
                method=method,
                figsize=figsize,
                title="Cluster Visualization"
            )
            visualization_results["2d_plot"] = fig
            
            # Save if requested
            if output_dir and self.config["visualization"]["save_plots"]:
                output_path = os.path.join(output_dir, "cluster_visualization_2d.png")
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                visualization_results["2d_plot_path"] = output_path
        
        elif dims == 3:
            fig = ClusterVisualizer.visualize_clusters_3d(
                vectors=vectors,
                labels=labels,
                method=method,
                title="Cluster Visualization (3D)"
            )
            visualization_results["3d_plot"] = fig
            
            # Save if requested
            if output_dir and self.config["visualization"]["save_plots"]:
                output_path = os.path.join(output_dir, "cluster_visualization_3d.html")
                fig.write_html(output_path)
                visualization_results["3d_plot_path"] = output_path
        
        return visualization_results
    
    def _save_results(self,
                    results: Dict[str, Any],
                    output_dir: str,
                    texts: List[str],
                    metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Save clustering results to disk.
        
        Args:
            results: Clustering results
            output_dir: Output directory
            texts: Original texts
            metadata: Optional metadata
        """
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        metrics = {
            "embedding_clustering": results["embedding_clustering"],
            "cross_encoder_refinement": results["cross_encoder_refinement"],
            "evaluation": results["evaluation"].get("metrics", {}),
            "runtime_seconds": results["runtime_seconds"]
        }
        with open(metrics_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(metrics, f, indent=2, default=convert_numpy)
        
        # Save cluster assignments
        cluster_data = []
        for i, (label, text) in enumerate(zip(results["labels"], texts)):
            item = {
                "id": i,
                "cluster_id": int(label),
                "text": text
            }
            
            # Add metadata if available
            if metadata and i < len(metadata):
                item.update(metadata[i])
            
            cluster_data.append(item)
        
        # Save as JSON
        clusters_json_path = os.path.join(output_dir, "cluster_assignments.json")
        with open(clusters_json_path, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        # Save as CSV
        clusters_csv_path = os.path.join(output_dir, "cluster_assignments.csv")
        pd.DataFrame(cluster_data).to_csv(clusters_csv_path, index=False)
        
        # Save cluster information
        clusters_info_path = os.path.join(output_dir, "clusters_info.json")
        with open(clusters_info_path, 'w') as f:
            # Convert numpy values and arrays to Python types
            processed_clusters = []
            for cluster in results["clusters"]:
                processed_cluster = {}
                for key, value in cluster.items():
                    if isinstance(value, np.ndarray):
                        processed_cluster[key] = value.tolist()
                    elif isinstance(value, np.integer):
                        processed_cluster[key] = int(value)
                    elif isinstance(value, np.floating):
                        processed_cluster[key] = float(value)
                    else:
                        processed_cluster[key] = value
                processed_clusters.append(processed_cluster)
            
            json.dump(processed_clusters, f, indent=2)
        
        # Save clusterer if requested
        if self.config["output"]["save_model"] and hasattr(self.clusterer, "__getstate__"):
            import pickle
            model_path = os.path.join(output_dir, "clusterer.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.clusterer, f)
        
        logger.info(f"Results saved to {output_dir}")


# Example usage with product embeddings
if __name__ == "__main__":
    import argparse
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run clustering pipeline on product data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file with product data")
    parser.add_argument("--text_column", type=str, default="description", help="Column containing text to cluster")
    parser.add_argument("--output_dir", type=str, default="clustering_results", help="Output directory")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=2, help="Min samples parameter for HDBSCAN")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="Sentence transformer model name")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with subset of data")
    parser.add_argument("--test_size", type=int, default=500, help="Number of samples to use in test mode")
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Sample data in test mode
    if args.test_mode:
        logger.info(f"Running in test mode with {args.test_size} samples")
        df = df.sample(min(args.test_size, len(df)), random_state=42)
    
    # Get texts to cluster
    texts = df[args.text_column].tolist()
    logger.info(f"Loaded {len(texts)} texts to cluster")
    
    # Create embeddings
    logger.info(f"Creating embeddings using {args.model_name}")
    model = SentenceTransformer(args.model_name)
    vectors = model.encode(texts, show_progress_bar=True)
    logger.info(f"Created embeddings with shape {vectors.shape}")
    
    # Set up metadata
    metadata = df.to_dict(orient='records')
    
    # Configure pipeline
    config = {
        "embedding_clusterer": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples
        },
        "preprocessing": {
            "normalize": True,
            "normalize_method": "l2",
            "remove_outliers": False
        },
        "visualization": {
            "method": "tsne",
            "dims": 2
        },
        "output": {
            "output_dir": args.output_dir
        }
    }
    
    # Create and run pipeline
    pipeline = ClusteringPipeline(config)
    results = pipeline.run(vectors, texts, metadata=metadata)
    
    # Print summary
    num_clusters = len([c for c in results["clusters"] if c["id"] != -1])
    noise_points = sum(1 for label in results["labels"] if label == -1)
    noise_pct = noise_points / len(results["labels"]) * 100
    
    print("\nClustering Results Summary:")
    print(f"Number of clusters: {num_clusters}")
    print(f"Noise points: {noise_points} ({noise_pct:.2f}%)")
    
    if "metrics" in results["evaluation"]:
        metrics = results["evaluation"]["metrics"]
        if metrics["silhouette_score"] is not None:
            print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
        
    print(f"\nResults saved to: {args.output_dir}")
    print("Cluster assignments saved as CSV and JSON")
    
    # Print cluster sizes
    cluster_sizes = {}
    for c in results["clusters"]:
        cluster_sizes[c["id"]] = c["size"]
    
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    print("\nCluster sizes:")
    for cluster_id, size in sorted_clusters:
        if cluster_id == -1:
            print(f"Noise points: {size}")
        else:
            print(f"Cluster {cluster_id}: {size} items")
