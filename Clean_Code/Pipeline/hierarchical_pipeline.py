"""
Hierarchical clustering pipeline for VectorDB.
Implements multi-level clustering with flexible granularity.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import components from various modules

# Custom JSON Encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
from Clustering.Embedding.hdbscan_clusterer import HdbscanClusterer
from Clustering.Embedding.kmeans_clusterer import KMeansClusterer
from Clustering.Processing.embedding_preprocessing import EmbeddingPreprocessor
from Clustering.Analytics.visualization import ClusterVisualizer
from Clustering.Analytics.evaluation import ClusterEvaluator
from Clustering.CrossEncoder.refinement import ClusterRefiner
from Vector_Embedding.sentence_transformer_encoder import SentenceTransformerEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalClusteringPipeline:
    """
    Hierarchical clustering pipeline that builds clusters in a multi-level structure.
    
    Key features:
    1. Supports multiple levels of clustering granularity
    2. Each level builds upon the clusters from previous levels
    3. Cross-encoder refinement to improve clustering quality
    4. Flexible configuration for different clustering approaches
    5. Comprehensive evaluation and visualization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the hierarchical clustering pipeline.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - levels: Number of hierarchical levels
                - embedding: Configuration for embedding-based clustering
                - preprocessing: Configuration for preprocessing
                - cross_encoder: Configuration for cross-encoder refinement
                - evaluation: Configuration for evaluation
                - visualization: Configuration for visualization
                - output: Configuration for output storage
                - strict_validation: Flag to enable strict validation (default: True)
        """
        # Default configuration
        self.default_config = {
            "levels": 3,  # Number of hierarchical levels
            "level_configs": {  # Per-level configurations
                # Level 1 config
                1: {
                    "use_cross_encoder": False,  # Use embedder only for initial clustering
                    "min_cluster_size": 3,
                    "min_samples": 2,
                    "refine_after_clustering": True,  # Apply refinement after embedding-based clustering
                    "refinement_method": "borderline"
                },
                # Level 2 config
                2: {
                    "use_cross_encoder": False,  # Default to embedder for L2
                    "min_cluster_size": 3,
                    "min_samples": 2,
                    "refine_after_clustering": True,
                    "refinement_method": "borderline"
                },
                # Level 3 config
                3: {
                    "use_cross_encoder": False,  # Default to embedder for L3
                    "min_cluster_size": 3,
                    "min_samples": 2,
                    "refine_after_clustering": True,
                    "refinement_method": "borderline"
                }
            },
            "embedding": {
                "algorithm": "hdbscan",
                "min_cluster_size": 3,  # Default, overridden by level configs
                "min_samples": 2,       # Default, overridden by level configs
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
                "reduce_dimensions": False
            },
            "cross_encoder": {
                "use_refinement": True,        # Global switch for refinement
                "refinement_method": "borderline",  # Default, overridden by level configs
                "embedding_weight": 0.7,
                "cross_encoder_weight": 0.3,
                "confidence_threshold": 0.6,
                "batch_size": 32,
                "strict_validation": False     # Allow skipping invalid centroids by default
            },
            "evaluation": {
                "compute_metrics": True,
                "analyze_clusters": True
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
                "output_dir": "hierarchical_clustering_results",
                "save_model": False
            }
        }
        
        # Merge provided config with defaults
        self.config = self.default_config.copy()
        if config:
            self._merge_config(self.config, config)
        
        # Initialize components
        self.preprocessor = EmbeddingPreprocessor()
        self.clusterers = {}
        self.cross_encoder_refiner = None
        
        # Will store results for each level
        self.results = {}
    
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
    
    def _initialize_clusterer(self, level: int) -> Union[HdbscanClusterer, KMeansClusterer]:
        """
        Initialize a clusterer for the specified level using level-specific configuration.
        
        Args:
            level: Hierarchical level
            
        Returns:
            Initialized clusterer (HDBSCAN or KMeans)
        """
        # Get level-specific configuration if available
        level_config = self.config["level_configs"].get(level, {})
        
        # Get clustering method (HDBSCAN by default)
        clustering_method = level_config.get("clustering_method", "hdbscan").lower()
        
        if clustering_method == "kmeans":
            # Get KMeans-specific parameters
            n_clusters = level_config.get("n_clusters", 8)
            init = level_config.get("init", "k-means++")
            n_init = level_config.get("n_init", 10)
            max_iter = level_config.get("max_iter", 300)
            
            # Create KMeans clusterer
            logger.info(f"Using KMeans clustering for level {level} with {n_clusters} clusters")
            return KMeansClusterer(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter
            )
        else:
            # Get HDBSCAN parameters
            min_cluster_size = level_config.get("min_cluster_size", self.config["embedding"]["min_cluster_size"])
            min_samples = level_config.get("min_samples", self.config["embedding"]["min_samples"])
            metric = level_config.get("metric", self.config["embedding"]["metric"])
            cluster_selection_method = level_config.get(
                "cluster_selection_method", 
                self.config["embedding"]["cluster_selection_method"]
            )
            prediction_data = level_config.get("prediction_data", self.config["embedding"]["prediction_data"])
            
            # Additional optional parameters
            epsilon = level_config.get("cluster_selection_epsilon", 0.0)
            allow_single_cluster = level_config.get("allow_single_cluster", False)
            
            # Create HDBSCAN clusterer with level-specific parameters
            logger.info(f"Using HDBSCAN clustering for level {level} with min_cluster_size={min_cluster_size}")
            return HdbscanClusterer(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=epsilon,
                allow_single_cluster=allow_single_cluster,
                prediction_data=prediction_data
            )
    
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
        output_dir = os.path.join(output_dir, f"hierarchical_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    def _preprocess_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Preprocess vectors according to configuration.
        
        Args:
            vectors: Array of embedding vectors
            
        Returns:
            Preprocessed vectors
        """
        # Normalize vectors if configured
        if self.config["preprocessing"]["normalize"]:
            vectors = self.preprocessor.normalize(
                vectors, 
                norm=self.config["preprocessing"]["normalize_method"]
            )
        
        # Remove outliers if configured
        if self.config["preprocessing"]["remove_outliers"]:
            vectors, _, _ = self.preprocessor.detect_and_remove_outliers(
                vectors,
                method=self.config["preprocessing"]["outlier_method"],
                threshold=self.config["preprocessing"]["outlier_threshold"]
            )
        
        # Reduce dimensions if configured
        if self.config["preprocessing"]["reduce_dimensions"]:
            vectors = self.preprocessor.reduce_dimensions(
                vectors,
                method=self.config["preprocessing"].get("dimension_reduction_method", "pca"),
                n_components=self.config["preprocessing"].get("n_components", 50)
            )
        
        return vectors
    
    def run(self,
           vectors: np.ndarray,
           texts: List[str],
           reranker: Optional[Any] = None,
           metadata: Optional[List[Dict[str, Any]]] = None,
           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the hierarchical clustering pipeline.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            texts: List of text strings corresponding to vectors
            reranker: Optional cross-encoder reranker for refinement
            metadata: Optional metadata for each text item
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary with hierarchical clustering results
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
        
        # Initialize cross-encoder refiner if enabled and reranker provided
        self.cross_encoder_refiner = None
        if self.config["cross_encoder"]["use_refinement"]:
            if reranker is None:
                logger.warning("Cross-encoder refinement enabled but no reranker provided. Refinement will be skipped.")
            else:
                self.cross_encoder_refiner = self._initialize_cross_encoder_refiner(reranker)
        
        # Step 1: Preprocess vectors
        logger.info("Preprocessing vectors...")
        processed_vectors = self._preprocess_vectors(vectors)
        
        # Step 2: Perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering with {self.config['levels']} levels...")
        hierarchy_results = self._perform_hierarchical_clustering(
            processed_vectors=processed_vectors,
            texts=texts,
            num_levels=self.config["levels"]
        )
        
        # Step 3: Save results
        if self.config["output"]["save_results"]:
            self._save_results(hierarchy_results, output_dir, texts, processed_vectors, metadata)
        
        # Step 4: Create visualizations
        if self.config["visualization"]["create_visualizations"]:
            self._create_visualizations(hierarchy_results, processed_vectors, output_dir)
        
        # Calculate runtime
        runtime = time.time() - start_time
        logger.info(f"Hierarchical clustering completed in {runtime:.2f} seconds")
        
        # Add summary information
        hierarchy_results["summary"] = {
            "runtime_seconds": runtime,
            "config": self.config,
            "num_samples": len(vectors),
            "num_levels": self.config["levels"]
        }
        
        # Store results
        self.results = hierarchy_results
        
        return hierarchy_results
    
    def _perform_hierarchical_clustering(self,
                                       processed_vectors: np.ndarray,
                                       texts: List[str],
                                       num_levels: int) -> Dict[str, Any]:
        """
        Perform hierarchical clustering with the specified number of levels.
        
        Args:
            processed_vectors: Preprocessed embedding vectors
            texts: Text data corresponding to vectors
            num_levels: Number of hierarchical levels
            
        Returns:
            Dictionary with hierarchical clustering results
        """
        hierarchy_results = {
            "levels": {},
            "clustering_path": {}
        }
        
        # Log data size for performance reference
        logger.info(f"Starting hierarchical clustering on {len(processed_vectors)} vectors with dimension {processed_vectors.shape[1]}")
        
        # Initialize parent-child relationships for tracking hierarchy
        parent_child_map = {}
        child_parent_map = {}
        
        # Track which points are in which clusters at each level
        cluster_membership = {}
        
        # Level 1: Base clustering on all points
        logger.info("Processing Level 1 (base clustering)...")
        level_1_clusterer = self._initialize_clusterer(level=1)
        level_1_results = level_1_clusterer.fit_predict(processed_vectors, texts)
        
        # Store level 1 results
        hierarchy_results["levels"][1] = level_1_results
        self.clusterers[1] = level_1_clusterer
        
        # Track level 1 cluster membership
        for i, label in enumerate(level_1_results["labels"]):
            if label not in cluster_membership:
                cluster_membership[label] = []
            cluster_membership[label].append(i)
        
        # Optional: Refine level 1 with cross-encoder if enabled for this level
        level_1_config = self.config["level_configs"].get(1, {})
        refine_level_1 = level_1_config.get("refine_after_clustering", self.config["cross_encoder"]["use_refinement"])
        
        if self.cross_encoder_refiner is not None and refine_level_1:
            logger.info("Refining Level 1 clusters with cross-encoder...")
            refine_start_time = time.time()
            refined_labels, refined_clusters, _ = self.cross_encoder_refiner.refine_clusters(
                clusters=level_1_results["clusters"],
                labels=level_1_results["labels"],
                vectors=processed_vectors,
                texts=texts,
                refine_method=self.config["cross_encoder"]["refinement_method"]
            )
            refine_duration = time.time() - refine_start_time
            logger.info(f"Level 1 refinement completed in {refine_duration:.2f} seconds")
            level_1_results["labels"] = refined_labels
            level_1_results["clusters"] = refined_clusters
            hierarchy_results["levels"][1] = level_1_results
        else:
            logger.info("Skipping Level 1 refinement (disabled in configuration)")

        
        # Initialize clustering path tracking for each point
        for i in range(len(texts)):
            hierarchy_results["clustering_path"][i] = {
                1: int(level_1_results["labels"][i])
            }
        
        # Process higher levels if requested
        if num_levels > 1:
            for level in range(2, num_levels + 1):
                logger.info(f"Processing Level {level}...")
                level_start_time = time.time()
                self._process_level(
                    level=level,
                    processed_vectors=processed_vectors,
                    texts=texts,
                    hierarchy_results=hierarchy_results,
                    parent_child_map=parent_child_map,
                    child_parent_map=child_parent_map
                )
                level_duration = time.time() - level_start_time
                logger.info(f"Level {level} processing completed in {level_duration:.2f} seconds")
        
        # Add parent-child relationships to results
        hierarchy_results["parent_child_map"] = parent_child_map
        hierarchy_results["child_parent_map"] = child_parent_map
        
        return hierarchy_results
    
    def _process_level(self,
                       level: int,
                       processed_vectors: np.ndarray,
                       texts: List[str],
                       hierarchy_results: Dict[str, Any],
                       parent_child_map: Dict[str, List[str]],
                       child_parent_map: Dict[str, str]):
        """
        Process a single level of the hierarchy with support for per-level embedding or cross-encoder.
        
        Args:
            level: Current hierarchical level
            processed_vectors: Preprocessed embedding vectors
            texts: Text data corresponding to vectors
            hierarchy_results: Hierarchy results dictionary to update
            parent_child_map: Map of parent cluster IDs to child cluster IDs
            child_parent_map: Map of child cluster IDs to parent cluster IDs
        """
        # Get level-specific configuration
        level_config = self.config["level_configs"].get(level, {})
        use_cross_encoder = level_config.get("use_cross_encoder", False)
        refine_after_clustering = level_config.get("refine_after_clustering", True)
        refinement_method = level_config.get(
            "refinement_method", 
            self.config["cross_encoder"]["refinement_method"]
        )
        
        # Get results from previous level
        prev_level = level - 1
        prev_results = hierarchy_results["levels"][prev_level]
        prev_labels = prev_results["labels"]
        
        # Group points by cluster from previous level
        clusters_from_prev_level = {}
        for i, label in enumerate(prev_labels):
            if label == -1:  # Skip noise points
                continue
                
            if label not in clusters_from_prev_level:
                clusters_from_prev_level[label] = []
            clusters_from_prev_level[label].append(i)
        
        # Store level results
        level_results = {
            "clusters": [],
            "labels": np.full(len(texts), -1, dtype=int),  # Default to noise
            "parent_clusters": {},
            "statistics": {},
            "clustering_method": "cross_encoder" if use_cross_encoder else "embedding"
        }
        
        # Process each cluster from the previous level
        next_cluster_id = 0
        for parent_cluster_id, member_indices in clusters_from_prev_level.items():
            # Get minimum cluster size for this level
            min_cluster_size = level_config.get(
                "min_cluster_size", 
                self.config["embedding"]["min_cluster_size"]
            )
            
            # Skip clusters that are too small
            if len(member_indices) < min_cluster_size:
                logger.info(f"Skipping cluster {parent_cluster_id} at level {prev_level} (too small with {len(member_indices)} members)")
                continue

            # Get vectors and texts for this cluster
            cluster_vectors = processed_vectors[member_indices]
            cluster_texts = [texts[i] for i in member_indices]

            # Defensive: skip clusters with all-NaN, all-zero, or all-Inf vectors
            questions_path = "/Users/billnewman/Desktop/GitHub/VectorDB/Clean_Code/Questions.md"
            def log_skipped_cluster(reason):
                with open(questions_path, "a") as f:
                    f.write("\n---\n")
                    f.write(f"## Skipped Cluster Due to Data Anomaly\n")
                    f.write(f"**Parent Cluster ID:** {parent_cluster_id}\n")
                    f.write(f"**Level:** {prev_level}\n")
                    f.write(f"**Reason:** {reason}\n")
                    f.write(f"**Member Indices:** {member_indices}\n")
                    f.write(f"**Vectors Shape:** {cluster_vectors.shape}\n")
                    f.write(f"**Vectors (first 3 shown):**\n{cluster_vectors[:3]}\n")
                    f.write("---\n")
            if cluster_vectors.size == 0:
                logger.warning(f"Skipping cluster {parent_cluster_id} at level {prev_level} (no vectors)")
                log_skipped_cluster("No vectors in cluster")
                continue
            # Check for any invalid vectors (NaN/Inf/all-zero)
            invalid_vector_indices = []
            for idx, vec in enumerate(cluster_vectors):
                if np.isnan(vec).any() or np.isinf(vec).any():
                    invalid_vector_indices.append(idx)
                elif np.allclose(vec, 0):
                    invalid_vector_indices.append(idx)
            if invalid_vector_indices:
                logger.warning(f"Skipping cluster {parent_cluster_id} at level {prev_level} due to invalid member vectors at indices {invalid_vector_indices}")
                log_skipped_cluster(f"Invalid member vectors at indices: {invalid_vector_indices}")
                continue
            # Skip clusters with identical vectors
            if np.all(np.isclose(cluster_vectors[0], cluster_vectors)):
                logger.info(f"Skipping cluster {parent_cluster_id} at level {prev_level} (all vectors identical)")
                continue

            # Perform clustering based on method
            try:
                logger.info(f"Clustering points in level {prev_level} cluster {parent_cluster_id} ({len(member_indices)} points)...")
                
                if use_cross_encoder and self.cross_encoder_refiner is not None:
                    # Use cross-encoder for direct clustering
                    logger.info(f"Using cross-encoder for level {level} clustering...")
                    reranker = self.cross_encoder_refiner.reranker
                    cluster_results = self._perform_cross_encoder_clustering(
                        cluster_vectors=cluster_vectors,
                        cluster_texts=cluster_texts,
                        reranker=reranker
                    )
                    # Track that we're using the clusterer
                    self.clusterers[f"{level}_{parent_cluster_id}_cross_encoder"] = True
                else:
                    # Use embedding-based clustering
                    logger.info(f"Using embedding-based clustering for level {level}...")
                    clusterer = self._initialize_clusterer(level=level)
                    cluster_results = clusterer.fit_predict(cluster_vectors, cluster_texts)
                    # Store clusterer for later use
                    self.clusterers[f"{level}_{parent_cluster_id}"] = clusterer
                
                # Map subcluster IDs to global IDs
                for i, (local_idx, sub_label) in enumerate(zip(member_indices, cluster_results["labels"])):
                    if sub_label != -1:  # Not noise
                        # Generate a unique global ID for this subcluster
                        global_sub_cluster_id = next_cluster_id
                        next_cluster_id += 1
                        
                        # Store parent-child relationship
                        parent_key = f"{prev_level}_{parent_cluster_id}"
                        child_key = f"{level}_{global_sub_cluster_id}"
                        
                        if parent_key not in parent_child_map:
                            parent_child_map[parent_key] = []
                        
                        # Create the global cluster for non-noise points
                        global_members = []
                        for j, (idx, label) in enumerate(zip(member_indices, cluster_results["labels"])):
                            if label == sub_label:
                                global_members.append(idx)
                                # Map local index to global cluster ID
                                level_results["labels"][idx] = global_sub_cluster_id
                        
                        # Calculate centroid for this cluster
                        centroid = None
                        if len(global_members) > 0:
                            centroid = np.mean(processed_vectors[global_members], axis=0)
                        
                        global_cluster = {
                            "id": global_sub_cluster_id,
                            "parent_id": parent_cluster_id,
                            "size": len(global_members),
                            "members": global_members,
                            "centroid": centroid
                        }
                        
                        # Log every created cluster to Questions.md for audit/debugging
                        created_vectors = processed_vectors[global_members] if len(global_members) > 0 else np.array([])
                        with open("/Users/billnewman/Desktop/GitHub/VectorDB/Clean_Code/Questions.md", "a") as f:
                            f.write("\n---\n")
                            f.write(f"## Created Cluster\n")
                            f.write(f"**Cluster ID:** {global_cluster['id']}\n")
                            f.write(f"**Parent ID:** {global_cluster['parent_id']}\n")
                            f.write(f"**Size:** {global_cluster['size']}\n")
                            f.write(f"**Members:** {global_cluster['members']}\n")
                            f.write(f"**Vectors Shape:** {created_vectors.shape}\n")
                            f.write(f"**Vectors (first 3 shown):**\n{created_vectors[:3]}\n")
                            f.write("---\n")
                        
                        level_results["clusters"].append(global_cluster)
                    
            except Exception as e:
                logger.warning(f"Failed to cluster level {prev_level} cluster {parent_cluster_id}: {str(e)}")
                logger.exception(e)
        
        # Store level results
        hierarchy_results["levels"][level] = level_results
        
        # Optional: Refine with cross-encoder if enabled for this level and not already using cross-encoder
        if (not use_cross_encoder and 
            refine_after_clustering and 
            self.cross_encoder_refiner is not None and 
            level_results["clusters"]):
            
            logger.info(f"Refining Level {level} clusters with cross-encoder...")
            refine_start_time = time.time()
            try:
                # Validate that we have valid clusters to refine
                if not level_results["clusters"] or all(cluster.get('id', -1) == -1 for cluster in level_results["clusters"]):
                    logger.warning(f"No valid clusters to refine at level {level}. Skipping refinement.")
                else:
                    # Get strict_validation setting from config
                    strict_validation = self.config["cross_encoder"].get("strict_validation", False)
                    
                    refined_labels, refined_clusters, _ = self.cross_encoder_refiner.refine_clusters(
                        clusters=level_results["clusters"],
                        labels=level_results["labels"],
                        vectors=processed_vectors,
                        texts=texts,
                        refine_method=refinement_method,
                        strict_validation=strict_validation
                    )
                    refine_duration = time.time() - refine_start_time
                    logger.info(f"Level {level} refinement completed in {refine_duration:.2f} seconds")
                    level_results["labels"] = refined_labels
                    level_results["clusters"] = refined_clusters
                    level_results["refined"] = True
                    hierarchy_results["levels"][level] = level_results
                    
                    # Update clustering paths
                    for i, label in enumerate(refined_labels):
                        if label != -1:  # Not noise
                            hierarchy_results["clustering_path"][i][level] = int(label)
                    
                    logger.info(f"Successfully refined level {level} clusters with {refinement_method} method")
            except ValueError as ve:
                # More specific error for known validation issues
                logger.error(f"Validation error during level {level} refinement: {str(ve)}")
                # Let the error propagate - we don't want to hide validation errors
                raise
            except Exception as e:
                logger.error(f"Unexpected error during level {level} refinement: {str(e)}")
                logger.exception(e)
                # Raise the exception to stop the process
                raise RuntimeError(f"Failed to refine clusters at level {level}: {str(e)}")
        
        # Calculate statistics for this level
        num_clusters = len(level_results["clusters"])
        points_assigned = np.sum(level_results["labels"] != -1)
        noise_percentage = (len(level_results["labels"]) - points_assigned) / len(level_results["labels"]) * 100
        
        level_results["statistics"] = {
            "num_clusters": num_clusters,
            "points_assigned": int(points_assigned),
            "noise_points": int(len(level_results["labels"]) - points_assigned),
            "noise_percentage": float(noise_percentage),
            "clustering_method": "cross_encoder" if use_cross_encoder else "embedding",
            "refined": level_results.get("refined", False)
        }
        
        logger.info(f"Level {level} clustering complete: {num_clusters} clusters, {points_assigned}/{len(level_results['labels'])} points assigned")
    
    def get_cluster_assignment(self, point_idx: int, max_level: Optional[int] = None) -> Dict[int, int]:
        """
        Get cluster assignments for a specific point at all levels.
        
        Args:
            point_idx: Index of the point
            max_level: Maximum level to consider (None for all levels)
            
        Returns:
            Dictionary mapping level number to cluster ID
        """
        if not self.results or "clustering_path" not in self.results:
            raise ValueError("Clustering has not been performed yet")
            
        if point_idx not in self.results["clustering_path"]:
            raise ValueError(f"Point index {point_idx} not found in clustering results")
        
        # Get cluster path for this point
        cluster_path = self.results["clustering_path"][point_idx]
        
        # Filter by max level if specified
        if max_level is not None:
            return {level: cluster_id for level, cluster_id in cluster_path.items() if level <= max_level}
        
        return cluster_path
    
    def find_similar_items(self, query_idx: int, level: int = 1, k: int = 10) -> List[int]:
        """
        Find items similar to the query item based on cluster membership.
        
        Args:
            query_idx: Index of the query item
            level: Hierarchy level to use
            k: Number of similar items to return
            
        Returns:
            List of indices of similar items
        """
        if not self.results or "levels" not in self.results:
            raise ValueError("Clustering has not been performed yet")
            
        if level not in self.results["levels"]:
            raise ValueError(f"Level {level} not found in clustering results")
        
        # Get cluster ID for query item at the specified level
        if query_idx not in self.results["clustering_path"] or level not in self.results["clustering_path"][query_idx]:
            logger.warning(f"Query item {query_idx} not assigned to any cluster at level {level}")
            # Try the highest available level less than the requested level
            available_levels = [l for l in self.results["clustering_path"][query_idx].keys() if l < level]
            if not available_levels:
                return []
            level = max(available_levels)
            logger.info(f"Using level {level} instead")
        
        cluster_id = self.results["clustering_path"][query_idx][level]
        
        # Find cluster members
        members = []
        for cluster in self.results["levels"][level]["clusters"]:
            if cluster["id"] == cluster_id:
                members = cluster["members"]
                break
        
        # Sort members by distance to centroid (approximation of similarity)
        if not members:
            return []
        
        # Return up to k members (excluding the query item itself)
        members = [idx for idx in members if idx != query_idx]
        return members[:k]
    
    def _create_visualizations(self, 
                             hierarchy_results: Dict[str, Any],
                             vectors: np.ndarray,
                             output_dir: str) -> None:
        """
        Create visualizations for hierarchical clustering results.
        
        Args:
            hierarchy_results: Hierarchical clustering results
            vectors: Preprocessed embedding vectors
            output_dir: Output directory for visualizations
        """
        vis_start_time = time.time()
        logger.info("Creating visualizations...")
        visualizer = ClusterVisualizer()
        
        # Visualize each level
        for level, level_results in hierarchy_results["levels"].items():
            # Skip if no clusters at this level
            if not level_results["clusters"]:
                continue
                
            logger.info(f"Visualizing Level {level}...")
            
            # Create 2D plot
            if self.config["visualization"]["dims"] == 2:
                fig = visualizer.visualize_clusters_2d(
                    vectors=vectors,
                    labels=level_results["labels"],
                    method=self.config["visualization"]["method"],
                    figsize=tuple(self.config["visualization"]["figsize"])
                )
                
                # Save plot
                if self.config["visualization"]["save_plots"]:
                    plt.savefig(os.path.join(output_dir, f"level_{level}_clusters.png"), dpi=300, bbox_inches="tight")
                    plt.close()
        
            # Create 3D plot if requested
            elif self.config["visualization"]["dims"] == 3:
                fig = visualizer.visualize_clusters_3d(
                    vectors=vectors,
                    labels=level_results["labels"],
                    method=self.config["visualization"]["method"],
                    figsize=tuple(self.config["visualization"]["figsize"])
                )
                
                # Save figure
                if self.config["visualization"]["save_plots"]:
                    fig_path = os.path.join(output_dir, f"level_{level}_clusters_3d.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                    logger.info(f"Saved visualization to {fig_path}")
    
    def _save_results(self,
                     hierarchy_results: Dict[str, Any],
                     output_dir: str,
                     texts: List[str],
                     processed_vectors: np.ndarray,
                     metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Save hierarchical clustering results to files.
        
        Args:
            hierarchy_results: Hierarchical clustering results
            output_dir: Output directory
            texts: Text data
            metadata: Optional metadata for each text item
        """
        if not self.config["output"]["save_results"]:
            return
            
        logger.info("Saving results...")
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Create DataFrame with cluster assignments for each level
        df_data = {"text": texts}
        
        # Add cluster assignments for each level
        for level in hierarchy_results["levels"].keys():
            df_data[f"cluster_level_{level}"] = hierarchy_results["levels"][level]["labels"]
        
        # Add metadata if provided
        if metadata is not None:
            for i, item in enumerate(metadata):
                for key, value in item.items():
                    if key not in df_data:
                        df_data[key] = [None] * len(texts)
                    df_data[key][i] = value
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_dir, "hierarchical_clusters.csv")
        df.to_csv(csv_path, index=False)
        
        # Save cluster information for each level
        for level, level_results in hierarchy_results["levels"].items():
            # Convert cluster information to a serializable format
            serializable_clusters = []
            
            # Handle different cluster representations (K-means vs HDBSCAN)
            # For K-means: clusters is a dictionary with integer keys
            # For HDBSCAN: clusters is a list of dictionaries with 'id', 'members', etc.
            if isinstance(level_results["clusters"], dict):
                # K-means style clusters (dict with integer keys)
                for cluster_id, members in level_results["clusters"].items():
                    # For K-means, calculate centroid as mean of member vectors
                    centroid = np.mean([processed_vectors[m] for m in members], axis=0) if len(members) > 0 else None
                    
                    serializable_cluster = {
                        "id": int(cluster_id),
                        "size": len(members),
                        "members": [int(m) for m in members],
                        "centroid": centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
                    }
                    serializable_clusters.append(serializable_cluster)
            else:
                # HDBSCAN style clusters (list of dictionaries)
                for cluster in level_results["clusters"]:
                    serializable_cluster = {
                        "id": int(cluster["id"]),
                        "size": int(cluster["size"]) if "size" in cluster else len(cluster["members"]),
                        "members": [int(m) for m in cluster["members"]],
                        "centroid": cluster["centroid"].tolist() if isinstance(cluster["centroid"], np.ndarray) else cluster["centroid"],
                    }
                    
                    # Add parent ID if available
                    if "parent_id" in cluster:
                        serializable_cluster["parent_id"] = int(cluster["parent_id"])
                        
                    serializable_clusters.append(serializable_cluster)
            
            # Save level results
            level_data = {
                "clusters": serializable_clusters,
                "statistics": level_results.get("statistics", {}),
                "parent_clusters": level_results.get("parent_clusters", {})
            }
            
            level_path = os.path.join(output_dir, f"level_{level}_clusters.json")
            with open(level_path, "w") as f:
                json.dump(level_data, f, indent=2, cls=NumpyEncoder)
        
        # Save hierarchy information
        hierarchy_data = {
            "parent_child_map": hierarchy_results["parent_child_map"],
            "child_parent_map": hierarchy_results["child_parent_map"],
            "summary": hierarchy_results.get("summary", {})
        }
        
        hierarchy_path = os.path.join(output_dir, "hierarchy.json")
        with open(hierarchy_path, "w") as f:
            json.dump(hierarchy_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved results to {output_dir}")


# Simple usage example
if __name__ == "__main__":
    import argparse
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hierarchical Clustering Pipeline")
    parser.add_argument("--levels", type=int, default=3, help="Number of hierarchical levels")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples for HDBSCAN")
    parser.add_argument("--use_refinement", action="store_true", help="Use cross-encoder refinement")
    parser.add_argument("--n_samples", type=int, default=300, help="Number of samples to use from dataset")
    parser.add_argument("--save_plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--output_dir", type=str, default="hierarchical_clustering_results", help="Output directory")
    args = parser.parse_args()
    
    # Load sample data
    logger.info("Loading sample data...")
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space', 'talk.politics.guns']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Limit to requested number of samples
    n_samples = min(args.n_samples, len(newsgroups.data))
    texts = newsgroups.data[:n_samples]
    true_labels = newsgroups.target[:n_samples]
    
    # Create text IDs for cross-encoder lookup
    texts = [f"text_{i}: {text[:50].replace(chr(10), ' ')}..." for i, text in enumerate(texts)]
    
    # Create TF-IDF vectors
    logger.info("Creating embeddings...")
    vectorizer = TfidfVectorizer(max_features=100)
    vectors = vectorizer.fit_transform(newsgroups.data[:n_samples]).toarray()
    
    # Initialize hierarchical pipeline with custom config
    config = {
        "levels": args.levels,
        "embedding": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples
        },
        "cross_encoder": {
            "use_refinement": args.use_refinement
        },
        "visualization": {
            "save_plots": args.save_plots
        },
        "output": {
            "output_dir": args.output_dir
        }
    }
    
    # Mock reranker for demonstration
    class MockReranker:
        def __init__(self, vectors):
            self.vectors = vectors
        
        def compute_similarity(self, queries, passages):
            results = []
            for query, passage in zip(queries, passages):
                try:
                    idx1 = int(query.split('_')[-1].split(':')[0])
                    idx2 = int(passage.split('_')[-1].split(':')[0])
                    
                    vec1 = self.vectors[idx1]
                    vec2 = self.vectors[idx2]
                    
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    noise = np.random.normal(0, 0.1)
                    similarity = min(1.0, max(0.0, similarity + noise))
                    
                    results.append(float(similarity))
                except:
                    results.append(float(np.random.uniform(0.3, 0.7)))
            return results
    
    # Initialize reranker if needed
    reranker = None
    if args.use_refinement:
        reranker = MockReranker(vectors)
    
    # Initialize and run pipeline
    logger.info("Initializing hierarchical clustering pipeline...")
    pipeline = HierarchicalClusteringPipeline(config)
    
    # Add metadata with true labels
    metadata = [{"true_category": categories[label], "true_label": int(label)} for label in true_labels]
    
    # Run hierarchical clustering
    logger.info("Running hierarchical clustering...")
    results = pipeline.run(vectors, texts, reranker, metadata)
    
    # Print summary
    logger.info("Hierarchical clustering complete!")
    for level, level_results in results["levels"].items():
        stats = level_results.get("statistics", {})
        num_clusters = stats.get("num_clusters", 0)
        points_assigned = stats.get("points_assigned", 0)
        noise_percentage = stats.get("noise_percentage", 0)
        logger.info(f"Level {level}: {num_clusters} clusters, {points_assigned}/{n_samples} points assigned ({noise_percentage:.2f}% noise)")
