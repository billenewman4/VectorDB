"""
HDBSCAN clusterer implementation for the Clustering module.
Provides density-based clustering optimized for embedding vectors.
"""

import os
import sys
import numpy as np
import hdbscan
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.dirname(current_dir)
clustering_dir = os.path.dirname(embedding_dir)
sys.path.append(os.path.dirname(clustering_dir))

# Import base clusterer interface
from Clustering.base_clusterer import BaseClusterer

class HdbscanClusterer(BaseClusterer):
    """
    Density-based clustering using HDBSCAN algorithm.
    
    HDBSCAN is particularly well-suited for embedding vectors because it:
    - Discovers clusters of varying densities and shapes
    - Identifies outliers as noise points (-1 label)
    - Doesn't require specifying the number of clusters in advance
    - Works well with high-dimensional data
    
    Default parameters are optimized for product embeddings using the all-mpnet-base-v2 model:
    - min_cluster_size=3: Creates more focused product groups
    - min_samples=2: Balances noise detection with cluster formation
    """
    
    def __init__(self, 
                min_cluster_size: int = 3,
                min_samples: int = 2,
                metric: str = 'cosine',
                cluster_selection_method: str = 'eom',
                cluster_selection_epsilon: float = 0.0,
                alpha: float = 1.0,
                algorithm: str = 'best',
                leaf_size: int = 40,
                memory: Optional[str] = None,
                approx_min_span_tree: bool = True,
                gen_min_span_tree: bool = False,
                core_dist_n_jobs: int = -1,
                allow_single_cluster: bool = False,
                prediction_data: bool = True):
        """
        Initialize HDBSCAN clusterer with the specified parameters.
        
        Args:
            min_cluster_size: Minimum size of clusters (default: 3)
            min_samples: Minimum number of samples in a dense region (default: 2)
            metric: Distance metric (default: 'cosine')
            cluster_selection_method: Method to select flat clusters ('eom' or 'leaf')
            cluster_selection_epsilon: Distance threshold for cluster extraction
            alpha: Scaling factor for outlier scores
            algorithm: Algorithm to use ('best', 'generic', 'prims_kdtree', etc.)
            leaf_size: Leaf size for tree algorithms
            memory: Cache directory location
            approx_min_span_tree: Whether to use approximate minimum spanning tree
            gen_min_span_tree: Whether to generate minimum spanning tree for later analysis
            core_dist_n_jobs: Number of parallel jobs for core distance calculation
            allow_single_cluster: Whether to allow a single cluster (vs only noise)
            prediction_data: Whether to store data for predicting cluster assignments
            
        Note:
            Default parameters (min_cluster_size=3, min_samples=2) are optimized
            for product embeddings using the all-mpnet-base-v2 model based on
            empirical testing which showed these values create more focused
            product groups than larger cluster sizes.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.allow_single_cluster = allow_single_cluster
        self.prediction_data = prediction_data
        
        # Will be set during fitting
        self.clusterer = None
        self.labels_ = None
        self.probabilities_ = None
        self.vectors_ = None
        self.data_ = None
        self.is_fitted = False
        self.metrics_ = {}
    
    def fit(self, vectors: np.ndarray, data: Optional[List[Any]] = None) -> 'HdbscanClusterer':
        """
        Fit the HDBSCAN clusterer to the input vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            data: Optional additional data associated with each vector
            
        Returns:
            self: The fitted clusterer instance
            
        """
        # Store vectors and data for later use
        self.vectors_ = vectors
        self.data_ = data if data is not None else [None] * len(vectors)
        
        # Handle special case for cosine metric which isn't directly supported by HDBSCAN
        original_metric = self.metric
        vectors_for_clustering = vectors.copy()
        
        # If using cosine, normalize vectors and switch to euclidean metric
        # This is mathematically equivalent to cosine distance for unit vectors
        if original_metric == 'cosine':
            vectors_for_clustering = normalize(vectors, norm='l2', axis=1)
            self.metric = 'euclidean'  # Switch to euclidean on normalized vectors
        
        # Initialize HDBSCAN with modified parameters
        kwargs = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': self.metric,  # Will be 'euclidean' if original was 'cosine'
            'cluster_selection_method': self.cluster_selection_method,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'alpha': self.alpha,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'approx_min_span_tree': self.approx_min_span_tree,
            'gen_min_span_tree': self.gen_min_span_tree,
            'core_dist_n_jobs': self.core_dist_n_jobs,
            'allow_single_cluster': self.allow_single_cluster,
            'prediction_data': self.prediction_data
        }
        
        # Only add memory parameter if it's a valid string path
        if isinstance(self.memory, str) and self.memory:
            kwargs['memory'] = self.memory
        
        # Create and fit the HDBSCAN clusterer
        self.clusterer = hdbscan.HDBSCAN(**kwargs)
        self.clusterer.fit(vectors_for_clustering)
        
        # Store labels and probabilities
        self.labels_ = self.clusterer.labels_
        
        # Reset original metric for consistency
        self.metric = original_metric
        
        # Sometimes probabilities are not available (with old versions of hdbscan)
        try:
            self.probabilities_ = self.clusterer.probabilities_
        except AttributeError:
            self.probabilities_ = np.ones_like(self.labels_, dtype=float)
            
        # Calculate evaluation metrics
        self._calculate_metrics(vectors_for_clustering)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            
        Returns:
            Array of cluster labels for each input vector
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If called before fitting
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer must be fitted before calling predict")
        
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if vectors.shape[1] != self.vectors_.shape[1]:
            raise ValueError(
                f"Dimension mismatch: input vectors have {vectors.shape[1]} features, "
                f"but the clusterer was trained with {self.vectors_.shape[1]} features"
            )
        
        # Normalize vectors if using cosine metric
        if self.metric == 'cosine':
            vectors_for_prediction = normalize(vectors)
        else:
            vectors_for_prediction = vectors
        
        # Predict labels
        labels, _ = hdbscan.approximate_predict(self.clusterer, vectors_for_prediction)
        return labels
    
    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about each cluster.
        
        Returns:
            List of dictionaries, one per cluster, containing:
                - 'id': Cluster identifier
                - 'size': Number of points in the cluster
                - 'centroid': Centroid of the cluster
                - 'members': Indices of cluster members
                - 'core_samples': Indices of core samples
                - 'persistence': Cluster persistence
                - 'stability': Cluster stability
                - 'data': Associated data if provided during fit
                
        Raises:
            RuntimeError: If called before fitting
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer must be fitted before calling get_clusters")
        
        # Get unique labels (excluding noise points with label -1)
        unique_labels = np.unique(self.labels_)
        clusters = []
        
        for label in unique_labels:
            # Skip noise points (label -1)
            if label == -1:
                continue
            
            # Get indices of points in this cluster
            cluster_indices = np.where(self.labels_ == label)[0]
            cluster_vectors = self.vectors_[cluster_indices]
            
            # Calculate centroid
            centroid = np.mean(cluster_vectors, axis=0)
            
            # Validate centroid
            def is_valid_centroid(centroid_arr):
                if centroid_arr is None or centroid_arr.size == 0:
                    return False
                if np.isnan(centroid_arr).any() or np.isinf(centroid_arr).any():
                    return False
                if np.allclose(centroid_arr, 0):
                    return False
                return True
            if not is_valid_centroid(centroid):
                # Log to Questions.md
                questions_path = "/Users/billnewman/Desktop/GitHub/VectorDB/Clean_Code/Questions.md"
                with open(questions_path, "a") as f:
                    f.write("\n---\n")
                    f.write("## Invalid Cluster Centroid Detected\n")
                    f.write(f"**Cluster ID:** {label}\n")
                    f.write(f"**Size:** {len(cluster_indices)}\n")
                    f.write(f"**Centroid:** {centroid if centroid is not None else None}\n")
                    f.write(f"**Members:** {cluster_indices.tolist()}\n")
                    f.write(f"**Vectors:**\n{cluster_vectors}\n")
                    f.write("---\n")
                raise ValueError(f"Invalid centroid for cluster {label}: {centroid}. See Questions.md for details.")

            # Find core samples for this cluster
            if hasattr(self.clusterer, 'exemplars_') and self.clusterer.exemplars_ is not None:
                try:
                    # Convert exemplars to a flat array if it's a nested structure
                    exemplars = self.clusterer.exemplars_
                    if isinstance(exemplars, list):
                        # If exemplars is a list of arrays, flatten it
                        flattened_exemplars = []
                        for ex in exemplars:
                            if hasattr(ex, 'tolist'):
                                flattened_exemplars.extend(ex.tolist())
                            elif isinstance(ex, list):
                                flattened_exemplars.extend(ex)
                            else:
                                flattened_exemplars.append(ex)
                        
                        # Get exemplars for this cluster
                        exemplar_indices = np.where(
                            np.isin(cluster_indices, flattened_exemplars)
                        )[0]
                    else:
                        # Already a flat array
                        exemplar_indices = np.where(
                            np.isin(cluster_indices, exemplars)
                        )[0]
                    
                    core_samples = cluster_indices[exemplar_indices].tolist()
                except Exception as e:
                    # Fall back to using all points if there's an error with exemplars
                    self.logger.warning(f"Error processing exemplars: {e}. Using all points as core samples.")
                    core_samples = cluster_indices.tolist()
            else:
                # Otherwise just use all points
                core_samples = cluster_indices.tolist()
            
            # Get persistence and stability if available
            persistence = None
            stability = None
            if hasattr(self.clusterer, 'cluster_persistence_'):
                persistence = self.clusterer.cluster_persistence_[label]
            
            # Create cluster info dictionary
            cluster_info = {
                'id': int(label),
                'size': len(cluster_indices),
                'centroid': centroid,  # Keep as numpy array for consistent type handling
                'members': cluster_indices.tolist(),
                'core_samples': core_samples,
                'persistence': persistence,
                'stability': stability
            }
            
            # Add associated data if available
            if self.data_ is not None:
                cluster_data = [self.data_[i] for i in cluster_indices]
                cluster_info['data'] = cluster_data
            
            clusters.append(cluster_info)
        
        # Add noise cluster if it exists
        if -1 in unique_labels:
            noise_indices = np.where(self.labels_ == -1)[0]
            noise_info = {
                'id': -1,  # Noise always has label -1
                'size': len(noise_indices),
                'centroid': None,  # Noise has no meaningful centroid
                'members': noise_indices.tolist(),
                'core_samples': [],
                'persistence': None,
                'stability': None
            }
            
            # Add associated data if available
            if self.data_ is not None:
                noise_data = [self.data_[i] for i in noise_indices]
                noise_info['data'] = noise_data
            
            clusters.append(noise_info)
        
        return clusters
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics about the clustering quality.
        
        Returns:
            Dictionary of metric names and values
            
        Raises:
            RuntimeError: If called before fitting
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer must be fitted before calling get_metrics")
        
        return self.metrics_
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters used for this clusterer.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'algorithm': 'hdbscan',
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'cluster_selection_method': self.cluster_selection_method,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'alpha': self.alpha,
            'algorithm_specific': self.algorithm,
            'leaf_size': self.leaf_size,
            'allow_single_cluster': self.allow_single_cluster
        }
    
    def _calculate_metrics(self, vectors: np.ndarray) -> None:
        """
        Calculate metrics for the clustering result.
        
        Args:
            vectors: The vectors used for clustering
            
        Updates:
            self.metrics_: Dictionary of metric values
        """
        # Initialize metrics dictionary
        self.metrics_ = {
            'num_clusters': len(np.unique(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'num_samples': len(self.labels_),
            'noise_points': np.sum(self.labels_ == -1),
            'noise_percentage': np.sum(self.labels_ == -1) / len(self.labels_) * 100
        }
        
        # Calculate silhouette score if there's more than one cluster and not all points are noise
        if len(np.unique(self.labels_)) > 1 and not np.all(self.labels_ == -1):
            # Remove noise points for metrics calculation
            non_noise_indices = self.labels_ != -1
            
            if np.sum(non_noise_indices) > 1:
                cluster_labels = self.labels_[non_noise_indices]
                cluster_vectors = vectors[non_noise_indices]
                
                # Only calculate if there's more than one cluster after removing noise
                if len(np.unique(cluster_labels)) > 1:
                    try:
                        s_score = silhouette_score(cluster_vectors, cluster_labels, metric=self.metric)
                        self.metrics_['silhouette_score'] = s_score
                        
                        db_score = davies_bouldin_score(cluster_vectors, cluster_labels)
                        self.metrics_['davies_bouldin_score'] = db_score
                        
                        ch_score = calinski_harabasz_score(cluster_vectors, cluster_labels)
                        self.metrics_['calinski_harabasz_score'] = ch_score
                    except Exception as e:
                        # Add error information but continue
                        self.metrics_['metrics_error'] = str(e)
        else:
            # If no valid clusters, set metrics to None
            self.metrics_['silhouette_score'] = None
            self.metrics_['davies_bouldin_score'] = None
            self.metrics_['calinski_harabasz_score'] = None
            
        # Add HDBSCAN-specific metrics if available
        if hasattr(self.clusterer, 'relative_validity_'):
            self.metrics_['hdbscan_validity'] = self.clusterer.relative_validity_

# Test code
if __name__ == "__main__":
    # Create some test data
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Initialize and fit clusterer
    clusterer = HdbscanClusterer(min_cluster_size=3, min_samples=2, metric='euclidean')
    results = clusterer.fit_predict(X)
    
    # Print results
    print(f"Number of clusters found: {results['metrics']['num_clusters']}")
    print(f"Silhouette score: {results['metrics'].get('silhouette_score', 'N/A')}")
    print(f"Noise percentage: {results['metrics']['noise_percentage']:.2f}%")
    
    # Get cluster information
    clusters = results['clusters']
    print(f"\nCluster details:")
    for cluster in clusters:
        if cluster['id'] == -1:
            print(f"  Noise points: {cluster['size']}")
        else:
            print(f"  Cluster {cluster['id']}: {cluster['size']} points")
    
    # Test prediction
    new_points = np.random.randn(5, X.shape[1])
    predicted_labels = clusterer.predict(new_points)
    print(f"\nPredicted labels for new points: {predicted_labels}")
