"""
KMeans clusterer implementation for the Clustering module.
Provides centroid-based clustering optimized for embedding vectors.
"""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.dirname(current_dir)
clustering_dir = os.path.dirname(embedding_dir)
sys.path.append(os.path.dirname(clustering_dir))

# Import base clusterer interface
from Clustering.base_clusterer import BaseClusterer

class KMeansClusterer(BaseClusterer):
    """
    Centroid-based clustering using KMeans algorithm.
    
    KMeans is well-suited for embedding vectors when:
    - You want to ensure all points are assigned to a cluster
    - You want approximately equal-sized clusters
    - You know the number of clusters in advance
    - You want to minimize within-cluster variance
    
    Unlike HDBSCAN, KMeans does not identify noise points and assigns every point to a cluster.
    """
    
    def __init__(self, 
                n_clusters: int = 8,
                init: str = 'k-means++',
                n_init: int = 10,
                max_iter: int = 300,
                tol: float = 1e-4,
                random_state: Optional[int] = None,
                algorithm: str = 'lloyd'):
        """
        Initialize KMeans clusterer with specified parameters.
        
        Args:
            n_clusters: Number of clusters to form
            init: Method for initialization ('k-means++', 'random', or ndarray)
            n_init: Number of time the k-means algorithm will be run with different centroid seeds
            max_iter: Maximum number of iterations for a single run
            tol: Relative tolerance for convergence
            random_state: Random state for reproducibility
            algorithm: K-means algorithm to use ('lloyd', 'elkan', 'auto', 'full')
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.algorithm = algorithm
        
        # Model will be initialized during fit
        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.cluster_sizes_ = None
        self.inertia_ = None
        
    def fit(self, X: np.ndarray) -> 'KMeansClusterer':
        """
        Perform KMeans clustering on the input data.
        
        Args:
            X: Array of shape (n_samples, n_features)
            
        Returns:
            Self with fitted model
        """
        # Initialize and fit KMeans model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm
        )
        
        self.model.fit(X)
        
        # Store cluster information
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        self.cluster_sizes_ = {label: count for label, count in zip(unique_labels, counts)}
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X: Array of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples,) with cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")
        
        return self.model.predict(X)
    
    def fit_predict(self, X: np.ndarray, texts: List[str] = None) -> Dict[str, Any]:
        """
        Compute cluster centers and predict cluster index for each sample.
        Matches the BaseClusterer interface which requires handling both vectors and texts.
        
        Args:
            X: Array of shape (n_samples, n_features)
            texts: Optional list of text strings (ignored in KMeans, used in other clusterers)
            
        Returns:
            Dictionary with 'labels' array and 'clusters' dictionary mapping cluster labels to point indices
        """
        self.fit(X)
        
        # Create clusters dictionary mapping cluster label to list of point indices
        clusters = {}
        for i, label in enumerate(self.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Return both labels and clusters in dictionary format expected by pipeline
        return {
            "labels": self.labels_,
            "clusters": clusters
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for this clusterer.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'algorithm': self.algorithm
        }
    
    def set_params(self, **params) -> 'KMeansClusterer':
        """
        Set parameters for this clusterer.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self with updated parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get the size of each cluster.
        
        Returns:
            Dictionary mapping cluster label to size
        """
        if self.cluster_sizes_ is None:
            raise ValueError("Model not fitted yet. Call fit() before get_cluster_sizes().")
        
        return self.cluster_sizes_
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get the cluster centers.
        
        Returns:
            Array of shape (n_clusters, n_features) with cluster centers
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() before get_cluster_centers().")
        
        return self.cluster_centers_
        
    def get_clusters(self) -> Dict[int, List[int]]:
        """
        Get clusters as a dictionary mapping cluster IDs to lists of point indices.
        This is a required method from the BaseClusterer interface.
        
        Returns:
            Dictionary mapping cluster ID to list of point indices
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() before get_clusters().")
            
        clusters = {}
        for i, label in enumerate(self.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
            
        return clusters
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get clustering evaluation metrics.
        This is a required method from the BaseClusterer interface.
        
        Returns:
            Dictionary of metric names to values
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() before get_metrics().")
            
        return {
            "inertia": self.inertia_,
            "n_clusters": self.n_clusters,
            "n_points_assigned": len(self.labels_),  # All points are assigned in KMeans
            "noise_ratio": 0.0  # KMeans doesn't produce noise points
        }
