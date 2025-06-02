"""
Base clusterer interface for the Clustering module.
Defines the common API that all clustering implementations must follow.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np

class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.
    
    This class defines the common interface that all clustering implementations
    must implement, ensuring consistent behavior across different algorithms.
    """
    
    @abstractmethod
    def fit(self, vectors: np.ndarray, data: Optional[List[Any]] = None) -> 'BaseClusterer':
        """
        Fit the clusterer to the input vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            data: Optional additional data associated with each vector
            
        Returns:
            self: The fitted clusterer instance
            
        Raises:
            ValueError: If inputs are invalid
        """
        pass
    
    @abstractmethod
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for the input vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            
        Returns:
            Array of cluster labels for each input vector
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If called before fitting
        """
        pass
    
    def fit_predict(self, vectors: np.ndarray, data: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Fit the clusterer and predict cluster labels in one operation.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            data: Optional additional data associated with each vector
            
        Returns:
            Dictionary containing:
                - 'labels': Array of cluster labels
                - 'clusters': List of cluster information
                - 'metrics': Dictionary of clustering metrics
                - 'params': Parameters used for clustering
                
        Raises:
            ValueError: If inputs are invalid
        """
        self.fit(vectors, data)
        labels = self.predict(vectors)
        return {
            'labels': labels,
            'clusters': self.get_clusters(),
            'metrics': self.get_metrics(),
            'params': self.get_params()
        }
    
    @abstractmethod
    def get_clusters(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about each cluster.
        
        Returns:
            List of dictionaries, one per cluster, containing:
                - 'id': Cluster identifier
                - 'size': Number of points in the cluster
                - 'centroid': Centroid of the cluster
                - 'members': Indices of cluster members
                - Additional algorithm-specific information
                
        Raises:
            RuntimeError: If called before fitting
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics about the clustering quality.
        
        Returns:
            Dictionary of metric names and values
            
        Raises:
            RuntimeError: If called before fitting
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters used for this clusterer.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    def visualize(self, dim_reduction: str = 'tsne', **kwargs) -> Dict[str, Any]:
        """
        Generate visualization data for the clusters.
        
        This is a default implementation that should be overridden
        by subclasses if they have specific visualization needs.
        
        Args:
            dim_reduction: Dimensionality reduction method ('tsne', 'umap', 'pca')
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Dictionary containing visualization data:
                - 'coords': 2D or 3D coordinates for plotting
                - 'labels': Cluster labels
                - 'method': Method used for dimensionality reduction
                
        Raises:
            RuntimeError: If called before fitting
            ValueError: If requested method is not supported
        """
        raise NotImplementedError(
            "Visualization is not implemented in the base class. "
            "Use the visualization utilities or override this method in a subclass."
        )

def check_is_clusterer(obj: Any) -> bool:
    """
    Check if an object is a valid clusterer.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is a valid clusterer, False otherwise
    """
    if not isinstance(obj, BaseClusterer):
        return False
    
    # Check that all abstract methods are implemented
    required_methods = [
        'fit', 'predict', 'get_clusters', 'get_metrics', 'get_params'
    ]
    
    for method in required_methods:
        if not hasattr(obj, method) or not callable(getattr(obj, method)):
            return False
    
    return True
