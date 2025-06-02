"""
Preprocessing utilities for embedding vectors used in clustering.
Provides normalization, outlier detection, and dimensionality reduction functions.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats

class EmbeddingPreprocessor:
    """
    Preprocessing utilities for embedding vectors to optimize clustering.
    
    This class provides methods to prepare embedding vectors for clustering,
    including normalization, outlier detection, and dimensionality reduction.
    """
    
    @staticmethod
    def normalize(vectors: np.ndarray, norm: str = 'l2') -> np.ndarray:
        """
        Normalize embedding vectors using the specified norm.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            norm: Normalization type ('l1', 'l2', or 'max')
            
        Returns:
            Normalized embedding vectors
            
        Raises:
            ValueError: If inputs are invalid or normalization fails
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if len(vectors) == 0:
            raise ValueError("vectors cannot be empty")
        
        try:
            return sk_normalize(vectors, norm=norm, axis=1)
        except Exception as e:
            raise ValueError(f"Normalization failed: {str(e)}")
    
    @staticmethod
    def detect_outliers(
        vectors: np.ndarray, 
        method: str = 'zscore', 
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outliers in embedding vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection (e.g., z-score > 3.0)
            
        Returns:
            Boolean mask with True for inliers and False for outliers
            
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if len(vectors) == 0:
            raise ValueError("vectors cannot be empty")
        
        if method.lower() == 'zscore':
            # Compute pairwise distances to mean vector
            mean_vector = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - mean_vector, axis=1)
            
            # Compute z-scores of distances
            z_scores = stats.zscore(distances)
            
            # Return mask of non-outliers
            return np.abs(z_scores) <= threshold
            
        elif method.lower() == 'iqr':
            # Compute pairwise distances to mean vector
            mean_vector = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - mean_vector, axis=1)
            
            # Compute IQR
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Return mask of non-outliers
            return (distances >= lower_bound) & (distances <= upper_bound)
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
    
    @staticmethod
    def remove_outliers(
        vectors: np.ndarray, 
        data: Optional[List[Any]] = None,
        method: str = 'zscore', 
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, Optional[List[Any]], np.ndarray]:
        """
        Remove outliers from embedding vectors and associated data.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            data: Optional associated data for each vector
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple containing:
                - Filtered embedding vectors
                - Filtered data (if provided)
                - Boolean mask with True for inliers and False for outliers
                
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        # Get outlier mask
        inlier_mask = EmbeddingPreprocessor.detect_outliers(vectors, method, threshold)
        
        # Filter vectors
        filtered_vectors = vectors[inlier_mask]
        
        # Filter data if provided
        filtered_data = None
        if data is not None:
            if len(data) != len(vectors):
                raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but data has {len(data)} items")
            filtered_data = [d for i, d in enumerate(data) if inlier_mask[i]]
        
        return filtered_vectors, filtered_data, inlier_mask
    
    @staticmethod
    def reduce_dimensions(
        vectors: np.ndarray, 
        method: str = 'pca', 
        n_components: int = 50,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionality of embedding vectors.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to keep
            **kwargs: Additional parameters for the specific method
            
        Returns:
            Reduced dimensionality vectors
            
        Raises:
            ValueError: If inputs are invalid or method is unsupported
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if len(vectors) == 0:
            raise ValueError("vectors cannot be empty")
        
        if vectors.shape[1] <= n_components:
            return vectors  # No need to reduce dimensions
        
        if method.lower() == 'pca':
            pca = PCA(n_components=n_components, **kwargs)
            return pca.fit_transform(vectors)
            
        elif method.lower() == 'tsne':
            # Default parameters for t-SNE
            tsne_params = {
                'perplexity': min(30, len(vectors) - 1),  # Adjust perplexity based on dataset size
                'n_iter': 1000,
                'random_state': 42
            }
            tsne_params.update(kwargs)  # Update with user-provided parameters
            
            # Initialize and fit t-SNE
            tsne = TSNE(n_components=n_components, **tsne_params)
            return tsne.fit_transform(vectors)
            
        elif method.lower() == 'umap':
            # Default parameters for UMAP
            umap_params = {
                'n_neighbors': min(15, len(vectors) - 1),  # Adjust neighbors based on dataset size
                'min_dist': 0.1,
                'random_state': 42
            }
            umap_params.update(kwargs)  # Update with user-provided parameters
            
            # Initialize and fit UMAP
            reducer = umap.UMAP(n_components=n_components, **umap_params)
            return reducer.fit_transform(vectors)
            
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    @staticmethod
    def prepare_vectors_for_clustering(
        vectors: np.ndarray,
        data: Optional[List[Any]] = None,
        normalize_method: str = 'l2',
        remove_outliers_method: Optional[str] = None,
        outlier_threshold: float = 3.0,
        reduce_dimensions_method: Optional[str] = None,
        n_components: int = 50,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[List[Any]], Dict[str, Any]]:
        """
        Prepare embedding vectors for clustering with a complete pipeline.
        
        Args:
            vectors: Array of embedding vectors with shape (n_samples, n_features)
            data: Optional associated data for each vector
            normalize_method: Normalization method ('l1', 'l2', 'max')
            remove_outliers_method: Outlier removal method ('zscore', 'iqr', None for no removal)
            outlier_threshold: Threshold for outlier detection
            reduce_dimensions_method: Dimensionality reduction method ('pca', 'tsne', 'umap', None for no reduction)
            n_components: Number of components for dimensionality reduction
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Tuple containing:
                - Processed embedding vectors
                - Processed data (if provided)
                - Dictionary with processing metadata
                
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors must be a numpy array")
        
        if len(vectors.shape) != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        
        if len(vectors) == 0:
            raise ValueError("vectors cannot be empty")
        
        if data is not None and len(data) != len(vectors):
            raise ValueError(f"Length mismatch: vectors has {len(vectors)} items, but data has {len(data)} items")
        
        # Initialize metadata
        metadata = {
            'original_shape': vectors.shape,
            'preprocessing_steps': []
        }
        
        # Step 1: Normalize
        processed_vectors = EmbeddingPreprocessor.normalize(vectors, norm=normalize_method)
        metadata['preprocessing_steps'].append(f"Normalized with {normalize_method} norm")
        
        # Step 2: Remove outliers (if requested)
        if remove_outliers_method is not None:
            processed_vectors, processed_data, inlier_mask = EmbeddingPreprocessor.remove_outliers(
                processed_vectors,
                data,
                method=remove_outliers_method,
                threshold=outlier_threshold
            )
            outliers_removed = len(vectors) - len(processed_vectors)
            metadata['preprocessing_steps'].append(
                f"Removed {outliers_removed} outliers using {remove_outliers_method} method with threshold {outlier_threshold}"
            )
            metadata['outliers_removed'] = outliers_removed
            metadata['outliers_percentage'] = (outliers_removed / len(vectors)) * 100
        else:
            processed_data = data
            metadata['preprocessing_steps'].append("No outlier removal performed")
            metadata['outliers_removed'] = 0
            metadata['outliers_percentage'] = 0
        
        # Step 3: Reduce dimensions (if requested)
        if reduce_dimensions_method is not None:
            processed_vectors = EmbeddingPreprocessor.reduce_dimensions(
                processed_vectors,
                method=reduce_dimensions_method,
                n_components=n_components,
                **kwargs
            )
            metadata['preprocessing_steps'].append(
                f"Reduced dimensions from {metadata['original_shape'][1]} to {n_components} using {reduce_dimensions_method}"
            )
        else:
            metadata['preprocessing_steps'].append("No dimensionality reduction performed")
        
        metadata['final_shape'] = processed_vectors.shape
        
        return processed_vectors, processed_data, metadata


# Simple usage example
if __name__ == "__main__":
    # Create test data
    import numpy as np
    np.random.seed(42)
    vectors = np.random.randn(100, 300)  # 100 vectors with 300 dimensions
    data = [f"item_{i}" for i in range(100)]
    
    # Add some outliers
    vectors[0] = vectors[0] * 10  # Make first vector an outlier
    vectors[1] = vectors[1] * -10  # Make second vector an outlier
    
    # Process vectors
    processed_vectors, processed_data, metadata = EmbeddingPreprocessor.prepare_vectors_for_clustering(
        vectors=vectors,
        data=data,
        normalize_method='l2',
        remove_outliers_method='zscore',
        outlier_threshold=2.5,
        reduce_dimensions_method='pca',
        n_components=50
    )
    
    # Print results
    print(f"Original vectors shape: {vectors.shape}")
    print(f"Processed vectors shape: {processed_vectors.shape}")
    print(f"Original data length: {len(data)}")
    print(f"Processed data length: {len(processed_data)}")
    print(f"Metadata: {metadata}")
    
    # Simple test of normalization
    print("\nTesting normalization:")
    normalized = EmbeddingPreprocessor.normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    print(f"Norms after L2 normalization (should be all 1.0): {norms[:5]}...")
