"""
Vector normalization utilities for embedding vectors.

This module provides functions for normalizing embedding vectors to unit length,
which is important for certain similarity calculations and clustering algorithms.
"""

import numpy as np
from typing import List, Union, Optional


def normalize_vectors(vectors: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 norm).
    
    This function normalizes each vector to have a Euclidean norm (L2 norm) of 1.
    This is important for cosine similarity calculations and certain clustering algorithms.
    
    Args:
        vectors: Either a 2D numpy array of shape (n_vectors, n_dimensions)
                or a list of 1D numpy arrays
                
    Returns:
        np.ndarray: Normalized vectors with the same shape as input
        
    Raises:
        ValueError: If vectors is None, empty, or contains vectors with zero norm
        TypeError: If vectors is not a numpy array or list of numpy arrays
    """
    # Input validation
    if vectors is None:
        raise ValueError("Vectors cannot be None")
        
    # Handle list of numpy arrays
    if isinstance(vectors, list):
        if not vectors:
            raise ValueError("Vectors list cannot be empty")
            
        # Validate that all elements are numpy arrays
        for i, vec in enumerate(vectors):
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"Vector at index {i} is not a numpy array: {type(vec)}")
                
        # Convert list to 2D array
        vectors = np.vstack(vectors)
    
    # Handle numpy array
    elif isinstance(vectors, np.ndarray):
        if vectors.size == 0:
            raise ValueError("Vectors array cannot be empty")
            
        # Ensure 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {vectors.ndim}D")
    else:
        raise TypeError(f"Expected numpy array or list of numpy arrays, got {type(vectors)}")
    
    # Calculate norms
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Check for zero norms
    zero_indices = np.where(norms == 0)[0]
    if zero_indices.size > 0:
        raise ValueError(f"Vector(s) at indices {zero_indices} have zero norm and cannot be normalized")
    
    # Normalize
    normalized = vectors / norms
    
    return normalized


def check_normalized(vectors: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if vectors are already normalized to unit length.
    
    Args:
        vectors: 2D numpy array of shape (n_vectors, n_dimensions)
        tolerance: Tolerance for numerical precision issues
        
    Returns:
        bool: True if all vectors are normalized, False otherwise
        
    Raises:
        ValueError: If vectors is None or empty
        TypeError: If vectors is not a numpy array
    """
    # Input validation
    if vectors is None:
        raise ValueError("Vectors cannot be None")
        
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(vectors)}")
        
    if vectors.size == 0:
        raise ValueError("Vectors array cannot be empty")
        
    # Ensure 2D array
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    elif vectors.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got {vectors.ndim}D")
    
    # Calculate norms
    norms = np.linalg.norm(vectors, axis=1)
    
    # Check if all norms are approximately 1
    return np.all(np.abs(norms - 1.0) < tolerance)


if __name__ == "__main__":
    # Test the normalization functions
    try:
        print("Testing vector normalization...")
        
        # Create test vectors
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Normalize vectors
        normalized = normalize_vectors(vectors)
        
        # Check shapes
        print(f"Original shape: {vectors.shape}")
        print(f"Normalized shape: {normalized.shape}")
        
        # Check norms
        original_norms = np.linalg.norm(vectors, axis=1)
        normalized_norms = np.linalg.norm(normalized, axis=1)
        
        print(f"Original norms: {original_norms}")
        print(f"Normalized norms: {normalized_norms}")
        
        # Verify normalization
        assert check_normalized(normalized), "Vectors are not properly normalized"
        
        # Test list input
        vector_list = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])]
        normalized_list = normalize_vectors(vector_list)
        
        print(f"Normalized from list shape: {normalized_list.shape}")
        assert check_normalized(normalized_list), "List vectors are not properly normalized"
        
        print("Vector normalization tests passed!")
    except Exception as e:
        print(f"Error testing vector normalization: {e}")
