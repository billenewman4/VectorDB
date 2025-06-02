"""
Cross-encoder wrapper for hierarchical clustering.

This module provides a wrapper around the SentenceTransformers CrossEncoder
to add the compute_similarity method required by the ClusterRefiner.
"""

import numpy as np
from typing import List, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossEncoderWrapper:
    """
    Wrapper for SentenceTransformers CrossEncoder to add compute_similarity method.
    
    This class enhances the standard CrossEncoder with methods needed for
    cluster refinement in the hierarchical clustering pipeline.
    """
    
    def __init__(self, cross_encoder):
        """
        Initialize with a SentenceTransformers CrossEncoder instance.
        
        Args:
            cross_encoder: An initialized CrossEncoder from sentence_transformers
        """
        self.cross_encoder = cross_encoder
        # Store the model_name attribute
        self.model_name = getattr(cross_encoder, 'model_name', str(cross_encoder))
        logger.info(f"Initialized CrossEncoderWrapper with model: {self.model_name}")
    
    def compute_similarity(self, 
                           texts1=None, 
                           texts2=None, 
                           queries=None,
                           passages=None,
                           batch_size: int = 32) -> np.ndarray:
        """
        Compute similarity scores between pairs of texts using the cross-encoder.
        Supports both (texts1, texts2) and (queries, passages) parameter combinations.
        
        Args:
            texts1: First list of texts (or None if using queries/passages)
            texts2: Second list of texts (or None if using queries/passages)
            queries: List of query texts (or None if using texts1/texts2)
            passages: List of passage texts (or None if using texts1/texts2)
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of similarity scores in range [0, 1]
        """
        # Support both parameter combinations
        if queries is not None and passages is not None:
            # Use queries and passages parameters
            first_texts = queries
            second_texts = passages
        elif texts1 is not None and texts2 is not None:
            # Use texts1 and texts2 parameters
            first_texts = texts1
            second_texts = texts2
        else:
            raise ValueError("Either (texts1, texts2) or (queries, passages) must be provided")
        
        if len(first_texts) != len(second_texts):
            raise ValueError(f"Both text lists must have same length, got {len(first_texts)} and {len(second_texts)}")
        
        # Create sentence pairs for cross-encoder
        sentence_pairs = [[t1, t2] for t1, t2 in zip(first_texts, second_texts)]
        
        # Get scores from cross-encoder
        logger.info(f"Computing similarity for {len(sentence_pairs)} text pairs")
        similarity_scores = self.cross_encoder.predict(sentence_pairs, batch_size=batch_size)
        
        # Normalize scores to [0, 1] range if they aren't already
        # Most cross-encoders output logits that need to be converted to probabilities
        if np.min(similarity_scores) < 0 or np.max(similarity_scores) > 1:
            logger.info("Normalizing cross-encoder scores to [0, 1] range")
            similarity_scores = 1 / (1 + np.exp(-similarity_scores))  # sigmoid
        
        return similarity_scores
    
    def compute_pairwise_similarity(self, 
                                   texts: List[str], 
                                   batch_size: int = 32) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            n×n NumPy array of pairwise similarities
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        # Build pairs for all i,j combinations
        pairs = []
        pair_indices = []
        
        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                pairs.append([texts[i], texts[j]])
                pair_indices.append((i, j))
        
        logger.info(f"Computing pairwise similarity for {len(pairs)} text pairs")
        
        # Compute similarities in batches
        all_scores = []
        for idx in range(0, len(pairs), batch_size):
            batch_pairs = pairs[idx:idx + batch_size]
            batch_scores = self.cross_encoder.predict(batch_pairs)
            if isinstance(batch_scores, list):
                batch_scores = np.array(batch_scores)
            all_scores.append(batch_scores)
        
        # Combine batches
        if all_scores:
            all_scores = np.concatenate(all_scores)
            
            # Normalize scores if needed
            if np.min(all_scores) < 0 or np.max(all_scores) > 1:
                all_scores = 1 / (1 + np.exp(-all_scores))  # sigmoid
            
            # Fill the similarity matrix (both upper and lower triangles)
            for idx, (i, j) in enumerate(pair_indices):
                similarity_matrix[i, j] = all_scores[idx]
                similarity_matrix[j, i] = all_scores[idx]  # Mirror
        
        return similarity_matrix
