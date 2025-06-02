"""
Sentence Transformer Encoder for vector embeddings.

This module provides a simplified interface for generating embeddings using
sentence-transformers models, designed specifically for the hierarchical
clustering pipeline.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerEncoder:
    """
    Sentence Transformer encoder for generating embeddings.
    
    This class provides a simplified interface to generate embeddings using
    sentence-transformers models, with batching and progress tracking.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the encoder with the specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        self._load_model()
        
    def _load_model(self):
        """
        Load the sentence-transformers model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully: {self.model_name}")
        except ImportError:
            logger.error("Failed to import sentence_transformers. Please install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            NumPy array of embeddings
        """
        if self._model is None:
            self._load_model()
            
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a large list of texts in batches with progress tracking.
        
        Args:
            texts: List of texts to encode
            batch_size: Number of texts to encode in each batch
            
        Returns:
            NumPy array of embeddings
        """
        if self._model is None:
            self._load_model()
            
        num_texts = len(texts)
        if num_texts == 0:
            return np.array([])
            
        embeddings_list = []
        
        # Process in batches with progress bar
        logger.info(f"Encoding {num_texts} texts in batches of {batch_size}...")
        for i in tqdm(range(0, num_texts, batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._model.encode(batch_texts, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
            
        # Combine all batches
        embeddings = np.vstack(embeddings_list)
        logger.info(f"Encoding complete: {embeddings.shape}")
        
        return embeddings


if __name__ == "__main__":
    # Simple test of the encoder
    encoder = SentenceTransformerEncoder()
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for encoding.",
        "Let's see how well this works."
    ]
    
    embeddings = encoder.encode_batch(test_texts)
    print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    print(f"Sample embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
