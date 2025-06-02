"""
Local embedder using sentence-transformers models.

This module implements a local embedding model using sentence-transformers,
which can run entirely on the user's machine without API calls.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
vector_embedding_dir = os.path.dirname(models_dir)
sys.path.append(os.path.dirname(vector_embedding_dir))

# Import base embedder using relative import
from .base_embedder import BaseEmbedder

# Try to load from project config, but make it optional
try:
    from src.config import SENTENCE_TRANSFORMER_MODEL
except ImportError:
    # Throw error
    raise ImportError("SENTENCE_TRANSFORMER_MODEL not found in config")


class LocalEmbedder(BaseEmbedder):
    """
    Embedding model that uses local sentence-transformers models.
    
    This implementation uses the sentence-transformers library to generate 
    embeddings locally, without requiring API calls.
    """
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initialize local embedder with model name.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default is 'all-mpnet-base-v2', which produces higher
                        quality embeddings than the previously used 'all-MiniLM-L6-v2'.
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            print(f"Initialized local embedder with model: {model_name}")
        except Exception as e:
            print(f"Error initializing sentence-transformer model: {e}")
            raise
    
    def name(self) -> str:
        """
        Return a unique identifier for this embedder.
        
        Returns:
            str: Identifier in the format 'sentence_transformer_[model_name]'
        """
        return f"sentence_transformer_{self.model_name}"
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        if not text or not isinstance(text, str):
            # Return zero vector for empty or non-string input
            # Get the default dimension from the model
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim)
            
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Handles batching automatically for efficient processing of
        large input lists.
        
        Args:
            input: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
            
        Raises:
            Exception: If batch embedding generation fails
        """
        if not input:
            return []
            
        try:
            # Process in a single batch for small inputs
            if len(input) <= 100:
                embeddings = self.model.encode(input, convert_to_numpy=True)
                # Convert to list of numpy arrays
                if len(input) == 1:
                    # If single input, make sure it's still returned as a list
                    return [embeddings]
                return [np.array(emb) for emb in embeddings]
                
            # Handle batching for larger inputs
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(input), batch_size):
                batch = input[i:i+batch_size]
                # Safety check - ensure all are strings
                batch = [str(text) if text is not None else "" for text in batch]
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                
                # Add each embedding as a numpy array
                if len(batch) == 1:
                    all_embeddings.append(batch_embeddings)
                else:
                    all_embeddings.extend([np.array(emb) for emb in batch_embeddings])
                    
            return all_embeddings
            
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise


if __name__ == "__main__":
    # Test the embedder
    try:
        print("Testing LocalEmbedder...")
        
        # Create embedder
        embedder = LocalEmbedder()
        print(f"Created embedder: {embedder.name()}")
        
        # Test single embedding
        text = "This is a test sentence for embedding"
        embedding = embedder.embed_query(text)
        print(f"Single embedding shape: {embedding.shape}")
        
        # Test batch embedding
        texts = ["First test sentence", "Second test sentence", "Third test sentence"]
        embeddings = embedder(texts)
        print(f"Batch embedding count: {len(embeddings)}")
        print(f"First embedding shape: {embeddings[0].shape}")
        
        print("LocalEmbedder tests passed!")
    except Exception as e:
        print(f"Error testing LocalEmbedder: {e}")
