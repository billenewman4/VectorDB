import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Try to load from project config, but make it optional
try:
    from src.config import SENTENCE_TRANSFORMER_MODEL
except ImportError:
    SENTENCE_TRANSFORMER_MODEL = 'all-mpnet-base-v2'

class LocalEmbedder:
    """Handles embedding generation using local models via sentence-transformers."""
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initialize local embedder with model name.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            print(f"Initialized local embedder with model: {model_name}")
        except Exception as e:
            print(f"Error initializing sentence-transformer model: {e}")
            raise
    
    def name(self):
        """Return the name of this embedding function."""
        return f"sentence_transformer_{self.model_name}"
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
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
