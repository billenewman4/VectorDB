import os
import numpy as np
from typing import List, Dict, Any, Optional
import requests

# Try to load from project config, but make it optional
try:
    from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
except ImportError:
    OPENAI_API_KEY = None
    OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

class OpenAIEmbedder:
    """Handles embedding generation using OpenAI models via direct API calls."""
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder with API key and model name.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model_name: OpenAI embedding model to use
        """
        # Use provided API key or get from environment or config
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set it in .env file or pass it directly.")
          
        # Import requests only when needed
        import requests
        self.requests = requests
          
        self.model = model_name
        print(f"Initialized OpenAI embedder with model: {model_name}")
    
    def name(self):
        """Return the name of this embedding function (required for ChromaDB)."""
        return f"openai_{self.model}"
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string using direct API call."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "input": text,
                "model": self.model
            }
            response = self.requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return np.array(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts (ChromaDB interface requires 'input' parameter)."""
        if not input:
            return []
            
        try:
            # Handle batching for cost efficiency
            batch_size = 100  # Adjust as needed
            all_embeddings = []
            
            for i in range(0, len(input), batch_size):
                batch = input[i:i+batch_size]
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                payload = {
                    "input": batch,
                    "model": self.model
                }
                response = self.requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                result = response.json()
                # Extract embeddings in the same order as input
                batch_embeddings = [np.array(item["embedding"]) for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise
