import os
import numpy as np
import requests
from typing import List, Dict, Any, Optional

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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set it in .env file or pass it directly.")
          
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
            response = requests.post(
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
    
    def batch_embed(self, texts: List[str], batch_size: int = 100, retry_count: int = 3, retry_delay: int = 5) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching, retries, and error handling."""
        all_embeddings = []
        import time
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            retry = 0
            while retry <= retry_count:
                try:
                    headers = {
                        "Content-Type": "application/json", 
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    payload = {
                        "input": batch,
                        "model": self.model
                    }
                    response = requests.post(
                        "https://api.openai.com/v1/embeddings",
                        headers=headers, 
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract embeddings from response
                    batch_embeddings = [np.array(item["embedding"]) for item in result["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    retry += 1
                    if retry > retry_count:
                        print(f"Error after {retry_count} retries: {e}")
                        # Return empty embeddings for failed batch
                        all_embeddings.extend([np.zeros(1536)] * len(batch))  # Default to 1536 dim for OpenAI embeddings
                    else:
                        print(f"Retry {retry}/{retry_count} after error: {e}")
                        time.sleep(retry_delay)
        
        return all_embeddings
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts (ChromaDB interface requires 'input' parameter)."""
        return self.batch_embed(input)
