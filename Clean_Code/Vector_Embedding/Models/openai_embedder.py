"""
OpenAI embedder using OpenAI's API for embeddings.

This module implements an embedding model that uses OpenAI's API to generate
high-quality embeddings for text data.
"""

import os
import sys
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Add parent directories to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
vector_embedding_dir = os.path.dirname(models_dir)
sys.path.append(os.path.dirname(vector_embedding_dir))

# Import base embedder using relative import
from .base_embedder import BaseEmbedder

# Try to load from project config
try:
    from src.config import OPENAI_API_KEY
except ImportError:
    # Don't set a default, force proper configuration
    OPENAI_API_KEY = None


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding model that uses OpenAI's API for generating embeddings.
    
    This implementation makes direct API calls to OpenAI's embedding endpoint
    to generate high-quality embeddings for text data.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder with API key and model name.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model_name: OpenAI embedding model to use (default: text-embedding-3-small)
            
        Raises:
            ValueError: If no API key is found or provided
        """
        # Use provided API key or get from environment or config
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set it in .env file or pass it directly.")
          
        self.model = model_name
        print(f"Initialized OpenAI embedder with model: {model_name}")
    
    def name(self) -> str:
        """
        Return a unique identifier for this embedder.
        
        Returns:
            str: Identifier in the format 'openai_[model_name]'
        """
        return f"openai_{self.model}"
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string using direct API call.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
            requests.RequestException: If API request fails
            KeyError: If response format is unexpected
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Text for embedding must be a non-empty string")
            
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
            
            # Verify response format
            if "data" not in result or not result["data"] or "embedding" not in result["data"][0]:
                raise KeyError(f"Unexpected API response format: {result}")
                
            embedding = result["data"][0]["embedding"]
            return np.array(embedding)
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            raise
        except KeyError as e:
            print(f"Error parsing API response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error generating embedding: {e}")
            raise
    
    def batch_embed(self, texts: List[str], batch_size: int = 100, retry_count: int = 3, retry_delay: int = 5) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching, retries, and error handling.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in each API call
            retry_count: Number of times to retry failed API calls
            retry_delay: Seconds to wait between retries
            
        Returns:
            List[np.ndarray]: List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains invalid entries
            requests.RequestException: If API requests repeatedly fail
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")
            
        # Validate all texts
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} is not a string: {type(text)}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            retry = 0
            success = False
            
            while retry <= retry_count and not success:
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
                    
                    # Verify response format
                    if "data" not in result or len(result["data"]) != len(batch):
                        raise KeyError(f"Unexpected API response format or length mismatch: {result}")
                    
                    # Extract embeddings from response
                    batch_embeddings = [np.array(item["embedding"]) for item in result["data"]]
                    all_embeddings.extend(batch_embeddings)
                    success = True
                    
                except (requests.RequestException, KeyError) as e:
                    retry += 1
                    if retry > retry_count:
                        print(f"Error after {retry_count} retries: {e}")
                        raise
                    else:
                        print(f"Retry {retry}/{retry_count} after error: {e}")
                        time.sleep(retry_delay)
        
        return all_embeddings
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts (ChromaDB interface requires 'input' parameter).
        
        Args:
            input: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
            
        Raises:
            ValueError: If input is invalid
            requests.RequestException: If API requests fail
        """
        return self.batch_embed(input)


if __name__ == "__main__":
    # Test the embedder
    try:
        print("Testing OpenAIEmbedder...")
        
        # Create embedder - this will fail if no API key is available
        embedder = OpenAIEmbedder()
        print(f"Created embedder: {embedder.name()}")
        
        # Test single embedding
        text = "This is a test sentence for embedding"
        embedding = embedder.embed_query(text)
        print(f"Single embedding shape: {embedding.shape}")
        
        # Test batch embedding with a small batch
        texts = ["First test sentence", "Second test sentence"]
        embeddings = embedder(texts)
        print(f"Batch embedding count: {len(embeddings)}")
        print(f"First embedding shape: {embeddings[0].shape}")
        
        print("OpenAIEmbedder tests passed!")
    except Exception as e:
        print(f"Error testing OpenAIEmbedder: {e}")
